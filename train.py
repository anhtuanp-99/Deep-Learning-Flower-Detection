"""Huấn luyện MobileNetV2 và tự động xuất biểu đồ/báo cáo vào reports/run_<thời gian>/.

    python train.py
"""
import json
import random
import time
from datetime import datetime

import matplotlib

matplotlib.use("Agg")  # chỉ lưu file, không mở cửa sổ
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

import common as C


# ================================================================== DỮ LIỆU
class FlowerDataset(Dataset):
    def __init__(self, df, transform):
        self.paths = [C.IMAGES_DIR / f for f in df["Filename"]]
        self.labels = df["Label"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as img:
            return self.transform(img.convert("RGB")), self.labels[i]


def train_transform():
    return T.Compose([
        T.RandomResizedCrop(C.IMG_SIZE, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(20),
        T.ColorJitter(0.2, 0.2, 0.2),
        T.ToTensor(),
        T.Normalize(C.MEAN, C.STD),
    ])


def load_data(device):
    """Đọc 3 file CSV -> (datasets, loaders, class_names)."""
    dfs = {name: pd.read_csv(path) for name, path in C.CSV_FILES.items()}
    for name, df in dfs.items():
        missing = [f for f in df["Filename"] if not (C.IMAGES_DIR / f).exists()]
        if missing:
            raise SystemExit(f"Tập {name} thiếu {len(missing)} ảnh, ví dụ: {C.IMAGES_DIR / missing[0]}")

    mapping = dict(zip(dfs["train"]["Label"], dfs["train"]["ClassName"]))
    class_names = [mapping[i] for i in sorted(mapping)]   # thứ tự theo Label

    datasets, loaders = {}, {}
    for name, df in dfs.items():
        is_train = name == "train"
        datasets[name] = FlowerDataset(df, train_transform() if is_train else C.eval_transform())
        loaders[name] = DataLoader(datasets[name], batch_size=C.BATCH_SIZE, shuffle=is_train,
                                   num_workers=C.NUM_WORKERS, pin_memory=device.type == "cuda",
                                   persistent_workers=C.NUM_WORKERS > 0)
    return datasets, loaders, class_names


# =============================================================== HUẤN LUYỆN
def run_epoch(model, loader, loss_fn, device, optimizer=None, scaler=None, use_amp=False, freeze_bn=False):
    """Chạy 1 epoch (train nếu có optimizer, ngược lại là validation). Trả về (loss, accuracy)."""
    training = optimizer is not None
    model.train(training)
    if training and freeze_bn:
        model.features.eval()          # backbone đang đóng băng: giữ nguyên BatchNorm
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(x)
                loss = loss_fn(out, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def train_phase(title, epochs, lr, freeze_bn, model, loaders, device, use_amp, history, best):
    """Một giai đoạn huấn luyện. `best` lưu trọng số có val loss thấp nhất xuyên suốt cả 2 giai đoạn."""
    print(f"\n=== {title} ({epochs} epoch, lr={lr}) ===")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        tl, ta = run_epoch(model, loaders["train"], loss_fn, device, optimizer, scaler, use_amp, freeze_bn)
        vl, va = run_epoch(model, loaders["val"], loss_fn, device, use_amp=use_amp)
        for key, value in zip(("train_loss", "train_acc", "val_loss", "val_acc"), (tl, ta, vl, va)):
            history[key].append(value)
        star = ""
        if vl < best["loss"]:
            best["loss"], star = vl, "  *"
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"Epoch {epoch:>2}/{epochs} | train loss {tl:.4f} acc {ta:.4f} | val loss {vl:.4f} acc {va:.4f}{star}")


# ================================================================ BIỂU ĐỒ
def save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_history(history, fine_start, path):
    epochs = range(1, len(history["val_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, title in ((axes[0], "acc", "Độ chính xác (Accuracy)"), (axes[1], "loss", "Sai số (Loss)")):
        ax.plot(epochs, history[f"train_{key}"], marker="o", label="Train")
        ax.plot(epochs, history[f"val_{key}"], marker="o", label="Validation")
        if fine_start:
            ax.axvline(fine_start + 0.5, color="gray", linestyle="--", label="Bắt đầu fine-tune")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
        ax.legend()
    save_fig(fig, path)


def plot_confusion(cm, class_names, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ticks = list(range(len(class_names)))
    ax.set_xticks(ticks, class_names, rotation=45, ha="right")
    ax.set_yticks(ticks, class_names)
    for i in ticks:
        for j in ticks:
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    ax.set_title("Confusion Matrix - tập Test")
    save_fig(fig, path)


def plot_samples(paths, y_true, y_pred, probs, class_names, path, n=12):
    """Lưới ảnh test: ưu tiên ảnh đoán sai (viền đỏ), còn lại là ảnh đoán đúng (xanh)."""
    rng = np.random.default_rng(C.SEED)
    wrong = rng.permutation(np.where(y_true != y_pred)[0])[:n // 2]
    right = rng.permutation(np.where(y_true == y_pred)[0])[:n - len(wrong)]
    idx = list(wrong) + list(right)
    fig, axes = plt.subplots(3, 4, figsize=(13, 10))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), idx):
        with Image.open(paths[i]) as img:
            ax.imshow(img.convert("RGB"))
        ok = y_true[i] == y_pred[i]
        ax.set_title(f"Thật: {class_names[y_true[i]]}\nĐoán: {class_names[y_pred[i]]} ({probs[i, y_pred[i]]:.0%})",
                     fontsize=9, color="#2e7d32" if ok else "#c62828")
    fig.suptitle("Dự đoán mẫu trên tập Test (xanh = đúng, đỏ = sai)")
    save_fig(fig, path)


# ===================================================================== MAIN
def main():
    random.seed(C.SEED)
    np.random.seed(C.SEED)
    torch.manual_seed(C.SEED)
    device = C.get_device()
    use_amp = device.type == "cuda"
    print(f"Thiết bị: {C.device_label(device)} (PyTorch {torch.__version__})")
    if device.type == "cpu":
        print("CẢNH BÁO: không thấy GPU, đang train bằng CPU. Xem docs/cai_dat_gpu.md")

    datasets, loaders, class_names = load_data(device)
    print(f"Các lớp: {class_names} | " + ", ".join(f"{k}: {len(v)} ảnh" for k, v in datasets.items()))

    out_dir = C.REPORTS_DIR / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = C.build_model(len(class_names)).to(device)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best = {"loss": float("inf"), "state": None}
    args = (model, loaders, device, use_amp, history, best)
    t0 = time.time()

    for p in model.features.parameters():                       # giai đoạn 1: đóng băng backbone
        p.requires_grad = False
    train_phase("Giai đoạn 1: train lớp phân loại", C.EPOCHS_HEAD, C.LR_HEAD, True, *args)

    fine_start = None
    if C.EPOCHS_FINE > 0:                                       # giai đoạn 2: mở khóa các block cuối
        fine_start = len(history["val_loss"])
        for i, block in enumerate(model.features):
            for p in block.parameters():
                p.requires_grad = i >= C.FINE_FROM_BLOCK
        train_phase("Giai đoạn 2: fine-tune", C.EPOCHS_FINE, C.LR_FINE, False, *args)
    train_seconds = time.time() - t0

    model.load_state_dict(best["state"])                        # quay về epoch tốt nhất

    # ----- đánh giá trên tập test
    model.eval()
    probs, ys = [], []
    with torch.inference_mode():
        for x, y in tqdm(loaders["test"], desc="Test", leave=False):
            probs.append(torch.softmax(model(x.to(device)), dim=1).cpu().numpy())
            ys.append(y.numpy())
    probs, y_true = np.concatenate(probs), np.concatenate(ys)
    y_pred = probs.argmax(1)
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report_text = classification_report(y_true, y_pred, labels=labels, target_names=class_names, digits=4, zero_division=0)
    report = classification_report(y_true, y_pred, labels=labels, target_names=class_names, output_dict=True, zero_division=0)
    accuracy = float((y_true == y_pred).mean())
    print("\n" + report_text)
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    # ----- lưu kết quả
    plot_history(history, fine_start, out_dir / "training_history.png")
    plot_confusion(cm, class_names, out_dir / "confusion_matrix.png")
    plot_samples(datasets["test"].paths, y_true, y_pred, probs, class_names, out_dir / "sample_predictions.png")
    (out_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps({
        "test_accuracy": accuracy,
        "macro_f1": report["macro avg"]["f1-score"],
        "per_class": {c: report[c] for c in class_names},
        "device": C.device_label(device),
        "epochs_run": len(history["val_loss"]),
        "train_seconds": round(train_seconds, 1),
        "batch_size": C.BATCH_SIZE,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    C.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best["state"], "class_names": class_names}, C.MODEL_PATH)
    print(f"\nĐã lưu mô hình : {C.MODEL_PATH}")
    print(f"Biểu đồ/báo cáo: {out_dir}")


if __name__ == "__main__":
    main()
