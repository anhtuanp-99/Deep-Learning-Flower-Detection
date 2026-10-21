"""Tìm các ảnh model đoán sai để in ra demo với camera (công cụ phụ, không dùng cho báo cáo).

    python find_errors.py                  # quét tập test
    python find_errors.py --split all      # quét cả train + validation + test
    python find_errors.py --min-conf 0.6   # chỉ lấy ca sai mà model "tự tin" (>= 60%)

Kết quả: in danh sách (đường dẫn ảnh cụ thể) ra màn hình và chép ảnh sai vào thư mục demo_errors/.
"""
import argparse
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import common as C
from train import FlowerDataset   # dùng lại Dataset của train.py

OUT_DIR = C.ROOT / "demo_errors"


def scan_split(model, class_names, device, split):
    """Trả về danh sách ca sai của một tập: dict(split, path, true, pred, conf)."""
    df = pd.read_csv(C.CSV_FILES[split])
    mapping = dict(zip(df["Label"], df["ClassName"]))
    if [mapping[i] for i in sorted(mapping)] != class_names:
        raise SystemExit(f"Thứ tự lớp trong {C.CSV_FILES[split].name} khác với model. Hãy train lại.")

    ds = FlowerDataset(df, C.eval_transform())
    loader = DataLoader(ds, batch_size=C.BATCH_SIZE, num_workers=C.NUM_WORKERS)
    probs = []
    with torch.inference_mode():
        for x, _ in tqdm(loader, desc=f"Quét {split}", leave=False):
            probs.append(torch.softmax(model(x.to(device)), dim=1).cpu().numpy())
    probs = np.concatenate(probs)
    pred, true = probs.argmax(1), np.array(ds.labels)

    errors = [dict(split=split, path=ds.paths[i], true=class_names[true[i]],
                   pred=class_names[pred[i]], conf=float(probs[i, pred[i]]))
              for i in np.where(pred != true)[0]]
    return errors, len(ds)


def rel(path: Path) -> str:
    try:
        return path.relative_to(C.ROOT).as_posix()
    except ValueError:
        return str(path)


def main():
    parser = argparse.ArgumentParser(description="Tìm ảnh model đoán sai")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--min-conf", type=float, default=0.0,
                        help="chỉ giữ ca sai có độ tin cậy >= giá trị này (0-1)")
    args = parser.parse_args()

    device = C.get_device()
    model, class_names = C.load_model(device)
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    errors, scanned = [], 0
    for split in splits:
        errs, n = scan_split(model, class_names, device, split)
        errors += errs
        scanned += n
    print(f"\nĐã quét {scanned} ảnh ({', '.join(splits)}): {len(errors)} ảnh đoán sai "
          f"(accuracy {100 * (1 - len(errors) / scanned):.2f}%)")

    errors = sorted((e for e in errors if e["conf"] >= args.min_conf), key=lambda e: -e["conf"])
    if not errors:
        print("Không có ca sai nào thỏa điều kiện.")
        return

    # --- danh sách: sai mà tự tin nhất lên đầu (app sẽ hiện tên sai thay vì "Không chắc chắn")
    print(f"\n{'#':>3}  {'Thật':<10} {'Đoán':<10} {'Tin cậy':>8}  Đường dẫn")
    for i, e in enumerate(errors, 1):
        print(f"{i:>3}  {e['true']:<10} {e['pred']:<10} {e['conf']:>7.1%}  {rel(e['path'])}")

    pairs = Counter((e["true"], e["pred"]) for e in errors).most_common(5)
    print("\nCặp nhầm nhiều nhất: " + ", ".join(f"{t}→{p} ({n})" for (t, p), n in pairs))

    # --- chép ảnh sai ra thư mục riêng để in
    OUT_DIR.mkdir(exist_ok=True)
    rows = []
    for i, e in enumerate(errors, 1):
        name = f"{i:02d}_that-{e['true']}_doan-{e['pred']}_{round(e['conf'] * 100)}_{e['path'].name}"
        shutil.copy2(e["path"], OUT_DIR / name)
        rows.append({"stt": i, "that": e["true"], "doan": e["pred"], "tin_cay": round(e["conf"], 4),
                     "tap": e["split"], "duong_dan_goc": rel(e["path"]), "file_da_chep": name})
    pd.DataFrame(rows).to_csv(OUT_DIR / "danh_sach.csv", index=False, encoding="utf-8-sig")
    print(f"\nĐã chép {len(errors)} ảnh vào: {OUT_DIR}")
    print("Danh sách đầy đủ: demo_errors/danh_sach.csv")


if __name__ == "__main__":
    main()
