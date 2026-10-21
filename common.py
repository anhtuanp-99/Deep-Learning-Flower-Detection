"""Cấu hình + các hàm dùng chung cho train.py và app.py."""
from pathlib import Path

import torch
from torch import nn
from torchvision import models
from torchvision import transforms as T

# ---------------------------------------------------------------- Đường dẫn
ROOT = Path(__file__).resolve().parent
IMAGES_DIR = ROOT / "data" / "flowers"            # data/flowers/<tên_loài>/<ảnh>
CSV_FILES = {
    "train": ROOT / "data" / "train.csv",
    "val": ROOT / "data" / "validation.csv",
    "test": ROOT / "data" / "test.csv",
}
MODEL_PATH = ROOT / "models" / "flower_model.pt"
REPORTS_DIR = ROOT / "reports"                    # mỗi lần train tạo 1 thư mục con

# ----------------------------------------------------------- Siêu tham số
IMG_SIZE = 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
BATCH_SIZE = 32          # hết VRAM thì giảm xuống 16
NUM_WORKERS = 2          # lỗi đa tiến trình trên Windows thì đặt = 0
EPOCHS_HEAD = 5          # giai đoạn 1: chỉ train lớp phân loại
EPOCHS_FINE = 10         # giai đoạn 2: fine-tune các block cuối (0 = bỏ qua)
LR_HEAD = 1e-3
LR_FINE = 1e-4
FINE_FROM_BLOCK = 14     # MobileNetV2 có 19 block (0-18)
SEED = 42

VI_NAMES = {"daisy": "Hoa cúc", "dandelion": "Bồ công anh", "rose": "Hoa hồng",
            "sunflower": "Hướng dương", "tulip": "Hoa tulip"}


def display_name(name: str) -> str:
    return f"{VI_NAMES[name]} ({name})" if name in VI_NAMES else name


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_label(device: torch.device) -> str:
    return torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"


def eval_transform() -> T.Compose:
    """Tiền xử lý khi đánh giá/dự đoán (train.py và app.py dùng chung)."""
    return T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.classifier[1].in_features, num_classes))
    return model


def load_model(device):
    """Trả về (model, class_names). Tên lớp lưu cùng model nên không thể bị lệch nhãn."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Chưa có mô hình tại {MODEL_PATH}. Hãy chạy 'python train.py' trước.")
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model = build_model(len(ckpt["class_names"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt["class_names"]
