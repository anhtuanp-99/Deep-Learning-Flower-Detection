# Cài đặt và bật GPU (RTX 3050, Windows)

## Cài đặt

Tạo venv **mới** (không dùng lại venv cũ của TensorFlow), Python 3.10 – 3.12:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))" # phải ra True 
```

Không cần cài CUDA Toolkit riêng, bản PyTorch đã kèm sẵn. Chỉ cần driver NVIDIA đủ mới.

## Lỗi thường gặp khi train

| Lỗi | Cách xử lý |
|---|---|
| `CUDA out of memory` | Giảm `BATCH_SIZE` xuống 16 trong `common.py` |
| Lỗi đa tiến trình (DataLoader) trên Windows | Đặt `NUM_WORKERS = 0` trong `common.py` |
| Dòng đầu `train.py` in `CPU` | GPU chưa được bật, xem bảng trên |

## Lưu ý

GPU giúp nhiều cho **huấn luyện**. Nhận diện bằng camera chạy nhanh trên cả CPU lẫn GPU vì mỗi lần chỉ xử lý 1 ảnh.
