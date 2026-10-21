# Phân loại hoa bằng MobileNetV2

Nhận diện 5 loài hoa (daisy, dandelion, rose, sunflower, tulip) bằng MobileNetV2 + Transfer Learning (PyTorch, chạy GPU).
Có giao diện nhận diện từ file ảnh, camera laptop hoặc camera điện thoại. Sau khi train, biểu đồ và báo cáo được lưu tự động.

## Cấu trúc

```
Flower_Project/
├── common.py          # cấu hình + hàm dùng chung (đường dẫn, siêu tham số, model)
├── train.py           # huấn luyện + đánh giá + xuất biểu đồ
├── app.py             # giao diện: chọn ảnh / camera laptop / camera điện thoại
├── check_data.py      # kiểm tra dữ liệu trước khi train
├── find_errors.py     # lọc các ảnh model đoán sai để in ra demo
├── requirements.txt
├── data/
│   ├── flowers/       # daisy/ dandelion/ rose/ sunflower/ tulip/
│   └── train.csv, validation.csv, test.csv
├── models/            # flower_model.pt (tự tạo sau khi train)
├── reports/           # run_<thời gian>/ biểu đồ + báo cáo (tự tạo)
├── captures/          # ảnh chụp từ camera (tự tạo)
├── demo_errors/       # ảnh đoán sai do find_errors.py chép ra (tự tạo)
└── docs/              # cai_dat_gpu.md, ghi_chu_bao_cao.md
```

## Cài đặt

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"     # phải ra True
```

Chi tiết và cách xử lý khi ra `False`: [docs/cai_dat_gpu.md](docs/cai_dat_gpu.md).

## Chạy

| Việc cần làm | Lệnh |
|---|---|
| Kiểm tra dữ liệu | `python check_data.py` |
| Huấn luyện + xuất báo cáo | `python train.py` |
| Mở giao diện nhận diện | `python app.py` |
| Lọc ảnh đoán sai để in demo | `python find_errors.py` |

Chỉnh số epoch, batch size, learning rate ở đầu file `common.py`.

## Camera điện thoại (IP Webcam)

1. Cài app **IP Webcam** (Android) trên điện thoại, bấm **Start server**. Màn hình hiện địa chỉ dạng `http://192.168.1.5:8080`.
2. Điện thoại và laptop phải **cùng mạng Wi-Fi**. Có thể kiểm tra bằng cách mở địa chỉ đó trên trình duyệt laptop.
3. Mở `app.py`, điền địa chỉ vào biến ở đầu file (gõ thiếu `http://` và `/video` cũng được):
   ```python
   PHONE_CAMERA_URL = "192.168.1.5:8080"
   ```
4. Chạy `python app.py` và bấm **Mở camera điện thoại**.

Luồng hình bị trễ thì giảm độ phân giải trong cài đặt của app IP Webcam (Video preferences). Với iPhone, dùng app phát luồng MJPEG và điền đầy đủ URL luồng vào `PHONE_CAMERA_URL`.

## Lọc ảnh đoán sai để demo

```powershell
python find_errors.py                   # quét tập test
python find_errors.py --min-conf 0.6    # chỉ lấy ca sai mà model "tự tin" (app sẽ hiện tên sai)
python find_errors.py --split all       # quét cả train + validation + test
```

Danh sách (kèm đường dẫn ảnh cụ thể) được in ra màn hình, ảnh sai được chép vào `demo_errors/` cùng file `danh_sach.csv`.
Ảnh in ra rồi chụp lại bằng camera có thể cho kết quả khác ảnh gốc (ánh sáng, góc nghiêng), nên hãy thử trước với app.

## Kết quả sau khi train

`reports/run_<thời gian>/` gồm: `training_history.png`, `confusion_matrix.png`, `sample_predictions.png`,
`classification_report.txt`, `metrics.json`. Cách dùng trong báo cáo: [docs/ghi_chu_bao_cao.md](docs/ghi_chu_bao_cao.md).
