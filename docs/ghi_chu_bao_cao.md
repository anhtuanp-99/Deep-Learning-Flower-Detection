# Ghi chú làm báo cáo

Sau `python train.py`, mọi thứ nằm trong `reports/run_<thời gian>/` (ảnh 300 dpi, dán thẳng vào Word).

| File | Dùng ở mục | Nên nói gì |
|---|---|---|
| `training_history.png` | Huấn luyện | Accuracy/loss theo epoch. Đường đứt là lúc bắt đầu fine-tune. Khoảng cách train–val lớn nghĩa là overfit |
| `confusion_matrix.png` | Đánh giá | Hàng là thực tế, cột là dự đoán; ô ngoài đường chéo là các lần nhầm giữa 2 loài |
| `classification_report.txt` | Đánh giá | Precision/Recall/F1 từng lớp; lớp nào yếu nhất |
| `sample_predictions.png` | Kết quả | Ví dụ minh họa; ảnh viền đỏ là ảnh bị đoán sai, dùng để phân tích lỗi |
| `metrics.json` | Phụ lục | Accuracy test, thiết bị, thời gian train |

## Phương pháp (để viết vào báo cáo)

- Mô hình: MobileNetV2 pretrained ImageNet, thay lớp cuối bằng `Dropout(0.2) + Linear(1280, 5)`, ảnh vào 224×224.
- Huấn luyện 2 giai đoạn: (1) đóng băng backbone, train lớp cuối 5 epoch, lr 1e-3; (2) fine-tune các block 14–18, 10 epoch, lr 1e-4.
- Augmentation (chỉ tập train): random crop, lật ngang, xoay ±20°, đổi sáng/tương phản/bão hòa.
- Optimizer AdamW, mixed precision FP16 trên GPU; chọn trọng số của epoch có val loss thấp nhất.

## Giải thích chỉ số

- **Accuracy**: tỉ lệ ảnh đoán đúng trên toàn tập test.
- **Precision (lớp X)**: trong số ảnh model đoán là X, bao nhiêu phần trăm đúng.
- **Recall (lớp X)**: trong số ảnh thật sự là X, model tìm ra bao nhiêu phần trăm.
- **F1**: trung bình điều hòa của precision và recall.

## Lưu ý khi báo cáo

- Chỉ báo cáo accuracy trên **tập test**; tập validation dùng để chọn epoch tốt nhất.
- Model chỉ biết 5 loài. Vật khác đưa vào camera vẫn bị gán vào một trong 5 loài, ngưỡng tin cậy chỉ giảm chứ không loại hết được.
- Nhận diện qua camera có thể kém hơn tập test vì khác ánh sáng và nền.
