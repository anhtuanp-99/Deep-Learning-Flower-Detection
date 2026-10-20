import os
import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt

# ============================================================
# 👇 SỬA DUY NHẤT DÒNG NÀY ĐỂ ĐỔI ẢNH CẦN DỰ ĐOÁN 👇
IMG_PATH = "flowers/tulip/tulip_000781.png"
# ============================================================

# Cấu hình (không cần sửa)
MODEL_PATH = "flower_model_csv.h5"
CLASS_NAMES = ['daisy', 'dandelion', 'sunflower', 'tulip', 'rose']
IMG_SIZE = (224, 224)


def main():
    print("\n" + "=" * 55)
    print("🌸  PHÂN LOẠI HOA - MOBILENETV2 (Transfer Learning)")
    print("=" * 55)

    # 1. Kiểm tra file mô hình
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Không tìm thấy file mô hình '{MODEL_PATH}'")
        print(f"   → Hãy chạy 'python train_csv.py' trước để tạo mô hình.")
        sys.exit(1)

    # 2. Kiểm tra file ảnh
    if not os.path.exists(IMG_PATH):
        print(f"❌ Không tìm thấy ảnh tại: '{IMG_PATH}'")
        print(f"   → Kiểm tra lại đường dẫn trong biến IMG_PATH ở đầu file.")
        print(f"   → Đường dẫn đúng có dạng: flowers/<tên_loài>/<tên_file>")
        print(f"   → Ví dụ: flowers/daisy/daisy_000019.png")
        sys.exit(1)

    # 3. Load mô hình
    print(f"⏳ Đang tải mô hình...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Đã tải mô hình xong.\n")

    # 4. Load và xử lý ảnh
    print(f"🖼️  Ảnh cần dự đoán: {IMG_PATH}")
    img = load_img(IMG_PATH, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Thêm batch dimension

    # 5. Dự đoán
    print(f"⏳ Đang phân tích...\n")
    predictions = model.predict(img_array, verbose=0)
    scores = predictions[0]

    # 6. Hiển thị kết quả
    sorted_indices = np.argsort(scores)[::-1]
    top1_idx = sorted_indices[0]
    top1_class = CLASS_NAMES[top1_idx]
    top1_conf = scores[top1_idx] * 100

    print("=" * 55)
    print(f"🏆 DỰ ĐOÁN: {top1_class.upper()} ({top1_conf:.2f}%)")
    print("=" * 55)
    print("📊 Top 3 dự đoán:")
    for i in range(min(3, len(CLASS_NAMES))):
        idx = sorted_indices[i]
        conf = scores[idx] * 100
        bar = "█" * int(conf / 5)
        print(f"   {i+1}. {CLASS_NAMES[idx]:<12} {conf:>6.2f}%  {bar}")
    print("=" * 55)

    # 7. Hiển thị ảnh
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"Dự đoán: {top1_class} ({top1_conf:.2f}%)", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()