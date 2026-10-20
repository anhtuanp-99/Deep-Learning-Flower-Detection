# %% [Cell 1] Import thư viện
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# %% [Cell 2] Cấu hình
IMG_SIZE = (224, 224)
BATCH_SIZE = 16  # Giảm xuống 16 để tránh tràn RAM
EPOCHS = 10
DATA_DIR = "flowers"

# %% [Cell 3] Đọc dữ liệu từ 3 file CSV
print("=" * 60)
print("📂 ĐỌC DỮ LIỆU TỪ FILE CSV (Theo đúng quy trình tác giả)")
print("=" * 60)

train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("validation.csv")
test_df = pd.read_csv("test.csv")

print(f"  Train      : {len(train_df)} ảnh")
print(f"  Validation : {len(val_df)} ảnh")
print(f"  Test       : {len(test_df)} ảnh")

# Tạo mapping Label -> ClassName (đảm bảo thứ tự đúng)
label_to_class = train_df.groupby('Label')['ClassName'].first().to_dict()
class_names = [label_to_class[i] for i in range(len(label_to_class))]
print(f"\n  Các lớp: {class_names}")

# Tạo đường dẫn đầy đủ (thêm tiền tố "flowers/")
def get_full_paths(df):
    return [os.path.join(DATA_DIR, f) for f in df['Filename'].values]

train_paths = get_full_paths(train_df)
val_paths = get_full_paths(val_df)
test_paths = get_full_paths(test_df)

train_labels = train_df['Label'].values
val_labels = val_df['Label'].values
test_labels = test_df['Label'].values

# %% [Cell 4] Tạo pipeline dữ liệu
print("\n" + "=" * 60)
print("🔧 TẠO PIPELINE DỮ LIỆU")
print("=" * 60)

def load_and_preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.cast(img, tf.float32)  # Giữ trong khoảng [0, 255]
    img = tf.image.resize(img, IMG_SIZE)
    return img, label

def create_dataset(paths, labels, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(paths), 1000), seed=123)
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = create_dataset(train_paths, train_labels, shuffle=True)
val_ds = create_dataset(val_paths, val_labels)
test_ds = create_dataset(test_paths, test_labels)

print("  ✅ Đã tạo xong 3 dataset")

# %% [Cell 5] Xây dựng mô hình MobileNetV2
print("\n" + "=" * 60)
print("🏗️  XÂY DỰNG MÔ HÌNH MOBILENETV2")
print("=" * 60)

base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = models.Sequential([
    layers.Rescaling(1./127.5, offset=-1),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# %% [Cell 6] Huấn luyện
print("\n" + "=" * 60)
print("🚀 BẮT ĐẦU HUẤN LUYỆN")
print("=" * 60)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

model.save("flower_model_csv.h5")
print("\n✅ Đã lưu mô hình vào flower_model_csv.h5")

# %% [Cell 7] Vẽ biểu đồ
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Độ chính xác (Accuracy)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Sai số (Loss)')
plt.legend()

plt.tight_layout()
plt.savefig('training_history_csv.png')
print("✅ Đã lưu biểu đồ vào training_history_csv.png")

# %% [Cell 8] Đánh giá trên tập TEST (dữ liệu chưa từng thấy)
print("\n" + "=" * 60)
print("📊 ĐÁNH GIÁ TRÊN TẬP TEST")
print("=" * 60)

test_loss, test_acc = model.evaluate(test_ds, verbose=1)
print(f"\n  🎯 Test Accuracy: {test_acc * 100:.2f}%")
print(f"  📉 Test Loss    : {test_loss:.4f}")

# %% [Cell 9] Confusion Matrix & Báo cáo chi tiết
print("\n" + "=" * 60)
print("🔍 CONFUSION MATRIX & CLASSIFICATION REPORT")
print("=" * 60)

y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_labels

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - Tập Test')
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("✅ Đã lưu confusion matrix vào confusion_matrix.png")

print("\n📋 BÁO CÁO CHI TIẾT TỪNG LỚP:")
print(classification_report(y_true, y_pred, target_names=class_names))