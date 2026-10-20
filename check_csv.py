import pandas as pd
import os

# Đường dẫn tới các file CSV
train_csv = "train.csv"
val_csv = "validation.csv"
test_csv = "test.csv"

def analyze_csv(file_path, name):
    print(f"\n{'='*50}")
    print(f"📂 ĐANG PHÂN TÍCH: {name} ({file_path})")
    print(f"{'='*50}")
    
    if not os.path.exists(file_path):
        print(f"❌ Không tìm thấy file {file_path}")
        return None
    
    df = pd.read_csv(file_path)
    
    # 1. Tổng số ảnh
    print(f"  📊 Tổng số ảnh: {len(df)}")
    
    # 2. Số lượng ảnh theo từng loài hoa
    print(f"\n  🌸 Phân bổ theo loài hoa:")
    class_counts = df['ClassName'].value_counts()
    for class_name, count in class_counts.items():
        print(f"     - {class_name:<12}: {count} ảnh")
    
    # 3. Kiểm tra xem file ảnh có thực sự tồn tại trên ổ cứng không
    print(f"\n  🔍 Đang kiểm tra sự tồn tại của ảnh trên ổ cứng...")
    missing_files = []
    # Chỉ kiểm tra 20 ảnh đầu để tiết kiệm thời gian (vì dataset có mấy nghìn ảnh)
    for idx, row in df.head(20).iterrows():
        # Đường dẫn trong CSV là "daisy/daisy_000019.png", nhưng thực tế nằm trong "flowers/daisy/..."
        full_path = os.path.join("flowers", row['Filename'])
        if not os.path.exists(full_path):
            missing_files.append(full_path)
    
    if len(missing_files) == 0:
        print(f"     ✅ 20 ảnh kiểm tra đầu tiên đều tồn tại trên ổ cứng.")
    else:
        print(f"     ⚠️  Có {len(missing_files)} ảnh không tìm thấy. Ví dụ: {missing_files[0]}")
    
    return set(df['Filename'].values) # Trả về set các tên file để so sánh trùng lặp

# === CHẠY PHÂN TÍCH ===
train_files = analyze_csv(train_csv, "TRAIN SET")
val_files = analyze_csv(val_csv, "VALIDATION SET")
test_files = analyze_csv(test_csv, "TEST SET")

# === KIỂM TRA TRÙNG LẶP (OVERLAP) ===
print(f"\n{'='*50}")
print(f"🔎 KIỂM TRA TRÙNG LẶP GIỮA CÁC TẬP")
print(f"{'='*50}")

if train_files and val_files and test_files:
    overlap_train_val = train_files.intersection(val_files)
    overlap_train_test = train_files.intersection(test_files)
    overlap_val_test = val_files.intersection(test_files)
    
    print(f"  Train ∩ Validation : {len(overlap_train_val)} ảnh trùng")
    print(f"  Train ∩ Test       : {len(overlap_train_test)} ảnh trùng")
    print(f"  Validation ∩ Test  : {len(overlap_val_test)} ảnh trùng")
    
    if len(overlap_train_val) == 0 and len(overlap_train_test) == 0 and len(overlap_val_test) == 0:
        print("\n  ✅ TUYỆT VỜI! 3 tập dữ liệu hoàn toàn tách biệt, không có ảnh nào bị trùng lặp.")
    else:
        print("\n  ⚠️  CẢNH BÁO: Có ảnh bị xuất hiện ở nhiều tập khác nhau (Data Leakage)!")