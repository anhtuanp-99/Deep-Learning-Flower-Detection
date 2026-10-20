import os
from PIL import Image
from collections import defaultdict

data_dir = "flowers"
bad_files = []
wrong_ext_files = []
wrong_prefix_files = []
hidden_files = []
class_counts = defaultdict(int)
total_valid_images = 0

VALID_EXTENSIONS = ['.jpg', '.jpeg', '.png']

print("="*50)
print("🔍 BẮT ĐẦU KIỂM TRA DỮ LIỆU (BẢN NÂNG CẤP)...")
print("="*50)

for root, dirs, files in os.walk(data_dir):
    if root == data_dir:
        continue
        
    class_name = os.path.basename(root).lower()
    
    if len(files) == 0:
        print(f"⚠️  CẢNH BÁO: Thư mục '{class_name}' bị trống!")
        continue

    for file in files:
        file_path = os.path.join(root, file)
        file_lower = file.lower()
        ext = os.path.splitext(file_lower)[1]

        # 1. Kiểm tra file rác hệ thống
        if file.startswith('.') or file_lower in ['thumbs.db', 'desktop.ini']:
            hidden_files.append(file_path)
            continue

        # 2. Kiểm tra định dạng file
        if ext not in VALID_EXTENSIONS:
            wrong_ext_files.append(file_path)
            continue

        # 3. Kiểm tra tiền tố tên file (phải bắt đầu bằng tên loài hoa)
        if not file_lower.startswith(class_name):
            wrong_prefix_files.append(file_path)

        # 4. Kiểm tra file ảnh có bị hỏng không
        try:
            img = Image.open(file_path)
            img.verify() 
            
            class_counts[class_name] += 1
            total_valid_images += 1
            
        except (IOError, SyntaxError):
            bad_files.append(file_path)

# ================= IN KẾT QUẢ =================
print("\n📊 THỐNG KÊ SỐ LƯỢNG ẢNH THEO TỪNG LỚP:")
print("-" * 40)
for class_name, count in class_counts.items():
    print(f"  🌸 {class_name:<12}: {count} ảnh")
print("-" * 40)
print(f"  📁 TỔNG CỘNG     : {total_valid_images} ảnh hợp lệ")

print("\n❌ KẾT QUẢ KIỂM TRA LỖI:")
print("-" * 40)

if len(bad_files) == 0 and len(wrong_ext_files) == 0 and len(wrong_prefix_files) == 0 and len(hidden_files) == 0:
    print("  ✅ TUYỆT VỜI! Dữ liệu đạt chuẩn 100%, không có file rác hay sai tên.")
else:
    if hidden_files:
        print(f"  🗑️  File rác hệ thống ({len(hidden_files)} file) - Nên xóa:")
        for f in hidden_files: print(f"     - {f}")
        
    if wrong_ext_files:
        print(f"  📁 File sai định dạng ({len(wrong_ext_files)} file) - Nên xóa:")
        for f in wrong_ext_files: print(f"     - {f}")
        
    if wrong_prefix_files:
        print(f"  🏷️  File sai quy tắc đặt tên ({len(wrong_prefix_files)} file) - Cảnh báo:")
        for f in wrong_prefix_files: print(f"     - {f}")
        
    if bad_files:
        print(f"  💔 File ảnh bị hỏng ({len(bad_files)} file) - BẮT BUỘC xóa:")
        for f in bad_files: print(f"     - {f}")

print("="*50)