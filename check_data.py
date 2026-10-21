"""Kiểm tra dữ liệu trước khi train: ảnh hỏng/sai định dạng, số ảnh mỗi lớp, file CSV, trùng lặp giữa các tập.

    python check_data.py
"""
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent
IMAGES_DIR = ROOT / "data" / "flowers"
CSVS = {"train": "train.csv", "validation": "validation.csv", "test": "test.csv"}
VALID_EXT = {".jpg", ".jpeg", ".png"}


def check_images():
    print("=" * 50, "\nKIỂM TRA THƯ MỤC ẢNH\n" + "=" * 50)
    if not IMAGES_DIR.exists():
        print(f"Không thấy {IMAGES_DIR}")
        return
    bad, total = [], 0
    for class_dir in sorted(p for p in IMAGES_DIR.iterdir() if p.is_dir()):
        count = 0
        for f in class_dir.iterdir():
            if f.name.startswith(".") or f.name.lower() in ("thumbs.db", "desktop.ini"):
                bad.append((f, "file rác"))
            elif f.suffix.lower() not in VALID_EXT:
                bad.append((f, "sai định dạng"))
            else:
                try:
                    with Image.open(f) as img:
                        img.verify()
                    count += 1
                except Exception:
                    bad.append((f, "ảnh hỏng"))
        print(f"  {class_dir.name:<12}: {count} ảnh")
        total += count
    print(f"  {'TỔNG':<12}: {total} ảnh hợp lệ")
    for f, why in bad:
        print(f"  [!] {why}: {f}")
    if not bad:
        print("  Không có file lỗi.")


def check_csv():
    print("\n" + "=" * 50, "\nKIỂM TRA FILE CSV\n" + "=" * 50)
    sets, mappings = {}, []
    for name, fname in CSVS.items():
        path = ROOT / "data" / fname
        if not path.exists():
            print(f"  [!] Không thấy {path}")
            return
        df = pd.read_csv(path)
        missing = [f for f in df["Filename"] if not (IMAGES_DIR / f).exists()]
        print(f"  {name:<11}: {len(df)} ảnh | " + ", ".join(f"{k}={v}" for k, v in df["ClassName"].value_counts().items()))
        if missing:
            print(f"  [!] thiếu {len(missing)} ảnh, ví dụ: {IMAGES_DIR / missing[0]}")
        sets[name] = set(df["Filename"])
        mappings.append(dict(zip(df["Label"], df["ClassName"])))

    if any(m != mappings[0] for m in mappings):
        print("  [!] Cặp Label/ClassName không thống nhất giữa 3 file CSV")
    leak = False
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        n = len(sets[a] & sets[b])
        leak |= n > 0
        print(f"  {a} ∩ {b}: {n} ảnh trùng")
    print("  Các tập tách biệt hoàn toàn." if not leak else "  [!] CẢNH BÁO: rò rỉ dữ liệu giữa các tập!")


if __name__ == "__main__":
    check_images()
    check_csv()
