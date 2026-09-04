#!/usr/bin/env python3
"""
Scan and automatically remove any corrupted/unidentifiable image files in data/inat21.
"""
import os
import sys
from PIL import Image

def clean_corrupted_images(data_dir="data/inat21"):
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist.")
        return

    print(f"Scanning for corrupted images in {data_dir} ...")
    corrupt_count = 0
    total_scanned = 0

    for root, _, files in os.walk(data_dir):
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                total_scanned += 1
                fpath = os.path.join(root, fname)
                try:
                    # Check 0-byte
                    if os.path.getsize(fpath) == 0:
                        os.remove(fpath)
                        corrupt_count += 1
                        print(f"🗑️ Removed 0-byte file: {fpath}")
                        continue
                    
                    # Verify PIL readability
                    with Image.open(fpath) as img:
                        img.verify()
                except Exception as e:
                    try:
                        os.remove(fpath)
                        corrupt_count += 1
                        print(f"🗑️ Removed corrupted image: {fpath} ({e})")
                    except OSError:
                        pass

    print(f"✅ Scan completed: {total_scanned} files checked, {corrupt_count} corrupted files removed.")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "data/inat21"
    clean_corrupted_images(target)
