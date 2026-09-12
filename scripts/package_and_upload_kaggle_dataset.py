#!/usr/bin/env python3
"""
Package and upload the Full iNaturalist 2021 Insects Benchmark (Apidae & Vespidae)
to Kaggle Datasets as salala1706/inat21-insects-full (Public).
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path

TARGET_CLASSES = [
    "Animalia_Arthropoda_Insecta_Hymenoptera_Apidae",
    "Animalia_Arthropoda_Insecta_Hymenoptera_Vespidae"
]

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Candidate source directories
    src_candidates = [
        os.path.join(base_dir, "data", "inat21_full"),
        "/home/tnpdung_79/TTC/data/inat21_full",
        "/home/XiaoSha/TTC/data/inat21_full",
        os.path.join(base_dir, "data", "inat21")
    ]
    
    src_root = None
    for cand in src_candidates:
        if os.path.exists(cand) and os.path.isdir(cand):
            src_root = cand
            break
            
    if not src_root:
        print(f"❌ Error: Could not find inat21_full directory in candidates: {src_candidates}")
        sys.exit(1)
        
    print(f">>> Found source dataset at: {src_root}")
    
    pkg_dir = os.path.join(base_dir, "data", "kaggle_insects_full_pkg")
    os.makedirs(pkg_dir, exist_ok=True)
    
    total_images = 0
    
    for split in ["train", "val"]:
        src_split = os.path.join(src_root, split)
        dst_split = os.path.join(pkg_dir, split)
        os.makedirs(dst_split, exist_ok=True)
        
        if not os.path.exists(src_split):
            print(f"⚠️ Warning: Split {src_split} not found!")
            continue
            
        all_subdirs = [d for d in os.listdir(src_split) if os.path.isdir(os.path.join(src_split, d))]
        matching_subdirs = [d for d in all_subdirs if any(c in d for c in TARGET_CLASSES)]
        
        print(f"[{split}] Found {len(matching_subdirs)} matching taxonomic species folders out of {len(all_subdirs)}.")
        
        for d in matching_subdirs:
            src_cat = os.path.join(src_split, d)
            dst_cat = os.path.join(dst_split, d)
            os.makedirs(dst_cat, exist_ok=True)
            
            for fname in os.listdir(src_cat):
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                src_file = os.path.join(src_cat, fname)
                dst_file = os.path.join(dst_cat, fname)
                if not os.path.exists(dst_file):
                    try:
                        os.link(src_file, dst_file)
                    except OSError:
                        try:
                            os.symlink(src_file, dst_file)
                        except OSError:
                            shutil.copy2(src_file, dst_file)
                total_images += 1

    print(f">>> Successfully prepared {total_images} images in {pkg_dir}")
    
    metadata = {
        "title": "iNat21 Insects Full Benchmark",
        "id": "salala1706/inat21-insects-full",
        "licenses": [{"name": "CC0-1.0"}],
        "is_private": False
    }
    
    meta_path = os.path.join(pkg_dir, "dataset-metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f">>> Written dataset metadata to {meta_path}")
    
    # Configure Kaggle credentials for salala1706
    os.environ['KAGGLE_USERNAME'] = 'salala1706'
    os.environ['KAGGLE_KEY'] = 'KGAT_c50dd809cbcb96fb725040dee59239f5'
    
    print(">>> Uploading dataset to Kaggle as salala1706/inat21-insects-full (Public)...")
    cmd = ["kaggle", "datasets", "create", "-p", pkg_dir, "--dir-mode", "zip", "-u"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("Stderr:", res.stderr)
        
    if "Your Dataset" in res.stdout or "created" in res.stdout.lower() or "creating" in res.stdout.lower():
        print("🎉 Dataset successfully uploaded to Kaggle: https://www.kaggle.com/datasets/salala1706/inat21-insects-full")
    else:
        # Check if already exists, then create new version
        print(">>> Attempting version update in case dataset exists...")
        cmd_ver = ["kaggle", "datasets", "version", "-p", pkg_dir, "-m", "Full Insects Dataset 9438 train 980 val", "--dir-mode", "zip"]
        res_ver = subprocess.run(cmd_ver, capture_output=True, text=True)
        print(res_ver.stdout)
        if res_ver.stderr:
            print("Stderr:", res_ver.stderr)

if __name__ == "__main__":
    main()
