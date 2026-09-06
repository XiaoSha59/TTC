#!/usr/bin/env python3
"""
Package and upload the 3 natural benchmarks (Insects, Plants, Animals)
to Kaggle Datasets as salala1706/inat21-natural.
Uses hardlinks to consume 0 extra disk space.
"""

import os
import sys
import glob
import json
import subprocess
from pathlib import Path

TARGET_CLASSES = [
    # 1. Insects (Bees vs Wasps)
    "Animalia_Arthropoda_Insecta_Hymenoptera_Apidae",
    "Animalia_Arthropoda_Insecta_Hymenoptera_Vespidae",
    # 2. Plants (Oaks vs Saxifrage)
    "Plantae_Tracheophyta_Magnoliopsida_Fagales_Fagaceae_Quercus",
    "Plantae_Tracheophyta_Magnoliopsida_Saxifragales",
    # 3. Animals (Ungulates vs Carnivores)
    "Animalia_Chordata_Mammalia_Artiodactyla",
    "Animalia_Chordata_Mammalia_Carnivora",
]

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_root = os.path.join(base_dir, "data", "inat21")
    dst_root = os.path.join(base_dir, "data", "inat21_natural")
    
    if not os.path.exists(src_root):
        print(f"Source directory not found: {src_root}")
        sys.exit(1)
        
    print(f">>> Preparing package at {dst_root} from {src_root}...")
    os.makedirs(dst_root, exist_ok=True)
    
    total_linked = 0
    total_dirs = 0

    for split in ["train_mini", "val"]:
        src_split = os.path.join(src_root, split)
        dst_split = os.path.join(dst_root, split)
        os.makedirs(dst_split, exist_ok=True)
        
        if not os.path.exists(src_split):
            print(f"Warning: split directory {src_split} not found!")
            continue
            
        all_subdirs = [d for d in os.listdir(src_split) if os.path.isdir(os.path.join(src_split, d))]
        matching_subdirs = [d for d in all_subdirs if any(c in d for c in TARGET_CLASSES)]
        
        print(f"[{split}] Found {len(matching_subdirs)} matching species categories out of {len(all_subdirs)}.")
        
        for d in matching_subdirs:
            src_cat = os.path.join(src_split, d)
            dst_cat = os.path.join(dst_split, d)
            os.makedirs(dst_cat, exist_ok=True)
            total_dirs += 1
            
            for fname in os.listdir(src_cat):
                src_file = os.path.join(src_cat, fname)
                dst_file = os.path.join(dst_cat, fname)
                if os.path.isfile(src_file) and not os.path.exists(dst_file):
                    try:
                        os.link(src_file, dst_file)
                        total_linked += 1
                    except OSError:
                        pass

    print(f">>> Successfully hardlinked {total_linked} images across {total_dirs} categories (0 bytes extra disk used).")

    # Create dataset-metadata.json for Kaggle
    metadata = {
        "title": "iNat21 Natural Benchmarks",
        "id": "salala1706/inat21-natural",
        "licenses": [{"name": "CC0-1.0"}]
    }
    
    metadata_path = os.path.join(dst_root, "dataset-metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f">>> Created Kaggle dataset metadata at {metadata_path}")

    # Set credentials
    os.environ['KAGGLE_USERNAME'] = 'salala1706'
    os.environ['KAGGLE_KEY'] = 'KGAT_c50dd809cbcb96fb725040dee59239f5'

    print(">>> Uploading dataset to Kaggle as salala1706/inat21-natural...")
    cmd = [sys.executable, "-m", "kaggle", "datasets", "create", "-p", dst_root, "--dir-mode", "zip"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("Stderr:", res.stderr)
        
    print(">>> Finished Kaggle dataset push!")

if __name__ == "__main__":
    main()
