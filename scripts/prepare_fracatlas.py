#!/usr/bin/env python3
"""
Prepare FracAtlas dataset splits (train/val/test) for SimpleImageDataModule.
"""

import os
import shutil
import glob
import random
from pathlib import Path

def prepare_splits(data_root="data/FracAtlas", output_root="data/fracatlas_splits", seed=42):
    random.seed(seed)
    
    # Auto-detect on Kaggle or local if default data_root does not exist
    fractured_dir = None
    non_fractured_dir = None

    search_roots = ["/kaggle/input", "data", "."]
    for s_root in search_roots:
        if not os.path.exists(s_root):
            continue
        for root, dirs, files in os.walk(s_root):
            for d in dirs:
                dl = d.lower().replace("-", "_").replace(" ", "_")
                if dl in ["fractured", "fracture"] and fractured_dir is None:
                    # Verify it has image files
                    test_files = glob.glob(os.path.join(root, d, "*.*"))
                    if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in test_files):
                        fractured_dir = os.path.join(root, d)
                elif dl in ["non_fractured", "not_fractured", "nonfractured"] and non_fractured_dir is None:
                    test_files = glob.glob(os.path.join(root, d, "*.*"))
                    if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in test_files):
                        non_fractured_dir = os.path.join(root, d)
            if fractured_dir and non_fractured_dir:
                break
        if fractured_dir and non_fractured_dir:
            break

    if fractured_dir and non_fractured_dir:
        print(f">>> Auto-detected FracAtlas classes:\n    Fractured: {fractured_dir}\n    Non-fractured: {non_fractured_dir}")
    else:
        print(f">>> Could not find both Fractured and Non_fractured directories in {search_roots}.")
        return

    classes = [("Fractured", fractured_dir), ("Non_fractured", non_fractured_dir)]
    for split in ["train", "val", "test"]:
        for cls_name, _ in classes:
            os.makedirs(os.path.join(output_root, split, cls_name), exist_ok=True)

    for cls_name, cls_dir in classes:
        files = [f for f in glob.glob(os.path.join(cls_dir, "*.*")) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(files)
        
        n = len(files)
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)
        
        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]
        
        for f in train_files:
            dst = os.path.join(output_root, "train", cls_name, os.path.basename(f))
            if not os.path.exists(dst):
                try:
                    os.link(f, dst)
                except:
                    shutil.copy2(f, dst)
                    
        for f in val_files:
            dst = os.path.join(output_root, "val", cls_name, os.path.basename(f))
            if not os.path.exists(dst):
                try:
                    os.link(f, dst)
                except:
                    shutil.copy2(f, dst)
                    
        for f in test_files:
            dst = os.path.join(output_root, "test", cls_name, os.path.basename(f))
            if not os.path.exists(dst):
                try:
                    os.link(f, dst)
                except:
                    shutil.copy2(f, dst)
                    
        print(f"Class {cls_name}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test.")

    print(f"FracAtlas splits successfully created at {output_root}!")

if __name__ == "__main__":
    prepare_splits()
