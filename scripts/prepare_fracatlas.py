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
    images_dir = os.path.join(data_root, "images")
    if not os.path.exists(images_dir):
        # Check alternative capitalization
        images_dir = os.path.join("data/fracatlas", "images")
        if not os.path.exists(images_dir):
            images_dir = data_root

    classes = ["Fractured", "Non_fractured"]
    for split in ["train", "val", "test"]:
        for cls in classes:
            os.makedirs(os.path.join(output_root, split, cls), exist_ok=True)

    for cls in classes:
        cls_dir = os.path.join(images_dir, cls)
        if not os.path.exists(cls_dir):
            print(f"Directory {cls_dir} not found. Skipping.")
            continue
        
        files = glob.glob(os.path.join(cls_dir, "*.*"))
        random.shuffle(files)
        
        n = len(files)
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)
        
        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]
        
        for f in train_files:
            dst = os.path.join(output_root, "train", cls, os.path.basename(f))
            if not os.path.exists(dst):
                try:
                    os.link(f, dst)
                except:
                    shutil.copy2(f, dst)
                    
        for f in val_files:
            dst = os.path.join(output_root, "val", cls, os.path.basename(f))
            if not os.path.exists(dst):
                try:
                    os.link(f, dst)
                except:
                    shutil.copy2(f, dst)
                    
        for f in test_files:
            dst = os.path.join(output_root, "test", cls, os.path.basename(f))
            if not os.path.exists(dst):
                try:
                    os.link(f, dst)
                except:
                    shutil.copy2(f, dst)
                    
        print(f"Class {cls}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test.")

    print(f"FracAtlas splits successfully created at {output_root}!")

if __name__ == "__main__":
    prepare_splits()
