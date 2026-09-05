#!/usr/bin/env python3
"""
Test Linear Probing evaluation on pre-trained checkpoints.
Evaluates representation quality on 1% balanced subset + balanced test set.
"""

import os
import sys
import argparse
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score, classification_report
from tqdm import tqdm

# Import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.sup_cont import ContrastiveResNet50
from data.data_module import NClassesDataModule
from data.augmentation import SimCLRValTransform, SimCLRTrainTransform


def extract_features(model, dataloader, device):
    """Extract features from frozen backbone."""
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features", leave=False):
            if isinstance(batch, (tuple, list)):
                x, y = batch[0], batch[1]
            else:
                x, y = batch['image'], batch['label']
            
            x = x.to(device)
            if hasattr(model, 'base_encoder'):
                feat = model.base_encoder(x)
            elif hasattr(model, 'encoder'):
                feat = model.encoder(x)
            else:
                feat = model(x)
                
            if isinstance(feat, dict):
                feat = feat.get('feats', feat.get('features', feat))
            elif isinstance(feat, (tuple, list)):
                feat = feat[0]
                
            feat = feat.view(feat.size(0), -1)
            features.append(feat.cpu().numpy())
            labels.append(y.cpu().numpy())
            
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def evaluate_checkpoint(ckpt_path, dataset_type="insects", data_dir="data/inat21"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"🚀 EVALUATING LINEAR PROBE FOR: {os.path.basename(ckpt_path)}")
    print(f"Device: {device}")
    print("=" * 70)

    # Class definitions according to specs
    if dataset_type == "insects":
        classes = [
            "Animalia_Arthropoda_Insecta_Hymenoptera_Apidae",
            "Animalia_Arthropoda_Insecta_Hymenoptera_Vespidae"
        ]
    elif dataset_type == "plants":
        classes = [
            "Plantae_Tracheophyta_Magnoliopsida_Fagales_Fagaceae_Quercus",
            "Plantae_Tracheophyta_Magnoliopsida_Saxifragales"
        ]
    elif dataset_type == "animals":
        classes = [
            "Animalia_Chordata_Mammalia_Artiodactyla",
            "Animalia_Chordata_Mammalia_Carnivora"
        ]
    else:
        raise ValueError(f"Unknown dataset: {dataset_type}")

    # 1. Load model checkpoint
    print(">>> Loading pre-trained checkpoint...")
    try:
        model = ContrastiveResNet50.load_from_checkpoint(ckpt_path, map_location=device)
    except Exception as e:
        print(f"Direct load failed ({e}), loading via torch.load state_dict...")
        model = ContrastiveResNet50()
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        
    model = model.to(device)
    model.eval()

    # 2. Setup DataModule with balanced test and 1% balanced train
    val_transform = SimCLRValTransform(img_height=224, normalize=None)
    dm = NClassesDataModule(
        data_dir=data_dir,
        classes=classes,
        train_transform=val_transform,
        val_transform=val_transform,
        batch_size=128,
        num_workers=4,
        subsample_balanced=True
    )
    dm.setup()

    # 3. Extract features
    print(">>> Extracting features on test set...")
    test_loader = dm.test_dataloader()
    X_test, y_test = extract_features(model, test_loader, device)
    
    print(f"Test samples: {len(y_test)} (Class 0: {(y_test==0).sum()}, Class 1: {(y_test==1).sum()})")

    # Extract features on train set
    print(">>> Extracting features on 1% balanced training subset...")
    train_loader = dm.train_dataloader()
    X_train_full, y_train_full = extract_features(model, train_loader, device)

    # Subsample 1% balanced as in paper Supplementary S2.3
    np.random.seed(42)
    idx_0 = np.where(y_train_full == 0)[0]
    idx_1 = np.where(y_train_full == 1)[0]
    
    # Select 1% or minimum 10 samples per class
    n_samples_per_class = max(10, int(len(idx_0) * 0.01))
    sel_0 = np.random.choice(idx_0, min(len(idx_0), n_samples_per_class), replace=False)
    sel_1 = np.random.choice(idx_1, min(len(idx_1), n_samples_per_class), replace=False)
    probe_idx = np.concatenate([sel_0, sel_1])
    
    X_probe = X_train_full[probe_idx]
    y_probe = y_train_full[probe_idx]
    print(f"Probe Training samples: {len(y_probe)} ({n_samples_per_class} per class)")

    # 4. Train Balanced Logistic Regression Probe
    print(">>> Fitting Balanced Linear Classifier (L-BFGS / class_weight='balanced')...")
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0, solver='lbfgs')
    clf.fit(X_probe, y_probe)

    # 5. Predict & Evaluate
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    bal_acc = balanced_accuracy_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except:
        auc = 0.5

    print("\n" + "=" * 70)
    print(f"🎯 RESULTS FOR: {os.path.basename(ckpt_path)}")
    print("=" * 70)
    print(f"  ⭐ Balanced Accuracy (Paper Metric): {bal_acc * 100:.2f}%")
    print(f"  ⭐ Raw Test Accuracy:               {acc * 100:.2f}%")
    print(f"  ⭐ Area Under Curve (AUROC):        {auc * 100:.2f}%")
    print("=" * 70)
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Class 0", "Class 1"]))
    
    return bal_acc, auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument("--dataset", type=str, default="insects", choices=["plants", "insects", "animals"])
    parser.add_argument("--data_dir", type=str, default="data/inat21")
    args = parser.parse_args()

    ckpt_path = args.ckpt
    if ckpt_path is None:
        # Search for latest checkpoint matching dataset
        pattern = f"logs/**/*{args.dataset}*/**/*.ckpt"
        ckpts = glob.glob(pattern, recursive=True)
        if not ckpts:
            pattern2 = f"logs/**/{args.dataset}*.ckpt"
            ckpts = glob.glob(pattern2, recursive=True)
        if ckpts:
            # Pick a 95_5 or 99_1 checkpoint to test
            for c in ckpts:
                if "95_5" in c and "supproto" in c:
                    ckpt_path = c
                    break
            if ckpt_path is None:
                ckpt_path = ckpts[-1]
        else:
            print(f"No checkpoints found matching {args.dataset}")
            sys.exit(1)

    evaluate_checkpoint(ckpt_path, args.dataset, args.data_dir)
