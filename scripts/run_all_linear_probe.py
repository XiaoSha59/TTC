#!/usr/bin/env python3
"""
Comprehensive Linear Probe Evaluation across Table 1 (Plants, Insects, Animals).
Evaluates all pre-trained checkpoints using the paper's Balanced Linear Probing protocol (Supplementary S2.3).
"""

import os
import sys
import glob
import json
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score
from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8')

# Include project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.sup_cont import ContrastiveResNet50
from data.data_module import NClassesDataModule
from data.augmentation import SimCLRValTransform


DATASET_CLASSES = {
    "plants": [
        "Plantae_Tracheophyta_Magnoliopsida_Fagales_Fagaceae_Quercus",
        "Plantae_Tracheophyta_Magnoliopsida_Saxifragales"
    ],
    "insects": [
        "Animalia_Arthropoda_Insecta_Hymenoptera_Apidae",
        "Animalia_Arthropoda_Insecta_Hymenoptera_Vespidae"
    ],
    "animals": [
        "Animalia_Chordata_Mammalia_Artiodactyla",
        "Animalia_Chordata_Mammalia_Carnivora"
    ]
}

METHODS = ["weightedce", "supcon", "supmin", "supproto"]
RATIOS = ["50_50", "95_5", "99_1"]


def build_checkpoint_map(log_dir="logs"):
    """
    Scans logs/ and maps each run name (e.g. 'insects-95_5-supproto') 
    to its corresponding checkpoint file by reading .hydra/config.yaml.
    """
    ckpt_map = {}
    print(f"Scanning for checkpoints in '{log_dir}'...")

    # Method 1: Inspect .hydra/config.yaml in all run directories
    cfg_files = glob.glob(f"{log_dir}/**/.hydra/config.yaml", recursive=True)
    for cfg_file in cfg_files:
        run_dir = os.path.dirname(os.path.dirname(cfg_file))
        try:
            with open(cfg_file, 'r', encoding='utf-8', errors='ignore') as f:
                cfg = yaml.safe_load(f)
            run_name = cfg.get("name")
            if run_name:
                ckpt_dir = os.path.join(run_dir, "checkpoints")
                if os.path.exists(ckpt_dir):
                    ckpts = glob.glob(f"{ckpt_dir}/*.ckpt")
                    if ckpts:
                        best = None
                        for c in ckpts:
                            if "last.ckpt" in c:
                                best = c
                                break
                        if not best:
                            ckpts.sort(key=os.path.getmtime, reverse=True)
                            best = ckpts[0]
                        ckpt_map[run_name.lower().strip()] = best
        except Exception:
            pass

    # Method 2: Scan all .ckpt files directly
    all_ckpts = glob.glob(f"{log_dir}/**/*.ckpt", recursive=True)
    all_ckpts += glob.glob("checkpoints/**/*.ckpt", recursive=True)
    for c in all_ckpts:
        c_lower = c.lower()
        for d in ["plants", "insects", "animals"]:
            for r in ["50_50", "95_5", "99_1"]:
                for m in ["supproto", "supmin", "supcon", "weightedce"]:
                    tag = f"{d}-{r}-{m}"
                    if tag in c_lower and tag not in ckpt_map:
                        ckpt_map[tag] = c

    print(f"Mapped {len(ckpt_map)} runs to checkpoints.")
    return ckpt_map


def extract_features(model, dataloader, device):
    """Extract features from frozen ResNet-50 backbone."""
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


def evaluate_single_model(ckpt_path, dataset, data_dir="data/inat21", device="cuda"):
    """Evaluate linear probing on a single model checkpoint."""
    classes = DATASET_CLASSES[dataset]

    # Load model
    try:
        model = ContrastiveResNet50.load_from_checkpoint(ckpt_path, map_location=device)
    except Exception:
        model = ContrastiveResNet50()
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        
    model = model.to(device)
    model.eval()

    # DataModule with balanced ratios [0.5, 0.5]
    val_transform = SimCLRValTransform(img_height=224, normalize=None)
    dm = NClassesDataModule(
        data_dir=data_dir,
        classes=classes,
        class_ratios=[0.5, 0.5],
        train_transform=val_transform,
        val_transform=val_transform,
        batch_size=128,
        num_workers=4,
        subsample_balanced=True
    )
    dm.setup()

    # Extract test features
    test_loader = dm.test_dataloader()
    X_test, y_test = extract_features(model, test_loader, device)

    # Extract train features and subsample 1% balanced subset (Paper protocol S2.3)
    train_loader = dm.train_dataloader()
    X_train_full, y_train_full = extract_features(model, train_loader, device)

    np.random.seed(42)
    idx_0 = np.where(y_train_full == 0)[0]
    idx_1 = np.where(y_train_full == 1)[0]
    
    n_samples_per_class = max(10, int(len(idx_0) * 0.01))
    sel_0 = np.random.choice(idx_0, min(len(idx_0), n_samples_per_class), replace=False)
    sel_1 = np.random.choice(idx_1, min(len(idx_1), n_samples_per_class), replace=False)
    probe_idx = np.concatenate([sel_0, sel_1])
    
    X_probe = X_train_full[probe_idx]
    y_probe = y_train_full[probe_idx]

    # Fit balanced linear probe
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0, solver='lbfgs')
    clf.fit(X_probe, y_probe)

    # Predict
    y_pred = clf.predict(X_test)
    try:
        y_prob = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    except:
        auc = 0.5

    bal_acc = balanced_accuracy_score(y_test, y_pred)
    raw_acc = accuracy_score(y_test, y_pred)

    return bal_acc, auc, raw_acc


def df_to_markdown_simple(df):
    """Simple markdown table formatter that does not require tabulate."""
    headers = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join([str(val) for val in row]) + " |")
    return "\n".join(lines)


def run_all(datasets=None, ratios=None, data_dir="data/inat21"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 85)
    print(f"🌟 STARTING SYSTEMATIC LINEAR PROBE EVALUATION (Device: {device})")
    print("=" * 85)

    if datasets is None:
        datasets = ["plants", "insects", "animals"]
    if ratios is None:
        ratios = ["50_50", "95_5", "99_1"]

    ckpt_map = build_checkpoint_map("logs")

    results = {d: {r: {m: None for m in METHODS} for r in ratios} for d in datasets}
    auc_results = {d: {r: {m: None for m in METHODS} for r in ratios} for d in datasets}

    total_tasks = len(datasets) * len(ratios) * len(METHODS)
    current_task = 0

    for d in datasets:
        print(f"\n==================== DATASET: {d.upper()} ====================")
        for r in ratios:
            for m in METHODS:
                current_task += 1
                tag = f"{d}-{r}-{m}"
                ckpt = ckpt_map.get(tag)
                
                # Try finding with alternate formats
                if not ckpt:
                    for k, v in ckpt_map.items():
                        if d in k and r in k and m in k:
                            ckpt = v
                            break

                if ckpt is None or not os.path.exists(ckpt):
                    print(f"[{current_task}/{total_tasks}] ⚠️ Checkpoint NOT found for {tag}, skipping...")
                    continue

                print(f"[{current_task}/{total_tasks}] Evaluating: {tag} ({os.path.basename(ckpt)}) ...")
                try:
                    bal_acc, auc, raw_acc = evaluate_single_model(ckpt, d, data_dir=data_dir, device=device)
                    results[d][r][m] = bal_acc
                    auc_results[d][r][m] = auc
                    print(f"     👉 Balanced Acc: {bal_acc*100:.2f}% | AUC: {auc*100:.2f}% | Raw Acc: {raw_acc*100:.2f}%")
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"     ❌ Error evaluating {tag}: {e}")

    # Generate final tables
    os.makedirs("results", exist_ok=True)
    summary_md = "# TABLE 1: BALANCED LINEAR PROBING RESULTS (Reproduced vs Paper)\n\n"
    
    method_names = {
        "weightedce": "Weighted CE (Baseline)",
        "supcon": "Standard SupCon (Baseline)",
        "supmin": "Sup Minority (Ours)",
        "supproto": "Sup Prototypes (Ours)"
    }

    for d in datasets:
        summary_md += f"### {d.capitalize()} Subset\n\n"
        rows = []
        for m in METHODS:
            rows.append({
                "Method": method_names[m],
                "50:50 Balanced": f"{results[d]['50_50'][m]*100:.2f}%" if results[d]['50_50'][m] is not None else "N/A",
                "95:5 Imbalanced": f"{results[d]['95_5'][m]*100:.2f}%" if results[d]['95_5'][m] is not None else "N/A",
                "99:1 Extreme": f"{results[d]['99_1'][m]*100:.2f}%" if results[d]['99_1'][m] is not None else "N/A",
            })
        df = pd.DataFrame(rows)
        summary_md += df_to_markdown_simple(df) + "\n\n"
        print(f"\n=== SUMMARY FOR {d.upper()} ===")
        print(df.to_string(index=False))

    output_path = "results/table1_linear_probe_full.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary_md)

    print("\n" + "=" * 85)
    print(f"✅ LINEAR PROBE EVALUATION COMPLETE! Saved to {output_path}")
    print("=" * 85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["plants", "insects", "animals"])
    parser.add_argument("--ratios", nargs="+", default=["50_50", "95_5", "99_1"])
    parser.add_argument("--data_dir", type=str, default="data/inat21")
    args = parser.parse_args()

    run_all(datasets=args.datasets, ratios=args.ratios, data_dir=args.data_dir)
