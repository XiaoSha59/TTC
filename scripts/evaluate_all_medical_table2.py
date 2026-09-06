import os
import sys
import glob
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
from torchvision import transforms
from torch.utils.data import DataLoader

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sup_cont import ContrastiveResNet50
from data.data_module import MedMNISTDataModule, SimpleImageDataModule

CHECKPOINTS = {
    "BreastMNIST": {
        "Weighted CE (Baseline)": "/home/tnpdung_79/TTC/logs/train/runs/2026-09-03_18-13-07/checkpoints/last.ckpt",
        "Standard SupCon (Baseline)": "/home/tnpdung_79/TTC/logs/train/runs/2026-09-03_17-53-33/checkpoints/last.ckpt",
        "Sup Minority (Ours)": "/home/tnpdung_79/TTC/logs/train/runs/2026-09-03_17-27-32/checkpoints/last.ckpt",
        "Sup Prototypes (Ours)": "/home/tnpdung_79/TTC/logs/train/runs/2026-09-03_17-40-27/checkpoints/last.ckpt",
    },
    "PneumoniaMNIST": {
        "Weighted CE (Baseline)": "/home/tnpdung_79/TTC/logs/train/runs/2026-09-03_22-25-41/checkpoints/last.ckpt",
        "Standard SupCon (Baseline)": "/home/tnpdung_79/TTC/logs/train/runs/2026-09-03_21-15-30/checkpoints/last.ckpt",
        "Sup Minority (Ours)": "/home/tnpdung_79/TTC/logs/train/runs/2026-09-03_18-30-05/checkpoints/last.ckpt",
        "Sup Prototypes (Ours)": "/home/tnpdung_79/TTC/logs/train/runs/2026-09-03_20-11-28/checkpoints/last.ckpt",
    },
    "FracAtlas": {
        "Weighted CE (Baseline)": "/home/tnpdung_79/TTC/logs/train/runs/2026-09-04_02-53-18/checkpoints/last.ckpt",
        "Standard SupCon (Baseline)": "/home/tnpdung_79/TTC/logs/train/runs/2026-09-04_01-35-25/checkpoints/last.ckpt",
        "Sup Minority (Ours)": "/home/tnpdung_79/TTC/logs/train/runs/2026-09-03_22-58-59/checkpoints/last.ckpt",
        "Sup Prototypes (Ours)": "/home/tnpdung_79/TTC/logs/train/runs/2026-09-04_00-17-05/checkpoints/last.ckpt",
    }
}

PAPER_TARGETS = {
    "BreastMNIST": {
        "Weighted CE (Baseline)": "75.1%",
        "Standard SupCon (Baseline)": "75.1%",
        "Sup Minority (Ours)": "86.4%",
        "Sup Prototypes (Ours)": "90.7%"
    },
    "PneumoniaMNIST": {
        "Weighted CE (Baseline)": "98.8%",
        "Standard SupCon (Baseline)": "99.5%",
        "Sup Minority (Ours)": "99.6%",
        "Sup Prototypes (Ours)": "99.8%"
    },
    "FracAtlas": {
        "Weighted CE (Baseline)": "79.8%",
        "Standard SupCon (Baseline)": "84.8%",
        "Sup Minority (Ours)": "82.3%",
        "Sup Prototypes (Ours)": "86.0%"
    }
}

def evaluate_single(ckpt_path, dataset_key, data_dir="/home/tnpdung_79/TTC/data", device="cpu"):
    print(f"==> Loading: {os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))} ({dataset_key})")
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    if dataset_key == "BreastMNIST":
        dm = MedMNISTDataModule(
            data_dir=os.path.join(data_dir, "medmnist"),
            data_set="breast",
            train_transform=val_transform,
            val_transform=val_transform,
            batch_size=128,
            num_workers=4
        )
    elif dataset_key == "PneumoniaMNIST":
        dm = MedMNISTDataModule(
            data_dir=os.path.join(data_dir, "medmnist"),
            data_set="pneumonia",
            train_transform=val_transform,
            val_transform=val_transform,
            batch_size=128,
            num_workers=4
        )
    elif dataset_key == "FracAtlas":
        frac_dir = os.path.join(data_dir, "fracatlas_splits")
        dm = SimpleImageDataModule(
            train_dir=os.path.join(frac_dir, "train"),
            val_dir=os.path.join(frac_dir, "val"),
            test_dir=os.path.join(frac_dir, "test"),
            train_transform=val_transform,
            val_transform=val_transform,
            batch_size=128,
            num_workers=4
        )
    dm.setup()

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    
    is_weighted_ce = any("classifier" in k or "fc" in k for k in state_dict.keys()) and not any("projection_head" in k for k in state_dict.keys())

    if is_weighted_ce:
        from torchvision.models import resnet50
        model = resnet50(weights=None)
        cleaned_sd = {}
        for k, v in state_dict.items():
            k_clean = k.replace("model.", "").replace("net.", "").replace("base_encoder.", "")
            cleaned_sd[k_clean] = v
        model.load_state_dict(cleaned_sd, strict=False)
        encoder = torch.nn.Sequential(*list(model.children())[:-1]).to(device)
    else:
        model = ContrastiveResNet50()
        model.load_state_dict(state_dict, strict=False)
        encoder = model.base_encoder.to(device)
    
    encoder.eval()

    def get_feats(loader):
        feats, targets = [], []
        with torch.no_grad():
            for batch in loader:
                x, y = batch[0].to(device), batch[1]
                f = encoder(x)
                f = f.view(f.size(0), -1)
                feats.append(f.cpu().numpy())
                targets.append(y.flatten().cpu().numpy())
        return np.concatenate(feats, axis=0), np.concatenate(targets, axis=0)

    X_train, y_train = get_feats(dm.train_dataloader())
    X_test, y_test = get_feats(dm.test_dataloader())

    clf = LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0, solver='lbfgs')
    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    auc = roc_auc_score(y_test, y_prob) * 100
    bal_acc = balanced_accuracy_score(y_test, y_pred) * 100
    acc = accuracy_score(y_test, y_pred) * 100
    print(f"   --> AUROC: {auc:.2f}% | BalAcc: {bal_acc:.2f}% | RawAcc: {acc:.2f}%")
    return auc, bal_acc, acc

def main():
    device = "cpu"
    print(f"Evaluating Table 2 Medical AUROC on {device.upper()}...")

    results = {}
    for dset, methods in CHECKPOINTS.items():
        results[dset] = {}
        for method, ckpt in methods.items():
            if not os.path.exists(ckpt):
                print(f"Warning: ckpt not found: {ckpt}")
                continue
            auc, bal_acc, acc = evaluate_single(ckpt, dset, device=device)
            results[dset][method] = {
                "auroc": auc,
                "bal_acc": bal_acc,
                "acc": acc,
                "paper": PAPER_TARGETS[dset].get(method, "N/A")
            }

    print("\n" + "="*80)
    print("OFFICIAL TABLE 2 REPRODUCTION RESULTS (AUROC %)")
    print("="*80)
    methods_list = [
        "Weighted CE (Baseline)",
        "Standard SupCon (Baseline)",
        "Sup Minority (Ours)",
        "Sup Prototypes (Ours)"
    ]
    header = f"{'Method':<28} | {'Breast (Ours)':<13} | {'Breast (Ref)':<12} | {'Pneumonia (Ours)':<16} | {'Pneumonia (Ref)':<15} | {'FracAtlas (Ours)':<16} | {'FracAtlas (Ref)':<15}"
    print(header)
    print("-" * len(header))
    for m in methods_list:
        b_o = f"{results['BreastMNIST'].get(m, {}).get('auroc', 0.0):.2f}%"
        b_p = PAPER_TARGETS['BreastMNIST'].get(m, 'N/A')
        p_o = f"{results['PneumoniaMNIST'].get(m, {}).get('auroc', 0.0):.2f}%"
        p_p = PAPER_TARGETS['PneumoniaMNIST'].get(m, 'N/A')
        f_o = f"{results['FracAtlas'].get(m, {}).get('auroc', 0.0):.2f}%"
        f_p = PAPER_TARGETS['FracAtlas'].get(m, 'N/A')
        print(f"{m:<28} | {b_o:<13} | {b_p:<12} | {p_o:<16} | {p_p:<15} | {f_o:<16} | {f_p:<15}")

if __name__ == '__main__':
    main()
