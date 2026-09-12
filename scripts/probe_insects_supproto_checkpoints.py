import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score
from torchvision import transforms
from torch.utils.data import DataLoader

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sup_cont import ContrastiveResNet50
from data.iNatData import INaturalistNClasses

CHECKPOINTS = {
    "Insects 50:50 Sup-Prototypes (Ours)": {
        "ckpt": "/home/tnpdung_79/TTC/logs/train/runs/2026-09-06_01-53-15/checkpoints/last.ckpt",
        "class_ratios": [0.5, 0.5],
        "paper_target": "83.5%"
    },
    "Insects 99:1 Sup-Prototypes (Ours)": {
        "ckpt": "/home/tnpdung_79/TTC/logs/train/runs/2026-09-06_04-10-39/checkpoints/last.ckpt",
        "class_ratios": [0.01, 0.99],
        "paper_target": "78.9%"
    }
}

DATA_DIR = "/home/tnpdung_79/TTC/data/inat21"
CLASSES = [
    "Animalia_Arthropoda_Insecta_Hymenoptera_Apidae",
    "Animalia_Arthropoda_Insecta_Hymenoptera_Vespidae"
]

def evaluate_checkpoint(name, info, device="cuda" if torch.cuda.is_available() else "cpu"):
    ckpt_path = info["ckpt"]
    paper_target = info["paper_target"]
    
    print(f"\n{'='*75}")
    print(f"🚀 EVALUATING: {name}")
    print(f" Checkpoint : {ckpt_path}")
    print(f" Target Paper : {paper_target}")
    print(f" Device     : {device.upper()}")
    print(f"{'='*75}")
    
    if not os.path.exists(ckpt_path):
        print(f"❌ Error: Checkpoint not found at {ckpt_path}")
        return None
        
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load Train and Val/Test datasets
    train_ds = INaturalistNClasses(DATA_DIR, split="train", transform=val_transform, classes=CLASSES)
    val_ds = INaturalistNClasses(DATA_DIR, split="val", transform=val_transform, classes=CLASSES)
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2)
    
    print(f">>> Train dataset samples: {len(train_ds)}")
    print(f">>> Test/Val dataset samples: {len(val_ds)}")
    
    # Load Model
    print(">>> Loading backbone checkpoint...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    
    model = ContrastiveResNet50()
    model.load_state_dict(state_dict, strict=False)
    encoder = model.base_encoder.to(device)
    encoder.eval()
    
    def extract_features(loader):
        feats, labels = [], []
        with torch.no_grad():
            for imgs, targets in loader:
                imgs = imgs.to(device)
                f = encoder(imgs)
                f = f.view(f.size(0), -1)
                feats.append(f.cpu().numpy())
                labels.append(targets.cpu().numpy())
        return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)
        
    print(">>> Extracting representations on train & test sets...")
    X_train, y_train = extract_features(train_loader)
    X_test, y_test = extract_features(val_loader)
    
    # 1. Standard Linear Probe / Logistic Regression
    print(">>> Training Linear Probe Classifier on frozen representations...")
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0, solver='lbfgs')
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    bal_acc = balanced_accuracy_score(y_test, y_pred) * 100
    raw_acc = accuracy_score(y_test, y_pred) * 100
    auc = roc_auc_score(y_test, y_prob) * 100
    
    print(f"\n🎉 RESULTS FOR {name}:")
    print(f"  • Test Balanced Accuracy : {bal_acc:.2f}% (Paper: {paper_target})")
    print(f"  • Test Raw Accuracy      : {raw_acc:.2f}%")
    print(f"  • Test AUROC             : {auc:.2f}%")
    
    return bal_acc, raw_acc, auc

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    for name, info in CHECKPOINTS.items():
        res = evaluate_checkpoint(name, info, device=device)
        if res is not None:
            results[name] = res
            
    print("\n" + "="*80)
    print("🏁 SUMMARY OF EVALUATION RESULTS (TABLE 1)")
    print("="*80)
    for name, (bal_acc, raw_acc, auc) in results.items():
        target = CHECKPOINTS[name]["paper_target"]
        diff = bal_acc - float(target.replace("%", ""))
        status = f"🔥 VƯỢT PAPER (+{diff:.2f}%)" if diff >= 0 else f"Tiệm cận ({diff:.2f}%)"
        print(f"{name:<40} | Balanced Acc: {bal_acc:.2f}% | Paper: {target:<6} | Status: {status}")

if __name__ == "__main__":
    main()
