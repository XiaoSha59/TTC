import os
import sys
import glob
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
from torch.utils.data import DataLoader

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sup_cont import ContrastiveResNet50
from data.data_module import MedMNISTDataModule, SimpleImageDataModule
from torchvision import transforms

def evaluate_medical_checkpoint(ckpt_path, dataset_name, data_dir="data"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating {os.path.basename(ckpt_path)} on {dataset_name} (Device: {device})...")
    
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

    # Data loaders
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    if dataset_name in ["breast", "pneumonia"]:
        dm = MedMNISTDataModule(
            data_dir=data_dir,
            data_set=dataset_name,
            train_transform=val_transform,
            val_transform=val_transform,
            batch_size=128,
            num_workers=4
        )
    elif dataset_name == "fracatlas":
        frac_dir = os.path.join(data_dir, "FracAtlas")
        dm = SimpleImageDataModule(
            train_dir=os.path.join(frac_dir, "train"),
            val_dir=os.path.join(frac_dir, "val"),
            test_dir=os.path.join(frac_dir, "test"),
            train_transform=val_transform,
            val_transform=val_transform,
            batch_size=128,
            num_workers=4
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dm.setup()
    
    # Extract features
    def get_feats(loader):
        feats, targets = [], []
        with torch.no_grad():
            for batch in loader:
                x, y = batch[0].to(device), batch[1]
                if hasattr(model, 'base_encoder'):
                    f = model.base_encoder(x)
                else:
                    f = model(x)
                if isinstance(f, dict):
                    f = f.get('feats', f.get('features', f))
                elif isinstance(f, (tuple, list)):
                    f = f[0]
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

    auc = roc_auc_score(y_test, y_prob)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    print(f"  ⭐ AUROC (Paper Metric): {auc*100:.2f}% | Balanced Acc: {bal_acc*100:.2f}% | Raw Acc: {acc*100:.2f}%")
    return auc, bal_acc, acc

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=["breast", "pneumonia", "fracatlas"])
    args = parser.parse_args()
    evaluate_medical_checkpoint(args.ckpt, args.dataset)
