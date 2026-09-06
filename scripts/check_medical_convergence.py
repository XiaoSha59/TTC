import os
import wandb
import pandas as pd
import numpy as np

os.environ['WANDB_API_KEY'] = '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'

def check_medical():
    api = wandb.Api()
    runs = api.runs('tnpdung79hcmus/binary-learning', per_page=120)
    
    med_runs = [r for r in runs if any(m in r.name.lower() for m in ['breastmnist', 'pneumoniamnist', 'fracatlas']) and r.state == 'finished']
    print(f"Analyzing {len(med_runs)} finished medical runs...\n")
    
    records = []
    for r in sorted(med_runs, key=lambda x: x.name):
        history = r.history()
        if history.empty or 'epoch' not in history.columns:
            continue
            
        # Find loss and accuracy metrics
        loss_col = None
        for c in ['val.loss', 'val/loss', 'online_val_loss']:
            if c in history.columns:
                loss_col = c
                break
                
        acc_col = None
        for c in ['online_val_acc', 'val.acc', 'val/acc', 'val.auroc']:
            if c in history.columns:
                acc_col = c
                break
                
        total_epochs = int(history['epoch'].max())
        
        best_loss_ep = None
        min_loss = None
        if loss_col:
            v = history.dropna(subset=[loss_col, 'epoch'])
            if not v.empty:
                idx = v[loss_col].idxmin()
                best_loss_ep = int(v.loc[idx, 'epoch'])
                min_loss = v.loc[idx, loss_col]
                
        best_acc_ep = None
        max_acc = None
        if acc_col:
            v = history.dropna(subset=[acc_col, 'epoch'])
            if not v.empty:
                idx = v[acc_col].idxmax()
                best_acc_ep = int(v.loc[idx, 'epoch'])
                max_acc = v.loc[idx, acc_col]
                
        # Plateau analysis: Find epoch where loss reached within 5% of min_loss
        plateau_ep = None
        if loss_col and min_loss is not None:
            v = history.dropna(subset=[loss_col, 'epoch'])
            thresh = min_loss + 0.05 * (v[loss_col].iloc[0] - min_loss)
            plateau_rows = v[v[loss_col] <= thresh]
            if not plateau_rows.empty:
                plateau_ep = int(plateau_rows.iloc[0]['epoch'])
                
        records.append({
            'name': r.name,
            'total_ep': total_epochs,
            'best_loss_ep': best_loss_ep,
            'min_loss': f"{min_loss:.4f}" if min_loss is not None else "N/A",
            'plateau_95%_ep': plateau_ep,
            'best_acc_ep': best_acc_ep,
            'max_metric': f"{max_acc*100:.2f}%" if max_acc is not None else "N/A",
        })
        
    df = pd.DataFrame(records)
    print(f"{'Run Name':<30} | {'Total':<6} | {'Plateau Ep':<11} | {'Min Loss Ep':<12} | {'Min Loss':<10} | {'Best Metric Ep':<15} | {'Peak Metric':<10}")
    print("-" * 110)
    for _, row in df.iterrows():
        print(f"{row['name']:<30} | {row['total_ep']:<6} | {str(row['plateau_95%_ep']):<11} | {str(row['best_loss_ep']):<12} | {str(row['min_loss']):<10} | {str(row['best_acc_ep']):<15} | {str(row['max_metric']):<10}")

if __name__ == '__main__':
    check_medical()
