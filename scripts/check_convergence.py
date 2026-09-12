import os
import sys
import wandb
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

os.environ['WANDB_API_KEY'] = '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'
api = wandb.Api()
runs = api.runs('tnpdung79hcmus/binary-learning')

targets = [
    'insects-95_5-supmin-350ep-full',
    'insects-95_5-supproto-350ep-full',
    'insects-50_50-supproto-350ep',
    'insects-99_1-supproto-350ep'
]

for r in runs:
    if r.name in targets:
        print(f"\n=======================================================")
        print(f"📊 MODEL: {r.name} (ID: {r.id}, State: {r.state})")
        print(f"=======================================================")
        rows = []
        for row in r.scan_history(keys=['epoch', 'train.loss.epoch', 'train.loss', 'val.loss']):
            rows.append(row)
        if not rows:
            print("No history available.")
            continue
        hist = pd.DataFrame(rows)
        
        loss_col = 'train.loss.epoch' if 'train.loss.epoch' in hist.columns else 'train.loss'
        if loss_col not in hist.columns:
            loss_col = hist.columns[1]
            
        hist_clean = hist.dropna(subset=['epoch', loss_col]).sort_values('epoch')
        if len(hist_clean) > 0:
            initial_loss = hist_clean[loss_col].iloc[0]
            final_loss = hist_clean[loss_col].iloc[-1]
            min_loss_row = hist_clean.loc[hist_clean[loss_col].idxmin()]
            min_epoch = int(min_loss_row['epoch'])
            min_loss = min_loss_row[loss_col]
            
            # Find convergence plateau (where loss is within 5% of min loss)
            threshold = min_loss * 1.05 if min_loss > 0 else min_loss + 0.05
            converged_rows = hist_clean[hist_clean[loss_col] <= threshold]
            converge_epoch = int(converged_rows['epoch'].iloc[0]) if not converged_rows.empty else min_epoch
            
            print(f"Initial Loss (Epoch 0)   : {initial_loss:.4f}")
            print(f"Final Loss (Epoch 349)   : {final_loss:.4f}")
            print(f"Minimum Loss Reached     : {min_loss:.4f} tại Epoch {min_epoch}")
            print(f"🎯 Điểm bắt đầu Hội Tụ (Plateau): Epoch ~{converge_epoch} - {min_epoch}")
            
            # Milestones
            print("\nTiến trình giảm Loss qua các mốc:")
            milestones = [0, 50, 100, 150, 200, 250, 300, 349]
            for m in milestones:
                row = hist_clean[hist_clean['epoch'] >= m].head(1)
                if not row.empty:
                    ep = int(row['epoch'].values[0])
                    val = row[loss_col].values[0]
                    print(f"  • Epoch {ep:<3}: Loss = {val:.4f}")
