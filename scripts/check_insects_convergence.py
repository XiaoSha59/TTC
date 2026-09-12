import os
import sys
import wandb
import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

os.environ['WANDB_API_KEY'] = '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'
api = wandb.Api()

print("=== [1] Pretraining: insects-95_5-supmin-350ep-full ===")
runs = api.runs('tnpdung79hcmus/binary-learning', filters={'display_name': 'insects-95_5-supmin-350ep-full'})
for r in runs:
    if r.state == 'finished':
        hist = r.history(samples=1000)
        for col in ['train.loss', 'val.loss', 'online_train_loss', 'online_val_loss', 'online_val_acc']:
            if col in hist.columns:
                df = hist[['epoch', col]].dropna().sort_values('epoch')
                if not df.empty:
                    if 'loss' in col:
                        idx = df[col].idxmin()
                        ep = int(df.loc[idx, 'epoch'])
                        val = df.loc[idx, col]
                        print(f"  • Lowest {col:<18}: {val:.4f} (Epoch {ep})")
                    else:
                        idx = df[col].idxmax()
                        ep = int(df.loc[idx, 'epoch'])
                        val = df.loc[idx, col] * 100
                        print(f"  • Peak   {col:<18}: {val:.2f}% (Epoch {ep})")

print("\n=== [2] Linear Probe: insects-95_5-supmin-full-probe ===")
runs_p = api.runs('tnpdung79hcmus/binary-learning', filters={'display_name': 'insects-95_5-supmin-full-probe'})
for r in runs_p:
    if r.state == 'finished':
        hist = r.history(samples=1000)
        for col in hist.columns:
            if any(k in col.lower() for k in ['acc', 'loss', 'auc']) and not col.startswith('_'):
                df = hist[['epoch', col]].dropna().sort_values('epoch')
                if not df.empty:
                    if 'loss' in col:
                        idx = df[col].idxmin()
                        ep = int(df.loc[idx, 'epoch'])
                        val = df.loc[idx, col]
                        print(f"  • Lowest {col:<22}: {val:.4f} (Epoch {ep})")
                    else:
                        idx = df[col].idxmax()
                        ep = int(df.loc[idx, 'epoch'])
                        val = df.loc[idx, col] * 100 if df.loc[idx, col] <= 1.0 else df.loc[idx, col]
                        print(f"  • Peak   {col:<22}: {val:.2f}% (Epoch {ep})")
