import os
os.environ["HOME"] = "/tmp"
os.environ["WANDB_DIR"] = "/tmp"
os.environ["WANDB_CONFIG_DIR"] = "/tmp"
os.environ["WANDB_CACHE_DIR"] = "/tmp"
os.environ["WANDB_API_KEY"] = "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"
import wandb
import pandas as pd

try:
    api = wandb.Api()
    run = api.run("tnpdung79hcmus/binary-learning/p2sspiez")
    print(">>> Run Name:", run.name)
    print(">>> State:", run.state)
    print("\n--- Summary Metrics ---")
    for k, v in run.summary.items():
        if not k.startswith("_"):
            print(f"  {k}: {v}")
            
    hist = run.history(samples=1000)
    if not hist.empty:
        cols = [c for c in hist.columns if any(k in c.lower() for k in ["epoch", "valid", "loss", "auc", "acc", "lr"])]
        print(f"\n--- History (Total recorded points: {len(hist)}) ---")
        df_clean = hist[cols].dropna(subset=[c for c in cols if c != "epoch"], how="all")
        print(df_clean.tail(15).to_string(index=False))
        
        # Check convergence trend
        if "loss_valid" in df_clean.columns:
            min_loss_row = df_clean.loc[df_clean["loss_valid"].idxmin()]
            print(f"\n>>> Lowest Val Loss: {min_loss_row['loss_valid']:.4f} at Epoch {min_loss_row.get('epoch', 'N/A')}")
        if "auc_valid" in df_clean.columns:
            max_auc_row = df_clean.loc[df_clean["auc_valid"].idxmax()]
            print(f">>> Peak Val AUC: {max_auc_row['auc_valid']*100:.2f}% at Epoch {max_auc_row.get('epoch', 'N/A')}")
        if "acc_valid" in df_clean.columns:
            max_acc_row = df_clean.loc[df_clean["acc_valid"].idxmax()]
            print(f">>> Peak Val Acc: {max_acc_row['acc_valid']*100:.2f}% at Epoch {max_acc_row.get('epoch', 'N/A')}")
except Exception as e:
    print("Error querying WandB:", e)
