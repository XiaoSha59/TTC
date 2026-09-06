import os
os.environ["HOME"] = "/tmp"
os.environ["WANDB_DIR"] = "/tmp"
os.environ["WANDB_CONFIG_DIR"] = "/tmp"
os.environ["WANDB_CACHE_DIR"] = "/tmp"
os.environ["WANDB_API_KEY"] = "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"

import wandb
import pandas as pd

run = wandb.Api().run("tnpdung79hcmus/binary-learning/p2sspiez")
df = run.history(samples=10000)

cols = ["train.loss", "val.loss", "online_train_acc", "lr-SGD"]
agg_dict = {c: "last" for c in cols if c in df.columns}
df_grouped = df.groupby("epoch").agg(agg_dict).dropna(subset=["train.loss"], how="all")

milestones = [1, 20, 50, 80, 100, 140, 180, 210, 240, 260, int(df_grouped.index.max())]
available_milestones = [m for m in milestones if m in df_grouped.index]

print("\n=== MILESTONES PROGRESSION (Epochs vs Metrics) ===")
print(df_grouped.loc[available_milestones].to_string())

print("\n=== SUMMARY AT CURRENT EPOCH ===")
latest_epoch = int(df_grouped.index.max())
latest_data = df_grouped.loc[latest_epoch]
print(f"Current Epoch: {latest_epoch} / 350")
print(f"Learning Rate: {latest_data.get('lr-SGD', 0):.6f}")
print(f"Train Loss: {latest_data.get('train.loss', 0):.4f}")
print(f"Val Loss: {latest_data.get('val.loss', 0):.4f}")
print(f"Online Train Acc: {latest_data.get('online_train_acc', 0)*100:.2f}%")
