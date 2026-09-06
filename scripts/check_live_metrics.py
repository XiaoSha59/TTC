import glob
import os
import pandas as pd

log_dirs = sorted(glob.glob("/home/tnpdung_79/TTC/logs/train/runs/*"), key=os.path.getmtime)
if not log_dirs:
    print("No run directories found.")
    exit(0)

latest_dir = log_dirs[-1]
print(">>> Latest Run Directory:", latest_dir)

csv_path = os.path.join(latest_dir, "csv", "version_0", "metrics.csv")
if not os.path.exists(csv_path):
    print("No CSV metrics found at", csv_path)
    exit(0)

df = pd.read_csv(csv_path)
val_cols = [c for c in df.columns if any(k in c for k in ["epoch", "valid", "loss", "auc", "acc", "lr"])]

if "loss_valid" in df.columns:
    df_val = df.dropna(subset=["loss_valid"])
    print("\n--- Recent Validation Metrics (Tail 15) ---")
    cols_to_show = [c for c in ["epoch", "loss_valid", "acc_valid", "auc_valid", "loss_train", "lr-Adam", "lr-SGD"] if c in df.columns]
    print(df_val[cols_to_show].tail(15).to_string(index=False))

    print("\n--- Peak Metrics Overview ---")
    min_loss_row = df.loc[df["loss_valid"].idxmin()]
    print(f"Lowest Val Loss: {min_loss_row['loss_valid']:.4f} at Epoch {int(min_loss_row['epoch'])}")

    if "auc_valid" in df.columns and not df["auc_valid"].isna().all():
        max_auc_row = df.loc[df["auc_valid"].idxmax()]
        print(f"Highest Val AUC: {max_auc_row['auc_valid']*100:.2f}% at Epoch {int(max_auc_row['epoch'])}")
    if "acc_valid" in df.columns and not df["acc_valid"].isna().all():
        max_acc_row = df.loc[df["acc_valid"].idxmax()]
        print(f"Highest Val Acc: {max_acc_row['acc_valid']*100:.2f}% at Epoch {int(max_acc_row['epoch'])}")
else:
    print("Columns available:", df.columns.tolist())
