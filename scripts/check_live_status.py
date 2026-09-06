#!/usr/bin/env python3
import os
import sys
import subprocess

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

os.environ['WANDB_API_KEY'] = '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'

print("=" * 65)
print("🔍 KIỂM TRA TRẠNG THÁI TIẾN TRÌNH THỜI GIAN THỰC (GCP VM & KAGGLE)")
print("=" * 65)

# 1. Kiểm tra GCP VM qua WandB
try:
    import wandb
    api = wandb.Api()
    runs = api.runs('tnpdung79hcmus/binary-learning', order='-created_at')
    
    print("\n[1] TRẠNG THÁI TRÊN GCP VM (GPU NVIDIA L4):")
    gcp_runs = [r for r in runs if 'insects' in r.name.lower()][:3]
    for r in gcp_runs:
        ep = r.summary.get('epoch', 0)
        status_icon = "🟢 ĐANG CHẠY" if r.state == "running" else "✅ HOÀN TẤT" if r.state == "finished" else "❌ LỖI"
        print(f"  • {r.name:<32} | {status_icon:<12} | Epoch: {ep}/350")
except Exception as e:
    print(f"  Không thể kết nối WandB: {e}")

# 2. Kiểm tra Kaggle qua CLI
print("\n[2] TRẠNG THÁI TRÊN KAGGLE (GPU TESLA T4):")
try:
    res = subprocess.run(
        ["python", "-m", "kaggle", "kernels", "status", "salala1706/ttc-medical-weighted-ce"],
        capture_output=True, text=True
    )
    output = res.stdout.strip()
    if "RUNNING" in output:
        k_status = "🟢 ĐANG CHẠY (KernelWorkerStatus.RUNNING)"
    elif "COMPLETE" in output:
        k_status = "✅ HOÀN THÀNH (KernelWorkerStatus.COMPLETE)"
    else:
        k_status = output
    print(f"  • salala1706/ttc-medical-weighted-ce: {k_status}")
except Exception as e:
    print(f"  Không thể gọi Kaggle CLI: {e}")

print("\n" + "=" * 65)
