import os
import sys
import wandb
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding='utf-8')
os.environ['WANDB_API_KEY'] = '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'
api = wandb.Api()
runs = api.runs('tnpdung79hcmus/binary-learning', order='-created_at', per_page=25)
tz_vn = timezone(timedelta(hours=7))

print(f"=== TIEN TRINH ANIMALS (CAP NHAT LUC {datetime.now(tz_vn).strftime('%H:%M:%S %d/%m')}) ===")
header = f"{'Run Name':<35} | {'State':<10} | {'Bat dau':<12} | {'Epoch':<6} | {'Val Acc'}"
print(header)
print("-" * len(header))

animals_runs = []
for r in runs:
    if "animals" in r.name.lower():
        animals_runs.append(r)

for r in animals_runs:
    dt = datetime.fromisoformat(r.created_at.replace('Z', '+00:00')).astimezone(tz_vn)
    ep = r.summary.get('epoch', 'N/A')
    acc = r.summary.get('online_val_acc') or r.summary.get('test.acc') or r.summary.get('val.acc')
    acc_str = f"{acc*100:.2f}%" if acc is not None else "N/A"
    print(f"{r.name:<35} | {r.state:<10} | {dt.strftime('%d/%m %H:%M'):<12} | {str(ep):<6} | {acc_str}")
