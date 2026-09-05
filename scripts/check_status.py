import os
import wandb

os.environ['WANDB_API_KEY'] = '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'
api = wandb.Api()
runs = api.runs('tnpdung79hcmus/binary-learning', order='-created_at', per_page=40)

print(f"{'Run Name':<45} | {'State':<10} | {'Created At':<22} | {'Val Acc / Metric'}")
print("-" * 95)
for r in runs[:20]:
    acc = r.summary.get("online_val_acc") or r.summary.get("test.acc") or r.summary.get("val.acc")
    acc_str = f"{acc*100:.2f}%" if acc is not None else "N/A"
    print(f"{r.name:<45} | {r.state:<10} | {r.created_at:<22} | {acc_str}")
