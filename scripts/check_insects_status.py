import os
import sys
import wandb

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

os.environ['WANDB_API_KEY'] = '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'
api = wandb.Api()
runs = api.runs('tnpdung79hcmus/binary-learning', order='-created_at', per_page=100)

print("=" * 90)
print(f"{'Run Name':<42} | {'State':<9} | {'Ep':<4} | {'Test/Val Acc':<12} | {'Val Loss'}")
print("=" * 90)

for r in runs:
    name = r.name
    if any(k in name.lower() for k in ['insect', 'inat', 'natural', 'probe']):
        s = r.summary._json_dict
        ep = s.get('epoch', '?')
        acc = s.get('test.acc') or s.get('online_val_acc') or s.get('acc.test') or s.get('val.acc') or s.get('test_acc')
        acc_s = f"{acc*100:.2f}%" if (isinstance(acc, (int, float)) and acc is not None) else str(acc)
        vloss = s.get('val.loss') or s.get('val_loss') or s.get('online_val_loss')
        vloss_s = f"{vloss:.4f}" if isinstance(vloss, (int, float)) else str(vloss)
        print(f"{name:<42} | {r.state:<9} | {str(ep):<4} | {acc_s:<12} | {vloss_s}")

print("=" * 90)
