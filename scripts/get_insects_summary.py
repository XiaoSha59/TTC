import os
import sys
import wandb

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

os.environ['WANDB_API_KEY'] = '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'
api = wandb.Api()

# Fetch latest runs
runs = api.runs('tnpdung79hcmus/binary-learning', order='-created_at', per_page=40)

print(f"{'Run Name':<45} | {'State':<10} | {'Ep':<4} | {'Test Acc':<10} | {'Val Acc':<10} | {'Val Loss':<10}")
print("-" * 95)

insect_runs = []
for r in runs:
    if 'insect' in r.name.lower() or 'probe' in r.name.lower() or 'inat' in r.name.lower():
        s = r.summary._json_dict
        ep = s.get('epoch', '?')
        test_acc = s.get('test.acc') or s.get('acc.test') or s.get('test_acc')
        test_acc_s = f"{test_acc*100:.2f}%" if (isinstance(test_acc, (int, float)) and test_acc is not None) else "-"
        
        val_acc = s.get('online_val_acc') or s.get('val.acc') or s.get('val_acc')
        val_acc_s = f"{val_acc*100:.2f}%" if (isinstance(val_acc, (int, float)) and val_acc is not None) else "-"
        
        vloss = s.get('val.loss') or s.get('val_loss') or s.get('online_val_loss')
        vloss_s = f"{vloss:.4f}" if isinstance(vloss, (int, float)) else "-"
        
        print(f"{r.name:<45} | {r.state:<10} | {str(ep):<4} | {test_acc_s:<10} | {val_acc_s:<10} | {vloss_s:<10}")
        insect_runs.append((r.name, r.state, ep, test_acc_s, val_acc_s, vloss_s))
