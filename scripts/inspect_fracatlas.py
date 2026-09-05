import os
import sys
import wandb

sys.stdout.reconfigure(encoding='utf-8')
os.environ['WANDB_API_KEY'] = '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'
api = wandb.Api()
runs = api.runs('tnpdung79hcmus/binary-learning', order='-created_at', per_page=60)

print(f"{'Run Name':<30} | {'test.acc':<10} | {'test.auc':<10} | {'online_val_acc':<16} | {'val.auroc':<10}")
print("-" * 80)
for r in runs:
    if 'fracatlas' in r.name.lower():
        s = r.summary._json_dict
        t_acc = s.get('test.acc')
        t_auc = s.get('test.auc')
        v_acc = s.get('online_val_acc') or s.get('val.acc')
        v_auc = s.get('val.auroc')
        
        t_acc_s = f"{t_acc*100:.2f}%" if t_acc is not None else "None"
        t_auc_s = f"{t_auc*100:.2f}%" if t_auc is not None else "None"
        v_acc_s = f"{v_acc*100:.2f}%" if v_acc is not None else "None"
        v_auc_s = f"{v_auc*100:.2f}%" if v_auc is not None else "None"
        
        print(f"{r.name:<30} | {t_acc_s:<10} | {t_auc_s:<10} | {v_acc_s:<16} | {v_auc_s:<10}")
