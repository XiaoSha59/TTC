import os
import wandb

os.environ['WANDB_API_KEY'] = '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'
api = wandb.Api()
runs = api.runs('tnpdung79hcmus/binary-learning', per_page=150)

print(f"{'Run Name':<45} | {'State':<9} | {'Ep':<4} | {'Test Acc':<9} | {'Val AUC':<9}")
print("-" * 85)

for r in runs:
    if 'insects' in r.name.lower():
        cfg = r.config
        ep = cfg.get('trainer', {}).get('max_epochs', '?') if isinstance(cfg.get('trainer'), dict) else '?'
        s = r.summary._json_dict
        acc = s.get('test.acc') or s.get('test/acc') or s.get('acc.test')
        auc = s.get('val.auroc') or s.get('test.auc')
        acc_s = f"{acc*100:.2f}%" if acc is not None else "-"
        auc_s = f"{auc*100:.2f}%" if auc is not None else "-"
        print(f"{r.name:<45} | {r.state:<9} | {str(ep):<4} | {acc_s:<9} | {auc_s:<9}")
