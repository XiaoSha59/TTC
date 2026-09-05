import os, wandb

os.environ['WANDB_API_KEY'] = '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'
api = wandb.Api()
runs = api.runs('tnpdung79hcmus/binary-learning', per_page=150)

data = {}
for r in runs:
    name = r.name.lower().strip()
    if 'finetune' in name:
        s = r.summary._json_dict
        acc = s.get('test.acc')
        auc = s.get('test.auc')
        val_auc = s.get('val.auroc')
        if acc is not None:
            k = name.replace('finetune-', '')
            if k not in data:
                data[k] = {'acc': acc, 'auc': auc, 'val_auc': val_auc}

print("="*80)
print(f"{'MODEL NAME':32s} | {'TEST ACC':10s} | {'TEST AUC':10s} | {'VAL AUROC':10s}")
print("="*80)
for k in sorted(data.keys()):
    v = data[k]
    acc_pct = f"{v['acc']*100:.2f}%" if v['acc'] is not None else 'N/A'
    auc_pct = f"{v['auc']*100:.2f}%" if v['auc'] is not None else 'N/A'
    val_auc_pct = f"{v['val_auc']*100:.2f}%" if v['val_auc'] is not None else 'N/A'
    print(f"{k:32s} | {acc_pct:10s} | {auc_pct:10s} | {val_auc_pct:10s}")
print("="*80)
