import os
import wandb
import json

os.environ['WANDB_API_KEY'] = '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'
api = wandb.Api()
runs = api.runs('tnpdung79hcmus/binary-learning', order='-created_at')

for r in runs:
    if 'supproto-probe' in r.name.lower() or 'supproto-full-probe' in r.name.lower():
        print(f"=== Run Name: {r.name} ===")
        print(f"State       : {r.state}")
        print(f"Created At  : {r.created_at}")
        print(f"Base Model  : {r.config.get('base_model_path')}")
        print(f"Batch Size  : {r.config.get('batch_size')}")
        print(f"LR          : {r.config.get('module', {}).get('lr', r.config.get('module.lr'))}")
        print("Summary Metrics:")
        for k in ['test.acc', 'val.acc', 'test.auc', 'val.auroc', 'test.loss', 'val.loss']:
            if k in r.summary:
                print(f"  {k:<12}: {r.summary[k]}")
        print("-" * 70)
