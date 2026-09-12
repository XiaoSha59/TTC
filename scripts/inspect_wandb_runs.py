import os
import wandb

os.environ['WANDB_API_KEY'] = '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'
api = wandb.Api()
runs = api.runs('tnpdung79hcmus/binary-learning', order='-created_at')

targets = [
    'insects-50_50-supproto',
    'insects-99_1-supproto',
    'insects-50_50-supmin',
    'insects-50_50-supcon',
    'insects-95_5-supmin-350ep'
]

seen = set()
for r in runs:
    for t in targets:
        if t in r.name.lower() and r.name not in seen:
            seen.add(r.name)
            cfg = r.config
            summary = r.summary
            runtime_hrs = summary.get('_runtime', 0) / 3600.0
            epoch = summary.get('epoch', 'N/A')
            batch_size = cfg.get('batch_size', 'N/A')
            max_epochs = cfg.get('trainer', {}).get('max_epochs', cfg.get('trainer.max_epochs', 'N/A')) if isinstance(cfg.get('trainer'), dict) else cfg.get('trainer.max_epochs', 'N/A')
            precision = cfg.get('trainer', {}).get('precision', cfg.get('trainer.precision', 'N/A')) if isinstance(cfg.get('trainer'), dict) else cfg.get('trainer.precision', 'N/A')
            lr = cfg.get('module', {}).get('lr', cfg.get('module.lr', 'N/A')) if isinstance(cfg.get('module'), dict) else cfg.get('module.lr', 'N/A')
            loss = summary.get('train.loss.epoch', summary.get('train.loss', summary.get('val.loss', 'N/A')))
            
            print(f"Run Name    : {r.name}")
            print(f"State       : {r.state}")
            print(f"Created At  : {r.created_at}")
            print(f"Runtime     : {runtime_hrs:.2f} hours")
            print(f"Epoch Reach : {epoch} / {max_epochs}")
            print(f"Batch Size  : {batch_size}")
            print(f"Precision   : {precision}")
            print(f"LearningRate: {lr}")
            print(f"Final Loss  : {loss}")
            print("-" * 70)
