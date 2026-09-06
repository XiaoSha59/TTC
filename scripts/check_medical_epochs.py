import os
import wandb

os.environ['WANDB_API_KEY'] = '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'
api = wandb.Api()
targets = ['breastmnist-weightedce', 'pneumoniamnist-weightedce', 'fracatlas-weightedce']

for name in targets:
    runs = api.runs('tnpdung79hcmus/binary-learning', filters={'display_name': name})
    for r in runs:
        if r.state == 'finished':
            cfg = r.config
            max_ep = cfg.get('trainer', {}).get('max_epochs')
            actual_ep = r.summary._json_dict.get('epoch')
            dur = r.summary._json_dict.get('_runtime', 0)
            print(f"{r.name:<28} | max_epochs: {max_ep} | completed epoch: {actual_ep} | duration: {dur:.1f}s")
