import os
import wandb

os.environ['WANDB_API_KEY'] = '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'
api = wandb.Api()

targets = [
    # Plants
    'plants-50_50-supproto', 'plants-50_50-supmin', 'plants-50_50-supcon',
    'plants-95_5-supproto', 'plants-95_5-supmin', 'plants-95_5-supcon',
    'plants-99_1-supproto', 'plants-99_1-supmin', 'plants-99_1-supcon',
    # Insects
    'insects-50_50-supproto', 'insects-50_50-supmin', 'insects-50_50-supcon',
    'insects-95_5-supproto', 'insects-95_5-supmin', 'insects-95_5-supcon',
    'insects-99_1-supproto', 'insects-99_1-supmin', 'insects-99_1-supcon',
    # Animals
    'animals-50_50-supproto', 'animals-50_50-supmin', 'animals-50_50-supcon',
    'animals-95_5-supproto', 'animals-95_5-supmin', 'animals-95_5-supcon',
    'animals-99_1-supproto', 'animals-99_1-supmin', 'animals-99_1-supcon',
    # Medical
    'fracatlas-supproto', 'fracatlas-supmin', 'fracatlas-supcon',
    'breastmnist-supproto', 'breastmnist-supmin', 'breastmnist-supcon',
    'pneumoniamnist-supproto', 'pneumoniamnist-supmin', 'pneumoniamnist-supcon'
]

print(f"{'Run Name':<28} | {'State':<9} | {'Ep':<4} | {'LR':<7} | {'BS':<5} | {'Acc':<8} | {'AUC':<8}")
print("-" * 85)

for name in targets:
    runs = api.runs('tnpdung79hcmus/binary-learning', filters={'display_name': name})
    if runs:
        r = runs[0]
        cfg = r.config
        lr = cfg.get('module', {}).get('lr') if isinstance(cfg.get('module'), dict) else cfg.get('lr')
        bs = cfg.get('batch_size')
        trainer_cfg = cfg.get('trainer', {}) if isinstance(cfg.get('trainer'), dict) else {}
        ep = trainer_cfg.get('max_epochs', '?')
        s = r.summary._json_dict
        acc = s.get('test.acc') or s.get('test/acc') or s.get('test_acc') or s.get('acc_test')
        auc = s.get('test.auc') or s.get('test/auc') or s.get('test_auc') or s.get('auc_test')
        acc_str = f"{acc*100:.1f}%" if acc is not None else "N/A"
        auc_str = f"{auc*100:.1f}%" if auc is not None else "N/A"
        print(f"{name:<28} | {r.state:<9} | {str(ep):<4} | {str(lr):<7} | {str(bs):<5} | {acc_str:<8} | {auc_str:<8}")
    else:
        print(f"{name:<28} | {'NOT FOUND':<9} | {'-':<4} | {'-':<7} | {'-':<5} | {'-':<8} | {'-':<8}")
