import glob, yaml, os

runs = sorted(glob.glob('/home/tnpdung_79/TTC/logs/train/runs/*'))
for r in runs:
    cfg_file = os.path.join(r, '.hydra', 'config.yaml')
    if not os.path.exists(cfg_file):
        continue
    with open(cfg_file) as f:
        cfg = yaml.safe_load(f)
    name = str(cfg.get('name', '')).lower()
    exp = str(cfg.get('experiment', '')).lower()
    dataset = str(cfg.get('data', {}).get('data_module', {}).get('data_set', '')).lower() if isinstance(cfg.get('data'), dict) else ''
    
    is_med = any(k in name or k in dataset for k in ['breast', 'pneumonia', 'frac'])
    if is_med:
        ckpts = glob.glob(os.path.join(r, 'checkpoints', '*.ckpt'))
        last_ckpt = [c for c in ckpts if 'last.ckpt' in c]
        epoch_ckpts = sorted([c for c in ckpts if 'last.ckpt' not in c])
        best_ckpt = last_ckpt[0] if last_ckpt else (epoch_ckpts[-1] if epoch_ckpts else None)
        print(f'{name:32s} | exp: {exp:20s} | dataset: {dataset:12s} | run: {os.path.basename(r)} | ckpt: {best_ckpt}')
