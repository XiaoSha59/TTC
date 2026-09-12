import os
import glob
import yaml

runs = sorted(glob.glob('/home/tnpdung_79/TTC/logs/train/runs/*/.hydra/config.yaml'))
print(f"{'Run Directory':<22} | {'Max Epochs':<10} | {'Experiment Name'}")
print("-" * 75)

for p in runs:
    try:
        with open(p, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        name = str(cfg.get('name', ''))
        epochs = cfg.get('trainer', {}).get('max_epochs', 'N/A')
        run_id = os.path.basename(os.path.dirname(os.path.dirname(p)))
        if 'insects' in name.lower():
            print(f"{run_id:<22} | {epochs:<10} | {name}")
    except Exception as e:
        pass
