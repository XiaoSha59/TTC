import os
import glob
import yaml

def scan_all_checkpoints(log_dir="logs"):
    found = {}
    print(f"Scanning for checkpoints in {log_dir}...")
    
    # Method 1: Look for .hydra/config.yaml in all subdirectories of log_dir
    config_files = glob.glob(f"{log_dir}/**/.hydra/config.yaml", recursive=True)
    for cfg_file in config_files:
        run_dir = os.path.dirname(os.path.dirname(cfg_file))
        try:
            with open(cfg_file, 'r', encoding='utf-8', errors='ignore') as f:
                cfg = yaml.safe_load(f)
            run_name = cfg.get("name")
            if run_name:
                ckpt_dir = os.path.join(run_dir, "checkpoints")
                if os.path.exists(ckpt_dir):
                    ckpts = glob.glob(f"{ckpt_dir}/*.ckpt")
                    if ckpts:
                        # Prefer last.ckpt
                        best_ckpt = None
                        for c in ckpts:
                            if "last.ckpt" in c:
                                best_ckpt = c
                                break
                        if not best_ckpt:
                            ckpts.sort(key=os.path.getmtime, reverse=True)
                            best_ckpt = ckpts[0]
                        found[run_name.lower()] = best_ckpt
        except Exception:
            pass

    # Method 2: Direct name matching if ckpt filename or parent folder has the name
    all_ckpts = glob.glob(f"{log_dir}/**/*.ckpt", recursive=True)
    for c in all_ckpts:
        c_lower = c.lower()
        for d in ["plants", "insects", "animals"]:
            for r in ["50_50", "95_5", "99_1"]:
                for m in ["supproto", "supmin", "supcon", "weightedce"]:
                    tag = f"{d}-{r}-{m}"
                    if tag in c_lower and tag not in found:
                        found[tag] = c

    print(f"Total discovered mapped runs with checkpoints: {len(found)}")
    for k, v in sorted(found.items()):
        print(f"  {k} -> {v}")
    return found

if __name__ == "__main__":
    scan_all_checkpoints()
