import glob, os, yaml

def main():
    found = {}
    for path in glob.glob('/home/tnpdung_79/TTC/logs/**/.hydra/config.yaml', recursive=True):
        try:
            with open(path) as f:
                cfg = yaml.safe_load(f)
            name = cfg.get('name')
            if name and name not in found:
                lr = cfg.get('module', {}).get('lr') if isinstance(cfg.get('module'), dict) else None
                opt = cfg.get('module', {}).get('optimizer_name') if isinstance(cfg.get('module'), dict) else None
                bs = cfg.get('batch_size')
                found[name] = (lr, opt, bs)
        except Exception:
            pass

    print(f"=== AUDITED {len(found)} RUN CONFIGS LOCALLY ON VM ===")
    print(f"{'Run Name':<38} | {'LR':<10} | {'Opt':<6} | {'Batch':<6} | {'Status'}")
    print("-" * 75)
    for k in sorted(found.keys()):
        lr, opt, bs = found[k]
        flag = " [ALERT WRONG LR!]" if lr is not None and lr > 0.1 else " [OK]"
        print(f"{k:<38} | {str(lr):<10} | {str(opt):<6} | {str(bs):<6} | {flag}")

if __name__ == '__main__':
    main()
