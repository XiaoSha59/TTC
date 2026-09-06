import os, wandb

os.environ['WANDB_API_KEY'] = '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'

def check_weighted():
    api = wandb.Api()
    runs = api.runs('tnpdung79hcmus/binary-learning', per_page=200)
    w_runs = [r for r in runs if 'weighted' in r.name.lower()]
    print(f"Total weighted CE runs found: {len(w_runs)}")
    print(f"{'Run Name':<35} | {'Top BS':<8} | {'DM BS':<8} | {'Test Acc':<10} | {'Test AUC':<10} | {'State':<10}")
    print("-" * 95)
    for r in sorted(w_runs, key=lambda x: x.name):
        data_cfg = r.config.get('data', {})
        dm = data_cfg.get('data_module', {}) if isinstance(data_cfg, dict) else {}
        dm_bs = dm.get('batch_size') if isinstance(dm, dict) else None
        top_bs = r.config.get('batch_size')
        s = r.summary._json_dict
        acc = s.get('test.acc') or s.get('test/acc') or s.get('test_acc')
        auc = s.get('test.auc') or s.get('test/auc') or s.get('test_auc')
        acc_str = f"{acc*100:.2f}%" if acc is not None else "N/A"
        auc_str = f"{auc*100:.2f}%" if auc is not None else "N/A"
        print(f"{r.name:<35} | {str(top_bs):<8} | {str(dm_bs):<8} | {acc_str:<10} | {auc_str:<10} | {r.state:<10}")

if __name__ == '__main__':
    check_weighted()
