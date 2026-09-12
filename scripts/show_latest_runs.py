import json

with open('all_evaluation_runs.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

runs = data['runs']
print(f"Total runs fetched: {len(runs)}")
print("=" * 120)
print(f"{'Run Name':<45} | {'State':<10} | {'Epoch':<6} | {'Val Acc':<12} | {'AUC':<12} | {'Created At':<20}")
print("-" * 120)
for r in runs[:30]:
    name = r['name']
    state = r['state']
    sm = r['summary']
    acc = sm.get('online_val_acc') or sm.get('test.acc') or sm.get('val.acc') or sm.get('test/balanced_accuracy') or sm.get('test_balanced_acc') or sm.get('val/balanced_accuracy')
    auc = sm.get('test.auc') or sm.get('val.auroc') or sm.get('test/auc')
    ep = sm.get('epoch')
    acc_str = f"{acc*100:.2f}%" if isinstance(acc, (int, float)) and acc <= 1.0 else str(acc)
    auc_str = f"{auc*100:.2f}%" if isinstance(auc, (int, float)) and auc <= 1.0 else str(auc)
    print(f"{name:<45} | {state:<10} | {str(ep):<6} | {acc_str:<12} | {auc_str:<12} | {r.get('created', '')[:19]}")
