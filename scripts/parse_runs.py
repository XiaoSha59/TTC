import json
import pandas as pd

with open('all_evaluation_runs.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

runs = data['runs']
kaggle_status = data['kaggle']

# Reference numbers from paper
paper_table2 = {
    "Weighted CE": {"breast": "75.1%", "pneumonia": "98.8%", "fracatlas": "79.8%"},
    "Standard SupCon": {"breast": "75.1%", "pneumonia": "99.5%", "fracatlas": "84.8%"},
    "Sup Minority (Ours)": {"breast": "86.4%", "pneumonia": "99.6%", "fracatlas": "82.3%"},
    "Sup Prototypes (Ours)": {"breast": "90.7%", "pneumonia": "99.8%", "fracatlas": "86.0%"}
}

paper_table1 = {
    "plants": {
        "Weighted CE": {"50_50": "81.1%", "95_5": "61.4%", "99_1": "60.1%"},
        "Standard SupCon": {"50_50": "93.7%", "95_5": "56.2%", "99_1": "54.4%"},
        "Sup Minority (Ours)": {"50_50": "—", "95_5": "89.8%", "99_1": "85.4%"},
        "Sup Prototypes (Ours)": {"50_50": "95.1%", "95_5": "88.7%", "99_1": "83.4%"}
    },
    "insects": {
        "Weighted CE": {"50_50": "82.4%", "95_5": "63.4%", "99_1": "62.8%"},
        "Standard SupCon": {"50_50": "93.3%", "95_5": "62.6%", "99_1": "56.4%"},
        "Sup Minority (Ours)": {"50_50": "—", "95_5": "82.8%", "99_1": "78.8%"},
        "Sup Prototypes (Ours)": {"50_50": "93.0%", "95_5": "81.2%", "99_1": "73.7%"}
    },
    "animals": {
        "Weighted CE": {"50_50": "70.7%", "95_5": "61.9%", "99_1": "57.3%"},
        "Standard SupCon": {"50_50": "80.8%", "95_5": "54.4%", "99_1": "56.9%"},
        "Sup Minority (Ours)": {"50_50": "—", "95_5": "77.9%", "99_1": "75.3%"},
        "Sup Prototypes (Ours)": {"50_50": "82.9%", "95_5": "79.2%", "99_1": "73.0%"}
    }
}

print("=== ALL RUNS IN DATABASE ===")
for r in runs:
    name = r['name']
    state = r['state']
    sm = r['summary']
    acc = sm.get('online_val_acc') or sm.get('test.acc') or sm.get('val.acc') or sm.get('test/balanced_accuracy') or sm.get('test_balanced_acc') or sm.get('val/balanced_accuracy')
    auc = sm.get('test.auc') or sm.get('val.auroc') or sm.get('test/auc')
    ep = sm.get('epoch')
    print(f"{name:<45} | {state:<10} | ep={str(ep):<5} | acc={str(acc):<18} | auc={str(auc):<18}")
