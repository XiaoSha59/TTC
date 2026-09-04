#!/usr/bin/env python3
"""
Export and format Table 2 and Table 1 comparison tables from WandB.
"""

import os
import sys
import wandb
import pandas as pd


def fetch_and_format_table2(entity="tnpdung79hcmus", project="binary-learning"):
    if not os.environ.get("WANDB_API_KEY"):
        os.environ["WANDB_API_KEY"] = "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"
    api = wandb.Api()
    path = f"{entity}/{project}"
    print(f"Fetching runs from {path} ...")
    runs = api.runs(path, per_page=100)

    # Collect best metric for each method & dataset
    datasets = ["breast", "pneumonia", "fracatlas"]
    methods = ["weightedce", "supcon", "supmin", "supproto"]

    results = {}
    for d in datasets:
        results[d] = {}
        for m in methods:
            results[d][m] = {"acc": None, "auc": None, "loss": None}

    for r in runs:
        if r.state != "finished":
            continue
        name = r.name.lower()
        s = r.summary._json_dict
        
        acc = s.get("online_val_acc") or s.get("val.acc") or s.get("test.acc")
        auc = s.get("test.auc") or s.get("val.auroc")
        loss = s.get("online_val_loss") or s.get("val.loss")
        
        for d in datasets:
            if d in name:
                for m in methods:
                    if m in name:
                        if acc is not None:
                            results[d][m]["acc"] = acc
                        if auc is not None:
                            results[d][m]["auc"] = auc
                        if loss is not None:
                            results[d][m]["loss"] = loss

    print("\n" + "=" * 90)
    print(" TABLE 2: MEDICAL BENCHMARKS REPRODUCED RESULTS")
    print("=" * 90)
    
    rows = [
        {"Method": "Weighted CE (Baseline)", "BreastMNIST (Paper: 75.1%)": results["breast"]["weightedce"]["acc"], "PneumoniaMNIST (Paper: 98.8%)": results["pneumonia"]["weightedce"]["acc"], "FracAtlas (Paper: 79.8%)": results["fracatlas"]["weightedce"]["acc"]},
        {"Method": "Standard SupCon (Baseline)", "BreastMNIST (Paper: 75.1%)": results["breast"]["supcon"]["acc"], "PneumoniaMNIST (Paper: 99.5%)": results["pneumonia"]["supcon"]["acc"], "FracAtlas (Paper: 84.8%)": results["fracatlas"]["supcon"]["acc"]},
        {"Method": "Sup Minority (Ours)", "BreastMNIST (Paper: 86.4%)": results["breast"]["supmin"]["acc"], "PneumoniaMNIST (Paper: 99.6%)": results["pneumonia"]["supmin"]["acc"], "FracAtlas (Paper: 82.3%)": results["fracatlas"]["supmin"]["acc"]},
        {"Method": "Sup Prototypes (Ours)", "BreastMNIST (Paper: 90.7%)": results["breast"]["supproto"]["acc"], "PneumoniaMNIST (Paper: 99.8%)": results["pneumonia"]["supproto"]["acc"], "FracAtlas (Paper: 86.0%)": results["fracatlas"]["supproto"]["acc"]},
    ]
    
    df_table = pd.DataFrame(rows)
    # Format percentages
    for col in df_table.columns[1:]:
        df_table[col] = df_table[col].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) and isinstance(x, (int, float)) else ("N/A" if pd.isnull(x) else str(x)))

    print(df_table.to_string(index=False))
    
    os.makedirs("results", exist_ok=True)
    df_table.to_markdown("results/table2_medical_results.md", index=False)
    print("\nSaved formatted table to results/table2_medical_results.md")

if __name__ == "__main__":
    fetch_and_format_table2()
