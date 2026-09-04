#!/usr/bin/env python3
"""
Export and format Table 2 (Medical) and Table 1 (Natural) comparison tables from WandB.
"""

import os
import sys
import wandb
import pandas as pd


def fetch_all_results(entity="tnpdung79hcmus", project="binary-learning"):
    if not os.environ.get("WANDB_API_KEY"):
        os.environ["WANDB_API_KEY"] = "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"
    api = wandb.Api()
    path = f"{entity}/{project}"
    print(f"Fetching runs from {path} ...")
    runs = api.runs(path, per_page=100)

    # Table 1: Natural Benchmarks (Plants, Insects, Animals) across 50:50, 95:5, 99:1
    t1_datasets = ["plants", "insects", "animals"]
    ratios = ["50_50", "95_5", "99_1"]
    methods = ["weightedce", "supcon", "supmin", "supproto"]

    table1_data = {d: {r: {m: None for m in methods} for r in ratios} for d in t1_datasets}

    # Table 2: Medical Benchmarks
    t2_datasets = ["breast", "pneumonia", "fracatlas"]
    table2_data = {d: {m: {"acc": None, "auc": None, "loss": None} for m in methods} for d in t2_datasets}

    for r in runs:
        if r.state != "finished":
            continue
        name = r.name.lower()
        s = r.summary._json_dict
        acc = s.get("online_val_acc") or s.get("test.acc") or s.get("val.acc")
        auc = s.get("test.auc") or s.get("val.auroc")
        loss = s.get("online_val_loss") or s.get("val.loss")

        # Check Table 1
        for d in t1_datasets:
            if d in name:
                for ratio in ratios:
                    if ratio in name:
                        for m in methods:
                            if m in name and table1_data[d][ratio][m] is None:
                                table1_data[d][ratio][m] = acc

        # Check Table 2
        for d in t2_datasets:
            if d in name:
                for m in methods:
                    if m in name and table2_data[d][m]["acc"] is None:
                        table2_data[d][m]["acc"] = acc
                        table2_data[d][m]["auc"] = auc
                        table2_data[d][m]["loss"] = loss

    os.makedirs("results", exist_ok=True)

    # --- Print Table 1: Plants ---
    print("\n" + "=" * 90)
    print(" TABLE 1: NATURAL BENCHMARKS (PLANTS SUBSET)")
    print("=" * 90)
    t1_rows = [
        {"Method": "Weighted CE (Baseline)", "50%:50% (Balanced)": table1_data["plants"]["50_50"]["weightedce"], "95%:5% (Imbalanced)": table1_data["plants"]["95_5"]["weightedce"], "99%:1% (Extreme)": table1_data["plants"]["99_1"]["weightedce"]},
        {"Method": "Standard SupCon (Baseline)", "50%:50% (Balanced)": table1_data["plants"]["50_50"]["supcon"], "95%:5% (Imbalanced)": table1_data["plants"]["95_5"]["supcon"], "99%:1% (Extreme)": table1_data["plants"]["99_1"]["supcon"]},
        {"Method": "Sup Minority (Ours)", "50%:50% (Balanced)": table1_data["plants"]["50_50"]["supmin"], "95%:5% (Imbalanced)": table1_data["plants"]["95_5"]["supmin"], "99%:1% (Extreme)": table1_data["plants"]["99_1"]["supmin"]},
        {"Method": "Sup Prototypes (Ours)", "50%:50% (Balanced)": table1_data["plants"]["50_50"]["supproto"], "95%:5% (Imbalanced)": table1_data["plants"]["95_5"]["supproto"], "99%:1% (Extreme)": table1_data["plants"]["99_1"]["supproto"]},
    ]
    df_t1 = pd.DataFrame(t1_rows)
    for col in df_t1.columns[1:]:
        df_t1[col] = df_t1[col].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) and isinstance(x, (int, float)) else ("N/A" if pd.isnull(x) else str(x)))
    print(df_t1.to_string(index=False))
    df_t1.to_markdown("results/table1_plants_results.md", index=False)

    # --- Print Table 2: Medical ---
    print("\n" + "=" * 90)
    print(" TABLE 2: MEDICAL BENCHMARKS REPRODUCED RESULTS")
    print("=" * 90)
    t2_rows = [
        {"Method": "Weighted CE (Baseline)", "BreastMNIST (Paper: 75.1%)": table2_data["breast"]["weightedce"]["acc"], "PneumoniaMNIST (Paper: 98.8%)": table2_data["pneumonia"]["weightedce"]["acc"], "FracAtlas (Paper: 79.8%)": table2_data["fracatlas"]["weightedce"]["acc"]},
        {"Method": "Standard SupCon (Baseline)", "BreastMNIST (Paper: 75.1%)": table2_data["breast"]["supcon"]["acc"], "PneumoniaMNIST (Paper: 99.5%)": table2_data["pneumonia"]["supcon"]["acc"], "FracAtlas (Paper: 84.8%)": table2_data["fracatlas"]["supcon"]["acc"]},
        {"Method": "Sup Minority (Ours)", "BreastMNIST (Paper: 86.4%)": table2_data["breast"]["supmin"]["acc"], "PneumoniaMNIST (Paper: 99.6%)": table2_data["pneumonia"]["supmin"]["acc"], "FracAtlas (Paper: 82.3%)": table2_data["fracatlas"]["supmin"]["acc"]},
        {"Method": "Sup Prototypes (Ours)", "BreastMNIST (Paper: 90.7%)": table2_data["breast"]["supproto"]["acc"], "PneumoniaMNIST (Paper: 99.8%)": table2_data["pneumonia"]["supproto"]["acc"], "FracAtlas (Paper: 86.0%)": table2_data["fracatlas"]["supproto"]["acc"]},
    ]
    df_t2 = pd.DataFrame(t2_rows)
    for col in df_t2.columns[1:]:
        df_t2[col] = df_t2[col].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) and isinstance(x, (int, float)) else ("N/A" if pd.isnull(x) else str(x)))
    print(df_t2.to_string(index=False))
    df_t2.to_markdown("results/table2_medical_results.md", index=False)


if __name__ == "__main__":
    fetch_all_results()

