#!/usr/bin/env python3
"""
Export results from WandB project and generate Markdown & LaTeX tables
matching Table 1 and Table 2 in the paper.
"""

import os
import sys
import wandb
import pandas as pd
from collections import defaultdict


def fetch_all_runs(entity=None, project="binary-learning"):
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    print(f"Connecting to WandB project: {path} ...")
    try:
        runs = api.runs(path)
    except Exception as e:
        print(f"Error connecting with project name '{path}': {e}")
        print("Trying project 'TTC'...")
        path = f"{entity}/TTC" if entity else "TTC"
        runs = api.runs(path)

    data = []
    for run in runs:
        summary = run.summary._json_dict
        config = run.config
        
        # Extract metrics
        row = {
            "name": run.name,
            "id": run.id,
            "state": run.state,
            "dataset": config.get("data", {}).get("data_module", {}).get("data_set", "unknown") if isinstance(config.get("data"), dict) else "unknown",
            "auroc": summary.get("test.auroc") or summary.get("test/auroc") or summary.get("val.auroc"),
            "balanced_accuracy": summary.get("test.balanced_accuracy") or summary.get("test/balanced_accuracy") or summary.get("val.balanced_accuracy"),
            "accuracy": summary.get("test.accuracy") or summary.get("test/accuracy") or summary.get("val.accuracy"),
            "saa": summary.get("test.saa") or summary.get("test/saa") or summary.get("val.saa"),
            "cac": summary.get("test.cac") or summary.get("test/cac") or summary.get("val.cac"),
            "loss": summary.get("train.loss") or summary.get("train/loss"),
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    return df


def generate_tables(df):
    print("\n" + "=" * 80)
    print(" SUMMARY OF ALL EXPERIMENT RUNS")
    print("=" * 80)
    print(df.to_string(index=False))

    # Save CSV
    os.makedirs("results", exist_ok=True)
    csv_path = "results/wandb_metrics_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved raw metrics to {csv_path}")

    # Generate Markdown Table
    md_path = "results/table2_medical_results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Table 2: Medical Benchmarks Results\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    print(f"Saved Markdown table to {md_path}")


if __name__ == "__main__":
    entity = sys.argv[1] if len(sys.argv) > 1 else None
    project = sys.argv[2] if len(sys.argv) > 2 else "binary-learning"
    df = fetch_all_runs(entity, project)
    generate_tables(df)
