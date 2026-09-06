import os
os.environ["HOME"] = "/tmp"
os.environ["WANDB_DIR"] = "/tmp"
os.environ["WANDB_CONFIG_DIR"] = "/tmp"
os.environ["WANDB_CACHE_DIR"] = "/tmp"
os.environ["WANDB_API_KEY"] = "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"

import wandb

try:
    api = wandb.Api()
    run = api.run("tnpdung79hcmus/binary-learning/iyt8ibrt")
    print(">>> Run Name:", run.name)
    print(">>> State:", run.state)
    print("=== FINAL TEST METRICS ===")
    for k, v in sorted(run.summary.items()):
        if not k.startswith("_"):
            print(f"  {k}: {v}")
except Exception as e:
    print("Error:", e)
