import os
import json
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi

# Ensure env var for Kaggle 2 token
os.environ["KAGGLE_API_TOKEN"] = "KGAT_d8bfa22b3323e9d1a32a52737cc6a634"

api = KaggleApi()
api.authenticate()
print(f"[OK] Authenticated successfully as Kaggle 2 User: {api.get_config_value(api.CONFIG_NAME_USER)}")

# -------------------------------------------------------------
# Kernel 1: 50:50 SupMinority
# -------------------------------------------------------------
k1_dir = r"d:\TTC\kaggle_account_2\kernel_supmin_50_50"
os.makedirs(k1_dir, exist_ok=True)

shutil.copyfile(r"d:\TTC\kaggle_account_2\kaggle2_gpu1_supmin_50_50.ipynb", os.path.join(k1_dir, "kernel.ipynb"))

meta1 = {
    "id": "ngocuyennhitran/ttc-insects-supmin-50-50",
    "title": "ttc insects supmin 50 50",
    "code_file": "kernel.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_tpu": "false",
    "enable_internet": "true",
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": []
}

with open(os.path.join(k1_dir, "kernel-metadata.json"), "w", encoding="utf-8") as f:
    json.dump(meta1, f, indent=2)

print(">>> Pushing Kernel 1 (SupMinority 50:50) to Kaggle Account 2...")
res1 = api.kernels_push(folder=k1_dir)
print(f"Result Kernel 1: {res1}")

# -------------------------------------------------------------
# Kernel 2: 50:50 & 99:1 Standard SupCon
# -------------------------------------------------------------
k2_dir = r"d:\TTC\kaggle_account_2\kernel_supcon_50_50_and_99_1"
os.makedirs(k2_dir, exist_ok=True)

shutil.copyfile(r"d:\TTC\kaggle_account_2\kaggle2_gpu2_supcon_50_50_and_99_1.ipynb", os.path.join(k2_dir, "kernel.ipynb"))

meta2 = {
    "id": "ngocuyennhitran/ttc-insects-supcon-50-50-and-99-1",
    "title": "ttc insects supcon 50 50 and 99 1",
    "code_file": "kernel.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_tpu": "false",
    "enable_internet": "true",
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": []
}

with open(os.path.join(k2_dir, "kernel-metadata.json"), "w", encoding="utf-8") as f:
    json.dump(meta2, f, indent=2)

print(">>> Pushing Kernel 2 (SupCon 50:50 & 99:1) to Kaggle Account 2...")
res2 = api.kernels_push(folder=k2_dir)
print(f"Result Kernel 2: {res2}")
