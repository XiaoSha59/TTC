import os
import sys
import json
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def create_kernel(folder_path, kernel_id, title, script_name):
    os.makedirs(folder_path, exist_ok=True)
    
    meta = {
        "id": kernel_id,
        "title": title,
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
    with open(os.path.join(folder_path, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 1. Stream Extract Full Insects Benchmark (~45s)\n",
                    "import os, subprocess\n",
                    "data_path = '/kaggle/temp/inat21_full'\n",
                    "if not os.path.exists(f'{data_path}/train'):\n",
                    "    print('>>> [1/2] Streaming full Insects training data (Apidae + Vespidae) from AWS S3...')\n",
                    "    os.makedirs(data_path, exist_ok=True)\n",
                    "    !curl -sL 'https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.tar.gz' | tar -xz -C /kaggle/temp/inat21_full --wildcards 'train/*_Animalia_Arthropoda_Insecta_Hymenoptera_Apidae*' 'train/*_Animalia_Arthropoda_Insecta_Hymenoptera_Vespidae*'\n",
                    "    print('>>> [2/2] Streaming full Insects validation data (Apidae + Vespidae) from AWS S3...')\n",
                    "    !curl -sL 'https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz' | tar -xz -C /kaggle/temp/inat21_full --wildcards 'val/*_Animalia_Arthropoda_Insecta_Hymenoptera_Apidae*' 'val/*_Animalia_Arthropoda_Insecta_Hymenoptera_Vespidae*'\n",
                    "    print('>>> Full dataset extraction complete!')\n",
                    "else:\n",
                    "    print('>>> Full dataset already present at', data_path)\n",
                    "\n",
                    "# 2. Clone & Update Repo\n",
                    "!git clone https://github.com/XiaoSha59/TTC.git || (cd TTC && git pull)\n",
                    "%cd TTC\n",
                    "!pip install -q 'lightning>=2.0.0' 'hydra-core>=1.3.2' omegaconf pyrootutils timm\n",
                    "import os\n",
                    "os.environ['WANDB_API_KEY'] = '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'\n",
                    "os.environ['INAT21_DATA_PATH'] = '/kaggle/temp/inat21_full'\n",
                    f"!bash scripts/{script_name}\n",
                    f"print('🎉 COMPLETED {title}!')"
                ]
            }
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 2
    }
    with open(os.path.join(folder_path, "kernel.ipynb"), "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f">>> Configured: {kernel_id} -> {script_name}")

# Define 4 Parallel GPU Kernels
kernels = [
    # Account 1
    {
        "account": 1,
        "folder": os.path.join(base_dir, "kaggle_kernels", "ttc-insects-weighted-ce-all"),
        "id": "salala1706/ttc-insects-weighted-ce-all",
        "title": "TTC Insects Weighted CE All Ratios Full",
        "script": "run_kaggle_insects_all_weighted_ce.sh"
    },
    {
        "account": 1,
        "folder": os.path.join(base_dir, "kaggle_kernels", "ttc-insects-supcon-50-50"),
        "id": "salala1706/ttc-insects-supcon-50-50",
        "title": "TTC Insects SupCon 50 50 Full",
        "script": "run_kaggle_insects_supcon_50_50.sh"
    },
    # Account 2
    {
        "account": 2,
        "folder": os.path.join(base_dir, "kaggle_account_2", "kernel_supcon_95_5"),
        "id": "ngocuyennhitran/ttc-insects-supcon-95-5",
        "title": "TTC Insects SupCon 95 5 Full",
        "script": "run_kaggle_insects_supcon_95_5.sh"
    },
    {
        "account": 2,
        "folder": os.path.join(base_dir, "kaggle_account_2", "kernel_supcon_99_1"),
        "id": "ngocuyennhitran/ttc-insects-supcon-99-1",
        "title": "TTC Insects SupCon 99 1 Full",
        "script": "run_kaggle_insects_supcon_99_1.sh"
    }
]

for k in kernels:
    create_kernel(k["folder"], k["id"], k["title"], k["script"])

def deploy_all():
    print("\n========================================================")
    print("🚀 DEPLOYING 4 PARALLEL GPU KERNELS TO KAGGLE")
    print("========================================================")
    
    # Account 1
    os.environ['KAGGLE_USERNAME'] = 'salala1706'
    os.environ['KAGGLE_KEY'] = 'KGAT_c50dd809cbcb96fb725040dee59239f5'
    if 'KAGGLE_API_TOKEN' in os.environ:
        del os.environ['KAGGLE_API_TOKEN']
    api1 = KaggleApi()
    api1.authenticate()
    print("\n>>> [Kaggle Account 1: salala1706]")
    for k in kernels[:2]:
        print(f">>> Pushing {k['id']}...")
        try:
            res = api1.kernels_push(folder=k["folder"])
            print("  Result:", res)
        except Exception as e:
            print("  Warning:", e)
            
    # Account 2
    if 'KAGGLE_USERNAME' in os.environ:
        del os.environ['KAGGLE_USERNAME']
    if 'KAGGLE_KEY' in os.environ:
        del os.environ['KAGGLE_KEY']
    os.environ['KAGGLE_API_TOKEN'] = 'KGAT_d8bfa22b3323e9d1a32a52737cc6a634'
    api2 = KaggleApi()
    api2.authenticate()
    print("\n>>> [Kaggle Account 2: ngocuyennhitran]")
    for k in kernels[2:]:
        print(f">>> Pushing {k['id']}...")
        try:
            res = api2.kernels_push(folder=k["folder"])
            print("  Result:", res)
        except Exception as e:
            print("  Warning:", e)

if __name__ == "__main__":
    deploy_all()
