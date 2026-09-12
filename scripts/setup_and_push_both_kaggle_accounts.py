import os
import json
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -------------------------------------------------------------
# 1. Setup Kernel for Account 1: salala1706 (Weighted CE All)
# -------------------------------------------------------------
k1_dir = os.path.join(base_dir, "kaggle_kernels", "ttc-insects-weighted-ce-all")
os.makedirs(k1_dir, exist_ok=True)

meta1 = {
    "id": "salala1706/ttc-insects-weighted-ce-all",
    "title": "TTC Insects Weighted CE All Ratios Full",
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

nb1 = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Stream Extract Full Insects Benchmark (~45 seconds directly from AWS S3)\n",
                "import os, subprocess, time\n",
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
                "!bash scripts/run_kaggle_insects_all_weighted_ce.sh\n",
                "print('🎉 ALL WEIGHTED CE BENCHMARKS FINISHED!')"
            ]
        }
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 2
}

with open(os.path.join(k1_dir, "kernel.ipynb"), "w", encoding="utf-8") as f:
    json.dump(nb1, f, indent=2)

print(">>> [Account 1] Created kernel files in", k1_dir)

# -------------------------------------------------------------
# 2. Setup Kernel for Account 2: ngocuyennhitran (SupCon All)
# -------------------------------------------------------------
k2_dir = os.path.join(base_dir, "kaggle_account_2", "kernel_supcon_all")
os.makedirs(k2_dir, exist_ok=True)

meta2 = {
    "id": "ngocuyennhitran/ttc-insects-supcon-all",
    "title": "TTC Insects SupCon All Ratios Full",
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

nb2 = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Stream Extract Full Insects Benchmark (~45 seconds directly from AWS S3)\n",
                "import os, subprocess, time\n",
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
                "!bash scripts/run_kaggle_insects_all_supcon.sh\n",
                "print('🎉 ALL SUPCON BENCHMARKS FINISHED!')"
            ]
        }
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 2
}

with open(os.path.join(k2_dir, "kernel.ipynb"), "w", encoding="utf-8") as f:
    json.dump(nb2, f, indent=2)

print(">>> [Account 2] Created kernel files in", k2_dir)

def push_kernels():
    # Push Account 1
    os.environ['KAGGLE_USERNAME'] = 'salala1706'
    os.environ['KAGGLE_KEY'] = 'KGAT_c50dd809cbcb96fb725040dee59239f5'
    api1 = KaggleApi()
    api1.authenticate()
    print(">>> Authenticated as:", api1.get_config_value(api1.CONFIG_NAME_USER))
    print(">>> Pushing Kernel 1 to Account 1...")
    res1 = api1.kernels_push(folder=k1_dir)
    print("Result 1:", res1)
    
    # Push Account 2
    if 'KAGGLE_USERNAME' in os.environ:
        del os.environ['KAGGLE_USERNAME']
    if 'KAGGLE_KEY' in os.environ:
        del os.environ['KAGGLE_KEY']
    os.environ['KAGGLE_API_TOKEN'] = 'KGAT_d8bfa22b3323e9d1a32a52737cc6a634'
    api2 = KaggleApi()
    api2.authenticate()
    print(">>> Authenticated as:", api2.get_config_value(api2.CONFIG_NAME_USER))
    print(">>> Pushing Kernel 2 to Account 2...")
    res2 = api2.kernels_push(folder=k2_dir)
    print("Result 2:", res2)

if __name__ == "__main__":
    if len(os.sys.argv) > 1 and os.sys.argv[1] == "--push":
        push_kernels()
    else:
        print("Run with --push to push to Kaggle accounts.")
