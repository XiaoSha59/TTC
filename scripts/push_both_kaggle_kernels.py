import os
import json

# Setup 1: Weighted CE
kernel_dir_1 = r"d:\TTC\kaggle_kernels\ttc-insects-weightedce"
os.makedirs(kernel_dir_1, exist_ok=True)
with open(os.path.join(kernel_dir_1, "kernel-metadata.json"), "w", encoding="utf-8") as f:
    json.dump({
        "id": "salala1706/ttc-insects-weighted-ce-95-5",
        "title": "TTC Insects Weighted CE 95:5",
        "code_file": "kernel.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": ["salala1706/inat21-natural"],
        "competition_sources": [],
        "kernel_sources": []
    }, f, indent=2)

nb1 = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Clone & Update Repo\n",
                "!git clone https://github.com/XiaoSha59/TTC.git || (cd TTC && git pull)\n",
                "%cd TTC\n",
                "!pip install -q 'lightning>=2.0.0' 'hydra-core>=1.3.2' omegaconf pyrootutils timm\n",
                "import os\n",
                "os.environ['WANDB_API_KEY'] = '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'\n",
                "!bash scripts/run_kaggle_insects_weightedce.sh\n",
                "print('DONE')"
            ]
        }
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 2
}
with open(os.path.join(kernel_dir_1, "kernel.ipynb"), "w", encoding="utf-8") as f:
    json.dump(nb1, f, indent=2)

# Setup 2: Standard SupCon
kernel_dir_2 = r"d:\TTC\kaggle_kernels\ttc-insects-supcon"
os.makedirs(kernel_dir_2, exist_ok=True)
with open(os.path.join(kernel_dir_2, "kernel-metadata.json"), "w", encoding="utf-8") as f:
    json.dump({
        "id": "salala1706/ttc-insects-supcon-95-5",
        "title": "TTC Insects SupCon 95:5",
        "code_file": "kernel.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": ["salala1706/inat21-natural"],
        "competition_sources": [],
        "kernel_sources": []
    }, f, indent=2)

nb2 = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Clone & Update Repo\n",
                "!git clone https://github.com/XiaoSha59/TTC.git || (cd TTC && git pull)\n",
                "%cd TTC\n",
                "!pip install -q 'lightning>=2.0.0' 'hydra-core>=1.3.2' omegaconf pyrootutils timm\n",
                "import os\n",
                "os.environ['WANDB_API_KEY'] = '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'\n",
                "!bash scripts/run_kaggle_insects_supcon.sh\n",
                "print('DONE')"
            ]
        }
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 2
}
with open(os.path.join(kernel_dir_2, "kernel.ipynb"), "w", encoding="utf-8") as f:
    json.dump(nb2, f, indent=2)

print("Both simplified notebooks generated!")
