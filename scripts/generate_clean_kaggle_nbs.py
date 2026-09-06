import os
import json

def make_nb(run_script_cmd):
    return {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# 1. GPU Check\n", "!nvidia-smi"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 2. Clone / Pull Repo\n",
                    "import os\n",
                    "if not os.path.exists('TTC'):\n",
                    "    !git clone https://github.com/XiaoSha59/TTC.git\n",
                    "%cd TTC\n",
                    "!git pull\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 3. Install packages\n",
                    "!pip install -q 'lightning>=2.0.0' 'hydra-core>=1.3.2' omegaconf pyrootutils timm\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 4. WandB Login\n",
                    "import os, wandb\n",
                    "os.environ['WANDB_API_KEY'] = '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'\n",
                    "!wandb login $WANDB_API_KEY\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 5. Run Training Script\n",
                    run_script_cmd + "\n"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

# 1. Weighted CE
nb1 = make_nb("!bash scripts/run_kaggle_insects_weightedce.sh")
with open(r"d:\TTC\kaggle_kernels\ttc-insects-weightedce\kernel.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb1, f, indent=2)

# 2. SupCon
nb2 = make_nb("!bash scripts/run_kaggle_insects_supcon.sh")
with open(r"d:\TTC\kaggle_kernels\ttc-insects-supcon\kernel.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb2, f, indent=2)

print("Generated clean multi-cell notebooks!")
