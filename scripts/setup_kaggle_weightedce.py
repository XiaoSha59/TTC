import os
import json

kernel_dir = r"d:\TTC\kaggle_kernels\ttc-insects-weightedce"
os.makedirs(kernel_dir, exist_ok=True)

metadata = {
    "id": "salala1706/ttc-insects-weightedce",
    "title": "TTC Insects Weighted CE 95:5",
    "code_file": "kernel.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_tpu": "false",
    "enable_internet": "true",
    "dataset_sources": [
        "salala1706/inat21-natural"
    ],
    "competition_sources": [],
    "kernel_sources": []
}

with open(os.path.join(kernel_dir, "kernel-metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🐝 TTC: Insects 95:5 Weighted Cross-Entropy Baseline (Table 1)\n",
                "Chạy 350 Epochs với FP16-mixed và inverse frequency class weights.\n",
                "Tự động đồng bộ lên WandB và tự ngắt GPU khi hoàn tất."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Kiểm tra GPU\n",
                "!nvidia-smi"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. Clone mã nguồn TTC mới nhất\n",
                "import os\n",
                "if not os.path.exists('TTC'):\n",
                "    !git clone https://github.com/XiaoSha59/TTC.git\n",
                "%cd TTC\n",
                "!git pull"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 3. Cài đặt các thư viện cần thiết\n",
                "!pip install -q 'lightning>=2.0.0' 'hydra-core>=1.3.2' omegaconf pyrootutils timm"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 4. Cấu hình WandB API\n",
                "import os, wandb\n",
                "os.environ['WANDB_API_KEY'] = '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'\n",
                "!wandb login $WANDB_API_KEY"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 5. Liên kết Dataset iNat21 Natural\n",
                "import os, subprocess, shutil\n",
                "os.makedirs('data', exist_ok=True)\n",
                "inat_input = '/kaggle/input/inat21-natural'\n",
                "if not os.path.exists('data/inat21'):\n",
                "    if os.path.exists(inat_input):\n",
                "        os.symlink(inat_input, 'data/inat21')\n",
                "        print('>>> Đã liên kết dataset từ /kaggle/input/inat21-natural thành công!')\n",
                "    else:\n",
                "        print('⚠️ Cảnh báo: Tìm kiếm dataset inat21...')\n",
                "        !find /kaggle/input -maxdepth 3 -type d"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 6. Khởi chạy Insects Weighted CE 95:5 (350 Epochs)\n",
                "!python train.py \\\n",
                "    experiment=weighted_ce \\\n",
                "    experiment/specs=insects \\\n",
                "    class_ratios=[0.05,0.95] \\\n",
                "    batch_size=256 \\\n",
                "    trainer.max_epochs=350 \\\n",
                "    trainer.precision=16-mixed \\\n",
                "    data.data_module.num_workers=2 \\\n",
                "    data.data_module.persistent_workers=False \\\n",
                "    name='insects-95_5-weightedce-full'"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('🎉 Hoàn thành xuất sắc Insects Weighted CE 95:5! Kaggle GPU tự động giải phóng.')"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open(os.path.join(kernel_dir, "kernel.ipynb"), "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print("Kaggle Kernel 1 (Weighted CE) prepared!")
