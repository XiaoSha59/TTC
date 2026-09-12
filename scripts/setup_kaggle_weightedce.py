import os
import json

kernel_dir = r"d:\TTC\kaggle_kernels\ttc-insects-weightedce"
os.makedirs(kernel_dir, exist_ok=True)

metadata = {
    "id": "salala1706/ttc-insects-weighted-ce-95-5",
    "title": "ttc insects weighted ce 95 5",
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
                "# 5. Thiết lập Dataset iNat21 Natural (train_mini từ input, val từ S3)\n",
                "import os, glob, subprocess, shutil\n",
                "os.makedirs('data/inat21', exist_ok=True)\n",
                "\n",
                "# Link train_mini & train\n",
                "train_src = '/kaggle/input/inat21-natural/train_mini'\n",
                "if not os.path.exists(train_src):\n",
                "    train_cands = glob.glob('/kaggle/input/**/train_mini', recursive=True)\n",
                "    if train_cands:\n",
                "        train_src = train_cands[0]\n",
                "print(f'>>> Train Source: {train_src}')\n",
                "\n",
                "if not os.path.exists('data/inat21/train_mini'):\n",
                "    os.symlink(train_src, 'data/inat21/train_mini')\n",
                "if not os.path.exists('data/inat21/train'):\n",
                "    os.symlink(train_src, 'data/inat21/train')\n",
                "\n",
                "# Download and extract val split into /tmp/val\n",
                "if not os.path.exists('/tmp/val'):\n",
                "    print('>>> Đang tải nhanh tập val (8.3GB) từ AWS S3...')\n",
                "    !wget -q --show-progress -O /tmp/val.tar.gz https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz\n",
                "    print('>>> Đang giải nén tập val vào /tmp...')\n",
                "    !tar -xzf /tmp/val.tar.gz -C /tmp/\n",
                "    !rm -f /tmp/val.tar.gz\n",
                "\n",
                "if not os.path.exists('data/inat21/val'):\n",
                "    os.symlink('/tmp/val', 'data/inat21/val')\n",
                "\n",
                "# Verify dataset\n",
                "from data.iNatData import INaturalistNClasses\n",
                "t_ds = INaturalistNClasses('data/inat21', split='train', classes=['Animalia_Arthropoda_Insecta_Hymenoptera_Apidae', 'Animalia_Arthropoda_Insecta_Hymenoptera_Vespidae'])\n",
                "v_ds = INaturalistNClasses('data/inat21', split='val', classes=['Animalia_Arthropoda_Insecta_Hymenoptera_Apidae', 'Animalia_Arthropoda_Insecta_Hymenoptera_Vespidae'])\n",
                "print(f'✅ Dataset iNat21 sẵn sàng! Train: {len(t_ds)} ảnh, Val: {len(v_ds)} ảnh')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 6. Khởi chạy chuỗi Insects Weighted CE (95:5 -> 99:1 -> 50:50)\n",
                "!python train.py \\\n",
                "    experiment=weighted_ce \\\n",
                "    experiment/specs=insects \\\n",
                "    class_ratios=[0.05,0.95] \\\n",
                "    batch_size=256 \\\n",
                "    trainer.max_epochs=350 \\\n",
                "    trainer.precision=16-mixed \\\n",
                "    data.data_module.num_workers=2 \\\n",
                "    data.data_module.persistent_workers=False \\\n",
                "    name='insects-95_5-weightedce-full'\n",
                "\n",
                "!python train.py \\\n",
                "    experiment=weighted_ce \\\n",
                "    experiment/specs=insects \\\n",
                "    class_ratios=[0.01,0.99] \\\n",
                "    batch_size=256 \\\n",
                "    trainer.max_epochs=350 \\\n",
                "    trainer.precision=16-mixed \\\n",
                "    data.data_module.num_workers=2 \\\n",
                "    data.data_module.persistent_workers=False \\\n",
                "    name='insects-99_1-weightedce-full'\n",
                "\n",
                "!python train.py \\\n",
                "    experiment=weighted_ce \\\n",
                "    experiment/specs=insects \\\n",
                "    class_ratios=[0.5,0.5] \\\n",
                "    batch_size=256 \\\n",
                "    trainer.max_epochs=350 \\\n",
                "    trainer.precision=16-mixed \\\n",
                "    data.data_module.num_workers=2 \\\n",
                "    data.data_module.persistent_workers=False \\\n",
                "    name='insects-50_50-weightedce-full'"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('🎉 Hoàn thành xuất sắc toàn bộ chuỗi Insects Weighted CE! Kaggle GPU tự động giải phóng.')"
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
