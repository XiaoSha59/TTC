import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🚀 TTC: Chạy Song Song Trên Kaggle GPU (NVIDIA P100 / Dual T4)\n",
    "Notebook này chạy song song cho phần còn lại của tập **INSECTS**:\n",
    "1. **Weighted CE (3 tỷ lệ: 50:50, 95:5, 99:1)**: 100 epochs với class_weights chuẩn.\n",
    "2. **SupMinority (2 tỷ lệ: 95:5, 99:1)**: 2 Lớp (Pretrain 350 epochs + Linear Probing 50 epochs).\n",
    "Toàn bộ kết quả tự động đổ về chung Dashboard WandB: `tnpdung79hcmus/binary-learning`."
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
    "# 2. Clone mã nguồn mới nhất từ GitHub\n",
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
    "# 3. Cài đặt các gói phụ thuộc (mất ~20 giây)\n",
    "!pip install -q lightning>=2.0.0 hydra-core>=1.3.2 omegaconf pyrootutils timm medmnist"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 4. Đăng nhập WandB (đồng bộ cùng dashboard)\n",
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
    "# 5. Kiểm tra và liên kết dữ liệu iNat21 từ Kaggle Input\n",
    "import os, glob\n",
    "os.makedirs('data/inat21', exist_ok=True)\n",
    "if os.path.exists('/kaggle/input'):\n",
    "    print('Dữ liệu tìm thấy trong Kaggle input:', os.listdir('/kaggle/input'))\n",
    "    for folder in glob.glob('/kaggle/input/**/train_mini', recursive=True):\n",
    "        parent = os.path.dirname(folder)\n",
    "        print('Đang liên kết dữ liệu từ:', parent)\n",
    "        !cp -rs {parent}/* data/inat21/ 2>/dev/null || !ln -s {parent}/* data/inat21/ 2>/dev/null || true\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 6. Khởi chạy Pipeline Song Song: Weighted CE (3 tỷ lệ) + SupMinority 2 Lớp (95:5 & 99:1)\n",
    "# Chạy ngầm toàn bộ quy trình chuẩn hóa: FP16 (16-mixed), Batch 256 physical\n",
    "!bash scripts/run_kaggle_insects_supmin_weightedce.sh"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

os.makedirs('notebooks', exist_ok=True)
with open('notebooks/kaggle_ttc_runner.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)
print('Done!')
