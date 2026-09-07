#!/bin/bash
# Script chạy 12 Official Linear Probes (50 epochs Adam lr=0.001) cho Table 2 Medical

cd /home/tnpdung_79/TTC
source .venv/bin/activate
export PYTHONPATH=.
export WANDB_API_KEY="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"

echo "================================================================================"
echo "🏥 BẮT ĐẦU CHẠY 12 OFFICIAL LIGHTNING PROBES CHO TABLE 2 MEDICAL (AUROC %)"
echo "================================================================================"

# 1. BreastMNIST (4 methods)
echo ">>> 1/12 BreastMNIST Weighted CE..."
python train.py experiment=finetune data=med_mnist data.data_module.data_set=breast batch_size=128 +base_model_path=/home/tnpdung_79/TTC/logs/train/runs/2026-09-03_18-13-07/checkpoints/last.ckpt trainer.max_epochs=50 module.optimizer_name=adam module.lr=0.001 name=breast-weightedce-probe

echo ">>> 2/12 BreastMNIST Standard SupCon..."
python train.py experiment=finetune data=med_mnist data.data_module.data_set=breast batch_size=128 +base_model_path=/home/tnpdung_79/TTC/logs/train/runs/2026-09-03_17-53-33/checkpoints/last.ckpt trainer.max_epochs=50 module.optimizer_name=adam module.lr=0.001 name=breast-supcon-probe

echo ">>> 3/12 BreastMNIST Sup Minority..."
python train.py experiment=finetune data=med_mnist data.data_module.data_set=breast batch_size=128 +base_model_path=/home/tnpdung_79/TTC/logs/train/runs/2026-09-03_17-27-32/checkpoints/last.ckpt trainer.max_epochs=50 module.optimizer_name=adam module.lr=0.001 name=breast-supmin-probe

echo ">>> 4/12 BreastMNIST Sup Prototypes..."
python train.py experiment=finetune data=med_mnist data.data_module.data_set=breast batch_size=128 +base_model_path=/home/tnpdung_79/TTC/logs/train/runs/2026-09-03_17-40-27/checkpoints/last.ckpt trainer.max_epochs=50 module.optimizer_name=adam module.lr=0.001 name=breast-supproto-probe

# 2. PneumoniaMNIST (4 methods)
echo ">>> 5/12 PneumoniaMNIST Weighted CE..."
python train.py experiment=finetune data=med_mnist data.data_module.data_set=pneumonia batch_size=128 +base_model_path=/home/tnpdung_79/TTC/logs/train/runs/2026-09-03_22-25-41/checkpoints/last.ckpt trainer.max_epochs=50 module.optimizer_name=adam module.lr=0.001 name=pneumonia-weightedce-probe

echo ">>> 6/12 PneumoniaMNIST Standard SupCon..."
python train.py experiment=finetune data=med_mnist data.data_module.data_set=pneumonia batch_size=128 +base_model_path=/home/tnpdung_79/TTC/logs/train/runs/2026-09-03_21-15-30/checkpoints/last.ckpt trainer.max_epochs=50 module.optimizer_name=adam module.lr=0.001 name=pneumonia-supcon-probe

echo ">>> 7/12 PneumoniaMNIST Sup Minority..."
python train.py experiment=finetune data=med_mnist data.data_module.data_set=pneumonia batch_size=128 +base_model_path=/home/tnpdung_79/TTC/logs/train/runs/2026-09-03_18-30-05/checkpoints/last.ckpt trainer.max_epochs=50 module.optimizer_name=adam module.lr=0.001 name=pneumonia-supmin-probe

echo ">>> 8/12 PneumoniaMNIST Sup Prototypes..."
python train.py experiment=finetune data=med_mnist data.data_module.data_set=pneumonia batch_size=128 +base_model_path=/home/tnpdung_79/TTC/logs/train/runs/2026-09-03_20-11-28/checkpoints/last.ckpt trainer.max_epochs=50 module.optimizer_name=adam module.lr=0.001 name=pneumonia-supproto-probe

# 3. FracAtlas (4 methods)
echo ">>> 9/12 FracAtlas Weighted CE..."
python train.py experiment=finetune experiment/specs=fracatlas batch_size=128 +base_model_path=/home/tnpdung_79/TTC/logs/train/runs/2026-09-04_02-53-18/checkpoints/last.ckpt trainer.max_epochs=50 module.optimizer_name=adam module.lr=0.001 name=fracatlas-weightedce-probe

echo ">>> 10/12 FracAtlas Standard SupCon..."
python train.py experiment=finetune experiment/specs=fracatlas batch_size=128 +base_model_path=/home/tnpdung_79/TTC/logs/train/runs/2026-09-04_01-35-25/checkpoints/last.ckpt trainer.max_epochs=50 module.optimizer_name=adam module.lr=0.001 name=fracatlas-supcon-probe

echo ">>> 11/12 FracAtlas Sup Minority..."
python train.py experiment=finetune experiment/specs=fracatlas batch_size=128 +base_model_path=/home/tnpdung_79/TTC/logs/train/runs/2026-09-03_22-58-59/checkpoints/last.ckpt trainer.max_epochs=50 module.optimizer_name=adam module.lr=0.001 name=fracatlas-supmin-probe

echo ">>> 12/12 FracAtlas Sup Prototypes..."
python train.py experiment=finetune experiment/specs=fracatlas batch_size=128 +base_model_path=/home/tnpdung_79/TTC/logs/train/runs/2026-09-04_00-17-05/checkpoints/last.ckpt trainer.max_epochs=50 module.optimizer_name=adam module.lr=0.001 name=fracatlas-supproto-probe

echo "================================================================================"
echo "✅ HOÀN THÀNH 12/12 OFFICIAL LIGHTNING PROBES CHO MEDICAL TABLE 2!"
echo "================================================================================"
