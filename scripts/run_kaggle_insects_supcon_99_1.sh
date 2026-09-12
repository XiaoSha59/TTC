#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${BASE_DIR}"

if [ -d "/kaggle/temp/inat21_full/train" ]; then
    export INAT21_DATA_PATH="/kaggle/temp/inat21_full"
elif [ -d "${BASE_DIR}/data/inat21_full/train" ]; then
    export INAT21_DATA_PATH="${BASE_DIR}/data/inat21_full"
fi

export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export TMPDIR="${TMPDIR:-/tmp}"
mkdir -p logs

echo "=================================================================="
echo "🚀 RUNNING SUPCON 99:1 (Target: 76.0% Balanced Accuracy)"
echo " Data: ${INAT21_DATA_PATH}"
echo "=================================================================="

echo ">>> [STAGE 1/2] Pretraining (350 Epochs, SupCon 99:1)..."
python3 train.py \
    experiment=contrastive_sup_prototype \
    experiment/specs=insects \
    class_ratios=[0.01,0.99] \
    batch_size=256 \
    trainer.max_epochs=350 \
    module.lr=0.0625 \
    module.pull_mode=pull_to_pr \
    data.data_module.num_workers=2 \
    data.data_module.persistent_workers=False \
    trainer.check_val_every_n_epoch=5 \
    name="insects-99_1-supcon-350ep-full"

CKPT_PATH=$(ls -td logs/train/runs/*/checkpoints/last.ckpt 2>/dev/null | head -n 1)
TARGET_CKPT="${BASE_DIR}/logs/insects_99_1_supcon_full_pretrain_last.ckpt"
cp "${CKPT_PATH}" "${TARGET_CKPT}"

echo ">>> [STAGE 2/2] Linear Probe (50 Epochs, Adam lr=1e-3)..."
python3 train.py \
    experiment=finetune \
    experiment/specs=insects \
    class_ratios=[0.01,0.99] \
    +base_model_path="${TARGET_CKPT}" \
    trainer.max_epochs=50 \
    module.optimizer_name=adam \
    module.lr=0.001 \
    train_transform._target_=data.augmentation.SimCLRValTransform \
    data.data_module.persistent_workers=True \
    name="insects-99_1-supcon-full-probe"

echo "🎉 SUPCON 99:1 COMPLETED SUCCESSFULLY at $(date)!"
