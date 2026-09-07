#!/usr/bin/env bash
# ==============================================================================
# Run Supervised Minority (Ours) 99:1 on Full iNat2021 Insects Dataset (Table 1)
# 1. Contrastive Pretraining: 350 Epochs (ratio_supervised_majority=0.0, bf16, batch 256)
# 2. Linear Probe Evaluation: 50 Epochs (Adam lr=1e-3)
# Paper Table 1 Target: 78.8% Balanced Accuracy (Prior run was 60.75% - Gap: -18.05%)
# Auto-shutdown: Safely stops VM when finished to stop billing
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${BASE_DIR}"
source .venv/bin/activate
export TMPDIR="${HOME}/tmp"
mkdir -p "$TMPDIR" logs

LOG_FILE="${BASE_DIR}/logs/insects_99_1_supmin_full.log"

echo "==================================================================" | tee -a "${LOG_FILE}"
echo "🚀 STARTING SUPERVISED MINORITY (Ours) 99:1 ON FULL INSECTS" | tee -a "${LOG_FILE}"
echo " Target (Paper Table 1): 78.8% Balanced Accuracy" | tee -a "${LOG_FILE}"
echo " Time: $(date)" | tee -a "${LOG_FILE}"
echo "==================================================================" | tee -a "${LOG_FILE}"

# ------------------------------------------------------------------------------
# STAGE 1: Contrastive Pretraining (350 Epochs)
# ------------------------------------------------------------------------------
echo "" | tee -a "${LOG_FILE}"
echo ">>> [STAGE 1/2] Contrastive Pretraining (350 Epochs, bf16, batch 256)..." | tee -a "${LOG_FILE}"

python train.py \
    experiment=contrastive \
    experiment/specs=insects \
    class_ratios=[0.01,0.99] \
    module.ratio_supervised_majority=0.0 \
    batch_size=256 \
    trainer.max_epochs=350 \
    module.lr=0.0625 \
    trainer.precision=bf16-mixed \
    data.data_module.num_workers=2 \
    data.data_module.persistent_workers=False \
    trainer.check_val_every_n_epoch=5 \
    name="insects-99_1-supmin-350ep-full" 2>&1 | tee -a "${LOG_FILE}"

CKPT_PATH=$(ls -td logs/train/runs/*/checkpoints/last.ckpt 2>/dev/null | head -n 1)
TARGET_CKPT="${BASE_DIR}/logs/insects_99_1_supmin_full_pretrain_last.ckpt"

if [ -f "${CKPT_PATH}" ]; then
    echo ">>> Saving backbone checkpoint to: ${TARGET_CKPT}" | tee -a "${LOG_FILE}"
    cp "${CKPT_PATH}" "${TARGET_CKPT}"
else
    echo "❌ Error: Could not locate last.ckpt!" | tee -a "${LOG_FILE}"
    exit 1
fi

# ------------------------------------------------------------------------------
# STAGE 2: Linear Probe Evaluation (50 Epochs)
# ------------------------------------------------------------------------------
echo "" | tee -a "${LOG_FILE}"
echo ">>> [STAGE 2/2] Linear Probe Evaluation (50 Epochs, Adam lr=1e-3)..." | tee -a "${LOG_FILE}"

python train.py \
    experiment=finetune \
    experiment/specs=insects \
    +base_model_path="${TARGET_CKPT}" \
    trainer.max_epochs=50 \
    module.optimizer_name=adam \
    module.lr=0.001 \
    train_transform._target_=data.augmentation.SimCLRValTransform \
    data.data_module.num_workers=2 \
    data.data_module.persistent_workers=False \
    name="insects-99_1-supmin-full-probe" 2>&1 | tee -a "${LOG_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo "==================================================================" | tee -a "${LOG_FILE}"
echo "🎉 PIPELINE COMPLETED SUCCESSFULLY at $(date)!" | tee -a "${LOG_FILE}"
echo "💤 Training finished! Powering off VM in 60 seconds to stop billing..." | tee -a "${LOG_FILE}"
echo "==================================================================" | tee -a "${LOG_FILE}"

sync
sleep 60
sudo shutdown -h now
