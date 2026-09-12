#!/usr/bin/env bash
# ==============================================================================
# Run Supervised Prototypes (Ours) 50:50 on Full iNat2021 Insects Dataset (Table 1)
# 1. Contrastive Pretraining: 350 Epochs (SupPrototypes, bf16, batch 256)
# 2. Linear Probe Evaluation: 50 Epochs (Adam lr=1e-3)
# Paper Table 1 Target: 93.0% Balanced Accuracy
# Auto-shutdown: Safely stops VM when finished to stop billing
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${BASE_DIR}"

# Source venv if present
if [ -f "${BASE_DIR}/.venv/bin/activate" ]; then
    source "${BASE_DIR}/.venv/bin/activate"
elif [ -f "/home/tnpdung_79/TTC/.venv/bin/activate" ]; then
    source "/home/tnpdung_79/TTC/.venv/bin/activate"
fi

export PYTHONUNBUFFERED=1
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export TMPDIR="${HOME}/tmp"
export INAT21_DATA_PATH="${BASE_DIR}/data/inat21_full"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p "$TMPDIR" logs

LOG_FILE="${BASE_DIR}/logs/insects_50_50_supproto_full.log"
SUCCESS_FLAG="${BASE_DIR}/logs/insects_50_50_completed.flag"
rm -f "${SUCCESS_FLAG}"

cleanup() {
    EXIT_CODE=$?
    echo "==================================================================" | tee -a "${LOG_FILE}"
    if [ "${EXIT_CODE}" -eq 0 ] && [ -f "${SUCCESS_FLAG}" ]; then
        echo "🛑 ALL STAGES COMPLETED SUCCESSFULLY. SYNCING WANDB & POWERING OFF IN 60 SECONDS..." | tee -a "${LOG_FILE}"
        wandb sync --sync-all 2>/dev/null || true
        sleep 60
        sudo poweroff || true
    else
        echo "⚠️ JOB STOPPED WITH CODE ${EXIT_CODE}. Leaving VM alive for investigation." | tee -a "${LOG_FILE}"
    fi
    echo "==================================================================" | tee -a "${LOG_FILE}"
}
trap cleanup EXIT

echo "==================================================================" | tee -a "${LOG_FILE}"
echo "🚀 STARTING SUPERVISED PROTOTYPES (Ours) 50:50 ON FULL INSECTS (350 Epochs)" | tee -a "${LOG_FILE}"
echo " Target (Paper Table 1): 93.0% Balanced Accuracy" | tee -a "${LOG_FILE}"
echo " Data Path: ${INAT21_DATA_PATH}" | tee -a "${LOG_FILE}"
echo " Time: $(date)" | tee -a "${LOG_FILE}"
echo "==================================================================" | tee -a "${LOG_FILE}"

# ------------------------------------------------------------------------------
# STAGE 1: Contrastive Pretraining (350 Epochs)
# ------------------------------------------------------------------------------
echo "" | tee -a "${LOG_FILE}"
echo ">>> [STAGE 1/2] Contrastive Pretraining (350 Epochs, bf16, batch 256)..." | tee -a "${LOG_FILE}"

python3 train.py \
    experiment=contrastive_sup_prototype \
    experiment/specs=insects \
    class_ratios=[0.5,0.5] \
    batch_size=256 \
    trainer.max_epochs=350 \
    module.lr=0.0625 \
    trainer.precision=bf16-mixed \
    data.data_module.num_workers=2 \
    data.data_module.persistent_workers=False \
    trainer.check_val_every_n_epoch=5 \
    name="insects-50_50-supproto-350ep-full" 2>&1 | tee -a "${LOG_FILE}"

CKPT_PATH=$(ls -td logs/train/runs/*/checkpoints/last.ckpt 2>/dev/null | head -n 1)
TARGET_CKPT="${BASE_DIR}/logs/insects_50_50_supproto_full_pretrain_last.ckpt"

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

python3 train.py \
    experiment=finetune \
    experiment/specs=insects \
    class_ratios=[0.5,0.5] \
    +base_model_path="${TARGET_CKPT}" \
    trainer.max_epochs=50 \
    module.optimizer_name=adam \
    module.lr=0.001 \
    train_transform._target_=data.augmentation.SimCLRValTransform \
    data.data_module.persistent_workers=True \
    name="insects-50_50-supproto-full-probe" 2>&1 | tee -a "${LOG_FILE}"

touch "${SUCCESS_FLAG}"
echo "" | tee -a "${LOG_FILE}"
echo "==================================================================" | tee -a "${LOG_FILE}"
echo "🎉 INSECTS 50:50 SUP-PROTOTYPES COMPLETED SUCCESSFULLY at $(date)!" | tee -a "${LOG_FILE}"
echo "==================================================================" | tee -a "${LOG_FILE}"
