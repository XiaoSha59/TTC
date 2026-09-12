#!/usr/bin/env bash
# ==============================================================================
# Evaluate Linear Probes for Insects 50:50 and 99:1 SupPrototypes
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${BASE_DIR}"
source .venv/bin/activate
mkdir -p logs

LOG_FILE="${BASE_DIR}/logs/probe_supproto_results.log"

echo "==================================================================" | tee -a "${LOG_FILE}"
echo "🚀 RUNNING LINEAR PROBE FOR 50:50 AND 99:1 SUP-PROTOTYPES" | tee -a "${LOG_FILE}"
echo " Time: $(date)" | tee -a "${LOG_FILE}"
echo "==================================================================" | tee -a "${LOG_FILE}"

CKPT_50_50="/home/tnpdung_79/TTC/logs/train/runs/2026-09-06_01-53-15/checkpoints/last.ckpt"
CKPT_99_1="/home/tnpdung_79/TTC/logs/train/runs/2026-09-06_04-10-39/checkpoints/last.ckpt"

# 1. Probe 50:50 SupPrototypes
if [ -f "${CKPT_50_50}" ]; then
    echo ">>> [1/2] Linear Probing 50:50 SupPrototypes (Paper Target: 83.5%)..." | tee -a "${LOG_FILE}"
    python train.py \
        experiment=finetune \
        experiment/specs=insects \
        +base_model_path="${CKPT_50_50}" \
        trainer.max_epochs=50 \
        module.optimizer_name=adam \
        module.lr=0.001 \
        train_transform._target_=data.augmentation.SimCLRValTransform \
        data.data_module.num_workers=2 \
        data.data_module.persistent_workers=False \
        name="insects-50_50-supproto-official-probe" 2>&1 | tee -a "${LOG_FILE}"
else
    echo "❌ Error: CKPT_50_50 not found at ${CKPT_50_50}" | tee -a "${LOG_FILE}"
fi

# 2. Probe 99:1 SupPrototypes
if [ -f "${CKPT_99_1}" ]; then
    echo ">>> [2/2] Linear Probing 99:1 SupPrototypes (Paper Target: 78.9%)..." | tee -a "${LOG_FILE}"
    python train.py \
        experiment=finetune \
        experiment/specs=insects \
        +base_model_path="${CKPT_99_1}" \
        trainer.max_epochs=50 \
        module.optimizer_name=adam \
        module.lr=0.001 \
        train_transform._target_=data.augmentation.SimCLRValTransform \
        data.data_module.num_workers=2 \
        data.data_module.persistent_workers=False \
        name="insects-99_1-supproto-official-probe" 2>&1 | tee -a "${LOG_FILE}"
else
    echo "❌ Error: CKPT_99_1 not found at ${CKPT_99_1}" | tee -a "${LOG_FILE}"
fi

echo "==================================================================" | tee -a "${LOG_FILE}"
echo "🎉 BOTH PROBES COMPLETED at $(date)!" | tee -a "${LOG_FILE}"
echo "==================================================================" | tee -a "${LOG_FILE}"
