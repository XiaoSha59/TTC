#!/usr/bin/env bash
# ==============================================================================
# Run All Supervised Contrastive (SupCon) Baselines on Full Insects (Table 1)
# 1. 50:50 SupCon (Pretrain 350ep + Probe 50ep, Target: 93.3%)
# 2. 95:5 SupCon  (Pretrain 350ep + Probe 50ep, Target: 81.6%)
# 3. 99:1 SupCon  (Pretrain 350ep + Probe 50ep, Target: 76.0%)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${BASE_DIR}"

# Data path resolution
if [ -d "/kaggle/input/inat21-insects-full/train" ]; then
    export INAT21_DATA_PATH="/kaggle/input/inat21-insects-full"
elif [ -d "/kaggle/input/inat21-insects-full/kaggle_insects_full_pkg/train" ]; then
    export INAT21_DATA_PATH="/kaggle/input/inat21-insects-full/kaggle_insects_full_pkg"
elif [ -d "${BASE_DIR}/data/inat21_full/train" ]; then
    export INAT21_DATA_PATH="${BASE_DIR}/data/inat21_full"
else
    echo "❌ Error: Could not find inat21_full dataset!"
    exit 1
fi

export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export TMPDIR="${TMPDIR:-/tmp}"
mkdir -p logs

run_supcon_pipeline() {
    local RATIO="$1"
    local RATIO_NAME="$2"
    local TARGET_ACC="$3"

    echo ""
    echo "=================================================================="
    echo "🚀 RUNNING SUPCON ${RATIO_NAME} (Target: ${TARGET_ACC})"
    echo "=================================================================="

    echo ">>> [STAGE 1/2] Contrastive Pretraining (350 Epochs, SupCon)..."
    python3 train.py \
        experiment=contrastive_sup_prototype \
        experiment/specs=insects \
        class_ratios="${RATIO}" \
        batch_size=256 \
        trainer.max_epochs=350 \
        module.lr=0.0625 \
        module.pull_mode=pull_to_pr \
        data.data_module.num_workers=4 \
        data.data_module.persistent_workers=False \
        trainer.check_val_every_n_epoch=5 \
        name="insects-${RATIO_NAME}-supcon-350ep-full"

    local CKPT_PATH=$(ls -td logs/train/runs/*/checkpoints/last.ckpt 2>/dev/null | head -n 1)
    local TARGET_CKPT="${BASE_DIR}/logs/insects_${RATIO_NAME}_supcon_full_pretrain_last.ckpt"

    if [ -f "${CKPT_PATH}" ]; then
        echo ">>> Saved backbone checkpoint to: ${TARGET_CKPT}"
        cp "${CKPT_PATH}" "${TARGET_CKPT}"
    else
        echo "❌ Error: Could not locate last.ckpt!"
        return 1
    fi

    echo ">>> [STAGE 2/2] Linear Probe Evaluation (50 Epochs, Adam lr=1e-3)..."
    python3 train.py \
        experiment=finetune \
        experiment/specs=insects \
        class_ratios="${RATIO}" \
        +base_model_path="${TARGET_CKPT}" \
        trainer.max_epochs=50 \
        module.optimizer_name=adam \
        module.lr=0.001 \
        train_transform._target_=data.augmentation.SimCLRValTransform \
        data.data_module.persistent_workers=True \
        name="insects-${RATIO_NAME}-supcon-full-probe"

    echo "🎉 SUPCON ${RATIO_NAME} COMPLETED SUCCESSFULLY!"
}

# 1. 50:50 SupCon (Target: 93.3%)
run_supcon_pipeline "[0.5,0.5]" "50_50" "93.3%"

# 2. 95:5 SupCon (Target: 81.6%)
run_supcon_pipeline "[0.05,0.95]" "95_5" "81.6%"

# 3. 99:1 SupCon (Target: 76.0%)
run_supcon_pipeline "[0.01,0.99]" "99_1" "76.0%"

echo ""
echo "=================================================================="
echo "🎉 ALL 3 SUPCON BENCHMARKS COMPLETED SUCCESSFULLY at $(date)!"
echo "=================================================================="
