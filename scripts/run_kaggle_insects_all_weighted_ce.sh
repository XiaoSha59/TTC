#!/usr/bin/env bash
# ==============================================================================
# Run All Weighted Cross-Entropy Baselines on Full Insects (Table 1)
# 1. 50:50 Weighted CE (Paper Target: 82.4%)
# 2. 95:5 Weighted CE (Paper Target: 74.3%)
# 3. 99:1 Weighted CE (Paper Target: 66.8%)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${BASE_DIR}"

# Data path resolution
if [ -d "/kaggle/temp/inat21_full/train" ]; then
    export INAT21_DATA_PATH="/kaggle/temp/inat21_full"
elif [ -d "/kaggle/input/inat21-insects-full/train" ]; then
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

echo "=================================================================="
echo "🚀 STARTING FULL INSECTS WEIGHTED CROSS-ENTROPY BENCHMARKS"
echo " Data Path: ${INAT21_DATA_PATH}"
echo " Time: $(date)"
echo "=================================================================="

# 1. Insects 50:50 Weighted CE (Target: 82.4%)
echo ">>> [1/3] Running Weighted CE 50:50..."
python3 train.py \
    experiment=weighted_ce \
    experiment/specs=insects \
    class_ratios=[0.5,0.5] \
    batch_size=256 \
    trainer.max_epochs=50 \
    module.optimizer_name=adam \
    module.lr=0.001 \
    data.data_module.num_workers=4 \
    data.data_module.persistent_workers=True \
    name="insects-50_50-weighted_ce-full"

# 2. Insects 95:5 Weighted CE (Target: 74.3%)
echo ">>> [2/3] Running Weighted CE 95:5..."
python3 train.py \
    experiment=weighted_ce \
    experiment/specs=insects \
    class_ratios=[0.05,0.95] \
    batch_size=256 \
    trainer.max_epochs=50 \
    module.optimizer_name=adam \
    module.lr=0.001 \
    data.data_module.num_workers=4 \
    data.data_module.persistent_workers=True \
    name="insects-95_5-weighted_ce-full"

# 3. Insects 99:1 Weighted CE (Target: 66.8%)
echo ">>> [3/3] Running Weighted CE 99:1..."
python3 train.py \
    experiment=weighted_ce \
    experiment/specs=insects \
    class_ratios=[0.01,0.99] \
    batch_size=256 \
    trainer.max_epochs=50 \
    module.optimizer_name=adam \
    module.lr=0.001 \
    data.data_module.num_workers=4 \
    data.data_module.persistent_workers=True \
    name="insects-99_1-weighted_ce-full"

echo "=================================================================="
echo "🎉 ALL 3 WEIGHTED CE BENCHMARKS COMPLETED SUCCESSFULLY at $(date)!"
echo "=================================================================="
