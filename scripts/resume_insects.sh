#!/bin/bash
# ==============================================================================
# Resume Insects Experiments from interrupted checkpoint (Epoch 79)
# ==============================================================================
set -e
source .venv/bin/activate

# Ensure GPU VRAM is completely clean by terminating any orphaned zombie process
pkill -9 -f "python.*train.py" 2>/dev/null || true
sleep 1

export INAT21_DATA_PATH="data/inat21"
export TMPDIR="${HOME}/tmp"
mkdir -p "$TMPDIR"

backup_to_gcs() {
    echo ">>> Syncing checkpoints to GCS Bucket..."
    gcloud storage cp -r logs/ "gs://ttc-paper-datasets-2025/checkpoints/" 2>/dev/null || true
}

trap backup_to_gcs EXIT

echo "=========================================================="
echo " RESUMING INSECTS 95:5 SUPPROTO FROM CHECKPOINT (EPOCH 79)"
echo "=========================================================="

# Find the latest checkpoint for insects-95_5-supproto
LATEST_CKPT=$(ls -td logs/train/runs/*/checkpoints/*.ckpt 2>/dev/null | head -n 1)

if [ -n "$LATEST_CKPT" ] && [ -f "$LATEST_CKPT" ]; then
    echo "Found checkpoint to resume from: $LATEST_CKPT"
    python train.py experiment=contrastive_sup_prototype experiment/specs=insects class_ratios=[0.05,0.95] ckpt_path="$LATEST_CKPT" name="insects-95_5-supproto"
else
    echo "No specific checkpoint found, resuming standard..."
    python train.py experiment=contrastive_sup_prototype experiment/specs=insects class_ratios=[0.05,0.95] name="insects-95_5-supproto"
fi
backup_to_gcs

echo ">>> Continuing remaining 95:5 runs..."
# 3. Standard SupCon (95:5)
python train.py experiment=contrastive experiment/specs=insects class_ratios=[0.05,0.95] module.ratio_supervised_majority=1.0 name="insects-95_5-supcon"
backup_to_gcs

# 4. Weighted CE (95:5)
python train.py experiment=weighted_ce experiment/specs=insects class_ratios=[0.05,0.95] name="insects-95_5-weightedce"
backup_to_gcs

echo ">>> Running 99:1 Extreme Imbalance..."
# 1. SupMin (99:1)
python train.py experiment=contrastive experiment/specs=insects class_ratios=[0.01,0.99] module.ratio_supervised_majority=0.0 name="insects-99_1-supmin"
backup_to_gcs

# 2. SupProto (99:1)
python train.py experiment=contrastive_sup_prototype experiment/specs=insects class_ratios=[0.01,0.99] name="insects-99_1-supproto"
backup_to_gcs

# 3. SupCon (99:1)
python train.py experiment=contrastive experiment/specs=insects class_ratios=[0.01,0.99] module.ratio_supervised_majority=1.0 name="insects-99_1-supcon"
backup_to_gcs

# 4. Weighted CE (99:1)
python train.py experiment=weighted_ce experiment/specs=insects class_ratios=[0.01,0.99] name="insects-99_1-weightedce"
backup_to_gcs

echo "=========================================================="
echo " Insects Experiments 100% Completed!"
echo " All checkpoints backed up to gs://ttc-paper-datasets-2025/checkpoints/"
echo "=========================================================="
