#!/bin/bash
# ==============================================================================
# Run PneumoniaMNIST Experiments (Table 2 in Paper)
# 4 methods: SupMin, SupProto, SupCon, Weighted CE
# ==============================================================================

set -e
source .venv/bin/activate

echo "=========================================================="
echo " Starting PneumoniaMNIST Experiments (Table 2 - Medical)"
echo "=========================================================="

# Function to backup checkpoints to Bucket
backup_to_gcs() {
    echo ">>> Syncing checkpoints to GCS Bucket..."
    gcloud storage cp -r logs/ "gs://ttc-paper-datasets-2025/checkpoints/" 2>/dev/null || true
}

# Trap interruption to sync before exit
trap backup_to_gcs EXIT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1. PneumoniaMNIST - Supervised Minority (Ours)
echo "[1/4] Running PneumoniaMNIST - Supervised Minority (SupMin)..."
python train.py experiment=contrastive experiment/specs=generic_2_class data=med_mnist data.data_module.data_set=pneumonia batch_size=128 module.ratio_supervised_majority=0.0 name="pneumoniamnist-supmin"
backup_to_gcs

# 2. PneumoniaMNIST - Supervised Prototypes (Ours)
echo "[2/4] Running PneumoniaMNIST - Supervised Prototypes (SupProto)..."
python train.py experiment=contrastive_sup_prototype experiment/specs=generic_2_class data=med_mnist data.data_module.data_set=pneumonia batch_size=128 name="pneumoniamnist-supproto"
backup_to_gcs

# 3. PneumoniaMNIST - Standard SupCon (Baseline)
echo "[3/4] Running PneumoniaMNIST - Standard SupCon..."
python train.py experiment=contrastive experiment/specs=generic_2_class data=med_mnist data.data_module.data_set=pneumonia batch_size=128 module.ratio_supervised_majority=1.0 name="pneumoniamnist-supcon"
backup_to_gcs

# 4. PneumoniaMNIST - Weighted Cross-Entropy (Baseline)
echo "[4/4] Running PneumoniaMNIST - Weighted Cross-Entropy..."
python train.py experiment=weighted_ce experiment/specs=generic_2_class data=med_mnist data.data_module.data_set=pneumonia batch_size=128 name="pneumoniamnist-weightedce"
backup_to_gcs

echo "=========================================================="
echo " All PneumoniaMNIST experiments completed successfully!"
echo " All checkpoints backed up to gs://ttc-paper-datasets-2025/checkpoints/"
echo "=========================================================="
