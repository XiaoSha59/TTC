#!/bin/bash
# ==============================================================================
# Run MedMNIST Experiments (Table 2 in Paper)
# BreastMNIST & PneumoniaMNIST across 4 methods
# ==============================================================================

set -e
source .venv/bin/activate

echo "=========================================================="
echo " Starting MedMNIST Experiments (Table 2)"
echo "=========================================================="

# Function to backup checkpoints to Bucket
backup_to_gcs() {
    echo ">>> Syncing checkpoints to GCS Bucket..."
    gcloud storage cp -r logs/ "gs://ttc-paper-datasets-2025/checkpoints/" 2>/dev/null || true
}

# Trap interruption to sync before exit
trap backup_to_gcs EXIT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1. BreastMNIST - Supervised Minority (Ours)
echo "[1/4] Running BreastMNIST - Supervised Minority..."
python train.py experiment=contrastive experiment/specs=generic_2_class data=med_mnist data.data_module.data_set=breast batch_size=128 module.ratio_supervised_majority=0.0 name="breastmnist-supmin"
backup_to_gcs

# 2. BreastMNIST - Supervised Prototypes (Ours)
echo "[2/4] Running BreastMNIST - Supervised Prototypes..."
python train.py experiment=contrastive_sup_prototype experiment/specs=generic_2_class data=med_mnist data.data_module.data_set=breast batch_size=128 name="breastmnist-supproto"
backup_to_gcs

# 3. BreastMNIST - Standard SupCon (Baseline)
echo "[3/4] Running BreastMNIST - Standard SupCon..."
python train.py experiment=contrastive experiment/specs=generic_2_class data=med_mnist data.data_module.data_set=breast batch_size=128 module.ratio_supervised_majority=1.0 name="breastmnist-supcon"
backup_to_gcs

# 4. BreastMNIST - Weighted Cross-Entropy (Baseline)
echo "[4/4] Running BreastMNIST - Weighted Cross-Entropy..."
python train.py experiment=weighted_ce experiment/specs=generic_2_class data=med_mnist data.data_module.data_set=breast batch_size=128 name="breastmnist-weightedce"
backup_to_gcs

echo "=========================================================="
echo " All MedMNIST experiments completed!"
echo " All checkpoints backed up to gs://ttc-paper-datasets-2025/checkpoints/"
echo " Check WandB dashboard for live metrics and curves."
echo "=========================================================="
