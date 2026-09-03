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

# 1. BreastMNIST - Supervised Minority (Ours)
echo "[1/4] Running BreastMNIST - Supervised Minority..."
python train.py experiment=contrastive experiment/specs=med_mnist data=med_mnist module.ratio_supervised_majority=0.0 name="breastmnist-supmin"

# 2. BreastMNIST - Supervised Prototypes (Ours)
echo "[2/4] Running BreastMNIST - Supervised Prototypes..."
python train.py experiment=contrastive_sup_prototype experiment/specs=med_mnist data=med_mnist name="breastmnist-supproto"

# 3. BreastMNIST - Standard SupCon (Baseline)
echo "[3/4] Running BreastMNIST - Standard SupCon..."
python train.py experiment=contrastive experiment/specs=med_mnist data=med_mnist module.ratio_supervised_majority=1.0 name="breastmnist-supcon"

# 4. BreastMNIST - Weighted Cross-Entropy (Baseline)
echo "[4/4] Running BreastMNIST - Weighted Cross-Entropy..."
python train.py experiment=weighted_ce experiment/specs=med_mnist data=med_mnist name="breastmnist-weightedce"

echo "=========================================================="
echo " All MedMNIST experiments completed!"
echo " Check WandB dashboard for live metrics and curves."
echo "=========================================================="
