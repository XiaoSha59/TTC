#!/bin/bash
# ==============================================================================
# Run iNaturalist Plants Experiments (Table 1 in Paper)
# Tests across 3 imbalance ratios: 50%-50%, 95%-5%, 99%-1%
# ==============================================================================

set -e
source .venv/bin/activate

echo "=========================================================="
echo " Starting iNat21 Plants Experiments (Table 1)"
echo "=========================================================="
export INAT21_DATA_PATH="data/inat21"

# Function to backup checkpoints to Bucket
backup_to_gcs() {
    echo ">>> Syncing checkpoints to GCS Bucket..."
    gcloud storage cp -r logs/ "gs://ttc-paper-datasets-2025/checkpoints/" 2>/dev/null || true
}

# Trap interruption to sync before exit
trap backup_to_gcs EXIT

mkdir -p data/inat21

if [ ! -d "data/inat21/train_mini" ] && [ ! -d "data/inat21/train" ]; then
    if [ -d "data/inat21/inat21/train_mini" ]; then
        echo "Relocating nested train_mini..."
        mv data/inat21/inat21/train_mini data/inat21/
    else
        echo "Downloading iNat21 dataset from Bucket..."
        gcloud storage cp -r gs://ttc-paper-datasets-2025/inat21/* data/inat21/ || true
        if [ -d "data/inat21/inat21/train_mini" ]; then
            mv data/inat21/inat21/train_mini data/inat21/
        fi
    fi
fi

# Ensure validation dataset is extracted and valid
bash scripts/prepare_inat_val.sh

RATIOS=("50_50:[0.5,0.5]" "95_5:[0.05,0.95]" "99_1:[0.01,0.99]")

for ITEM in "${RATIOS[@]}"; do
    NAME_TAG="${ITEM%%:*}"
    RATIO_VAL="${ITEM##*:}"
    
    echo ">>> Running Plants Ratio: $NAME_TAG ($RATIO_VAL)"

    # 1. Supervised Minority (Ours)
    python train.py experiment=contrastive experiment/specs=plants class_ratios=$RATIO_VAL module.ratio_supervised_majority=0.0 name="plants-$NAME_TAG-supmin"
    backup_to_gcs

    # 2. Supervised Prototypes (Ours)
    python train.py experiment=contrastive_sup_prototype experiment/specs=plants class_ratios=$RATIO_VAL name="plants-$NAME_TAG-supproto"
    backup_to_gcs

    # 3. Standard SupCon
    python train.py experiment=contrastive experiment/specs=plants class_ratios=$RATIO_VAL module.ratio_supervised_majority=1.0 name="plants-$NAME_TAG-supcon"
    backup_to_gcs

    # 4. Weighted Cross-Entropy
    python train.py experiment=weighted_ce experiment/specs=plants class_ratios=$RATIO_VAL name="plants-$NAME_TAG-weightedce"
    backup_to_gcs
done

echo "=========================================================="
echo " Plants Experiments Completed!"
echo " All checkpoints backed up to gs://ttc-paper-datasets-2025/checkpoints/"
echo "=========================================================="
