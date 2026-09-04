#!/bin/bash
# ==============================================================================
# Run iNaturalist Animals Experiments (Table 1 in Paper)
# Tests across 3 imbalance ratios: 50%-50%, 95%-5%, 99%-1%
# ==============================================================================

set -e
source .venv/bin/activate

echo "=========================================================="
echo " Starting iNat21 Animals Experiments (Table 1)"
echo "=========================================================="

export INAT21_DATA_PATH="data/inat21"

# Function to backup checkpoints to Bucket
backup_to_gcs() {
    echo ">>> Syncing checkpoints to GCS Bucket..."
    gcloud storage cp -r logs/ "gs://ttc-paper-datasets-2025/checkpoints/" 2>/dev/null || true
}

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

# Ensure val split is extracted and valid
if [ ! -d "data/inat21/val" ] || [ -z "$(ls -A data/inat21/val 2>/dev/null)" ]; then
    echo ">>> Preparing iNat21 validation dataset..."
    VAL_ARCHIVE=""
    if [ -f "data/inat21/val.tar.gz" ]; then
        VAL_ARCHIVE="data/inat21/val.tar.gz"
    elif [ -f "data/inat21/inat21/val.tar.gz" ]; then
        VAL_ARCHIVE="data/inat21/inat21/val.tar.gz"
    fi

    # Check if archive exists but is corrupted/truncated
    if [ -n "$VAL_ARCHIVE" ]; then
        echo "Testing archive integrity of $VAL_ARCHIVE..."
        if ! tar -tf "$VAL_ARCHIVE" >/dev/null 2>&1; then
            echo "⚠️ Archive $VAL_ARCHIVE is corrupted or truncated. Removing and downloading fresh copy..."
            rm -f "$VAL_ARCHIVE"
            VAL_ARCHIVE=""
        fi
    fi

    if [ -z "$VAL_ARCHIVE" ]; then
        echo "Downloading fresh val.tar.gz from GCS Bucket..."
        gcloud storage cp gs://ttc-paper-datasets-2025/inat21/val.tar.gz data/inat21/val.tar.gz || \
        gcloud storage cp -r gs://ttc-paper-datasets-2025/inat21/val data/inat21/ || true
        if [ -f "data/inat21/val.tar.gz" ]; then
            VAL_ARCHIVE="data/inat21/val.tar.gz"
        fi
    fi

    if [ -n "$VAL_ARCHIVE" ] && [ -f "$VAL_ARCHIVE" ]; then
        echo "Extracting $VAL_ARCHIVE to data/inat21/ ..."
        tar -xzf "$VAL_ARCHIVE" -C data/inat21/
    fi
fi

# Fix potential nested val/val
if [ -d "data/inat21/val/val" ]; then
    echo "Flattening nested data/inat21/val/val..."
    mv data/inat21/val/val/* data/inat21/val/ 2>/dev/null || true
    rmdir data/inat21/val/val 2>/dev/null || true
fi

RATIOS=("50_50:[0.5,0.5]" "95_5:[0.05,0.95]" "99_1:[0.01,0.99]")

for ITEM in "${RATIOS[@]}"; do
    NAME_TAG="${ITEM%%:*}"
    RATIO_VAL="${ITEM##*:}"
    
    echo ">>> Running Animals Ratio: $NAME_TAG ($RATIO_VAL)"

    # 1. Supervised Minority (Ours)
    python train.py experiment=contrastive experiment/specs=animals class_ratios=$RATIO_VAL module.ratio_supervised_majority=0.0 name="animals-$NAME_TAG-supmin"
    backup_to_gcs

    # 2. Supervised Prototypes (Ours)
    python train.py experiment=contrastive_sup_prototype experiment/specs=animals class_ratios=$RATIO_VAL name="animals-$NAME_TAG-supproto"
    backup_to_gcs

    # 3. Standard SupCon
    python train.py experiment=contrastive experiment/specs=animals class_ratios=$RATIO_VAL module.ratio_supervised_majority=1.0 name="animals-$NAME_TAG-supcon"
    backup_to_gcs

    # 4. Weighted Cross-Entropy
    python train.py experiment=weighted_ce experiment/specs=animals class_ratios=$RATIO_VAL name="animals-$NAME_TAG-weightedce"
    backup_to_gcs
done

echo "=========================================================="
echo " Animals Experiments Completed!"
echo " All checkpoints backed up to gs://ttc-paper-datasets-2025/checkpoints/"
echo "=========================================================="
