#!/bin/bash
# ==============================================================================
# Run iNaturalist Insects Experiments (Table 1 in Paper)
# Tests across 3 imbalance ratios: 50%-50%, 95%-5%, 99%-1%
# ==============================================================================

set -e
source .venv/bin/activate

echo "=========================================================="
echo " Starting iNat21 Insects Experiments (Table 1)"
echo "=========================================================="

export INAT21_DATA_PATH="data/inat21"
export TMPDIR="${HOME}/tmp"
mkdir -p "$TMPDIR"

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

# Ensure validation dataset is extracted and valid
bash scripts/prepare_inat_val.sh

# Verify and purge any corrupt/unreadable images in dataset
python scripts/verify_and_clean_images.py data/inat21

RATIOS=("50_50:[0.5,0.5]" "95_5:[0.05,0.95]" "99_1:[0.01,0.99]")

for ITEM in "${RATIOS[@]}"; do
    NAME_TAG="${ITEM%%:*}"
    RATIO_VAL="${ITEM##*:}"
    
    echo ""
    echo "=========================================================="
    echo ">>> BẮT ĐẦU TỶ LỆ INSECTS: $NAME_TAG ($RATIO_VAL)"
    echo "=========================================================="

    # 1. Supervised Minority (Ours) - Bỏ qua 50:50 vì Paper không áp dụng
    if [ "$NAME_TAG" != "50_50" ]; then
        echo ">>> [1/4] Chạy 2 lớp SupMinority Insects $NAME_TAG..."
        python train.py experiment=contrastive experiment/specs=insects class_ratios=$RATIO_VAL module.ratio_supervised_majority=0.0 batch_size=256 trainer.max_epochs=350 module.lr=0.0625 trainer.precision=bf16-mixed data.data_module.persistent_workers=True trainer.check_val_every_n_epoch=5 name="insects-$NAME_TAG-supmin-350ep"
        CKPT_MIN=$(ls -td logs/train/runs/*/checkpoints/last.ckpt 2>/dev/null | head -n 1)
        cp "$CKPT_MIN" "logs/insects_${NAME_TAG}_supmin_pretrain_last.ckpt"
        python train.py experiment=finetune experiment/specs=insects +base_model_path="logs/insects_${NAME_TAG}_supmin_pretrain_last.ckpt" trainer.max_epochs=50 module.optimizer_name=adam module.lr=0.001 train_transform._target_=data.augmentation.SimCLRValTransform data.data_module.persistent_workers=True name="insects-$NAME_TAG-supmin-probe"
        backup_to_gcs
    fi

    # 2. Supervised Prototypes (Ours)
    echo ">>> [2/4] Chạy 2 lớp SupPrototypes Insects $NAME_TAG..."
    python train.py experiment=contrastive_sup_prototype experiment/specs=insects class_ratios=$RATIO_VAL batch_size=256 trainer.max_epochs=350 module.lr=0.0625 trainer.precision=bf16-mixed data.data_module.persistent_workers=True trainer.check_val_every_n_epoch=5 name="insects-$NAME_TAG-supproto-350ep"
    CKPT_PROTO=$(ls -td logs/train/runs/*/checkpoints/last.ckpt 2>/dev/null | head -n 1)
    cp "$CKPT_PROTO" "logs/insects_${NAME_TAG}_supproto_pretrain_last.ckpt"
    python train.py experiment=finetune experiment/specs=insects +base_model_path="logs/insects_${NAME_TAG}_supproto_pretrain_last.ckpt" trainer.max_epochs=50 module.optimizer_name=adam module.lr=0.001 train_transform._target_=data.augmentation.SimCLRValTransform data.data_module.persistent_workers=True name="insects-$NAME_TAG-supproto-probe"
    backup_to_gcs

    # 3. Standard SupCon (Baseline)
    echo ">>> [3/4] Chạy 2 lớp Standard SupCon Insects $NAME_TAG..."
    python train.py experiment=contrastive experiment/specs=insects class_ratios=$RATIO_VAL module.ratio_supervised_majority=1.0 batch_size=256 trainer.max_epochs=350 module.lr=0.0625 trainer.precision=bf16-mixed data.data_module.persistent_workers=True trainer.check_val_every_n_epoch=5 name="insects-$NAME_TAG-supcon-350ep"
    CKPT_SUPCON=$(ls -td logs/train/runs/*/checkpoints/last.ckpt 2>/dev/null | head -n 1)
    cp "$CKPT_SUPCON" "logs/insects_${NAME_TAG}_supcon_pretrain_last.ckpt"
    python train.py experiment=finetune experiment/specs=insects +base_model_path="logs/insects_${NAME_TAG}_supcon_pretrain_last.ckpt" trainer.max_epochs=50 module.optimizer_name=adam module.lr=0.001 train_transform._target_=data.augmentation.SimCLRValTransform data.data_module.persistent_workers=True name="insects-$NAME_TAG-supcon-probe"
    backup_to_gcs

    # 4. Weighted Cross-Entropy (Baseline)
    echo ">>> [4/4] Chạy Weighted Cross-Entropy Insects $NAME_TAG (100 epochs)..."
    python train.py experiment=weighted_ce experiment/specs=insects class_ratios=$RATIO_VAL batch_size=256 trainer.max_epochs=100 trainer.precision=bf16-mixed data.data_module.persistent_workers=True name="insects-$NAME_TAG-weightedce"
    backup_to_gcs
done

echo "=========================================================="
echo " Insects Experiments Completed!"
echo " All checkpoints backed up to gs://ttc-paper-datasets-2025/checkpoints/"
echo "=========================================================="
