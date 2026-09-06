#!/bin/bash
# ==============================================================================
# Chạy toàn bộ các mô hình còn lại của Insects trên GCP VM:
# 1. SupMinority 95:5 và 99:1 (350ep Pretrain + 50ep Probe)
# 2. SupPrototypes 95:5 (350ep Pretrain + 50ep Probe)
# 3. Standard SupCon 50:50, 95:5, 99:1 (350ep Pretrain + 50ep Probe)
# 4. Standard Cross-Entropy (Unweighted) 50:50, 95:5, 99:1 (350ep)
# ==============================================================================

set -e
cd /home/tnpdung_79/TTC
git pull || true
source .venv/bin/activate
export INAT21_DATA_PATH="data/inat21"
export TMPDIR="${HOME}/tmp"
mkdir -p "$TMPDIR"

backup_to_gcs() {
    echo ">>> Đồng bộ checkpoints lên GCS Bucket..."
    gcloud storage cp -r logs/ "gs://ttc-paper-datasets-2025/checkpoints/" 2>/dev/null || true
}
trap backup_to_gcs EXIT

echo "=================================================================="
echo "🚀 BẮT ĐẦU CHẠY CÁC MÔ HÌNH CÒN LẠI CỦA INSECTS TRÊN GCP VM"
echo " (SupMinority + SupPrototypes 95:5 + SupCon + Standard CE)"
echo "=================================================================="

# ------------------------------------------------------------------------------
# 1. SUPERVISED MINORITY (Ours) - 95:5 và 99:1 (350ep + 50ep probe)
# ------------------------------------------------------------------------------
echo ""
echo "=================================================================="
echo "🐝 [1/4] CHẠY SUPERVISED MINORITY (95:5 & 99:1)"
echo "=================================================================="

for ITEM in "95_5:[0.05,0.95]" "99_1:[0.01,0.99]"; do
    NAME_TAG="${ITEM%%:*}"
    RATIO_VAL="${ITEM##*:}"
    echo ">>> [SupMinority] Chạy 2 lớp Insects $NAME_TAG ($RATIO_VAL)..."
    python train.py \
        experiment=contrastive \
        experiment/specs=insects \
        class_ratios=$RATIO_VAL \
        module.ratio_supervised_majority=0.0 \
        batch_size=256 \
        trainer.max_epochs=350 \
        module.lr=0.0625 \
        trainer.precision=bf16-mixed \
        data.data_module.persistent_workers=True \
        trainer.check_val_every_n_epoch=5 \
        name="insects-$NAME_TAG-supmin-350ep"

    CKPT_MIN=$(ls -td logs/train/runs/*/checkpoints/last.ckpt 2>/dev/null | head -n 1)
    cp "$CKPT_MIN" "logs/insects_${NAME_TAG}_supmin_pretrain_last.ckpt"

    python train.py \
        experiment=finetune \
        experiment/specs=insects \
        +base_model_path="logs/insects_${NAME_TAG}_supmin_pretrain_last.ckpt" \
        trainer.max_epochs=50 \
        module.optimizer_name=adam \
        module.lr=0.001 \
        train_transform._target_=data.augmentation.SimCLRValTransform \
        data.data_module.persistent_workers=True \
        name="insects-$NAME_TAG-supmin-probe"

    backup_to_gcs
    echo "✅ Xong SupMinority Insects $NAME_TAG!"
done

# ------------------------------------------------------------------------------
# 2. SUPERVISED PROTOTYPES (Ours) - 95:5 (350ep + 50ep probe)
# ------------------------------------------------------------------------------
echo ""
echo "=================================================================="
echo "🐝 [2/4] CHẠY SUPERVISED PROTOTYPES CHO 95:5"
echo "=================================================================="

echo ">>> [SupPrototypes] Chạy 2 lớp Insects 95:5 ([0.05,0.95])..."
python train.py \
    experiment=contrastive_sup_prototype \
    experiment/specs=insects \
    class_ratios=[0.05,0.95] \
    batch_size=256 \
    trainer.max_epochs=350 \
    module.lr=0.0625 \
    trainer.precision=bf16-mixed \
    data.data_module.persistent_workers=True \
    trainer.check_val_every_n_epoch=5 \
    name="insects-95_5-supproto-350ep"

CKPT_PROTO=$(ls -td logs/train/runs/*/checkpoints/last.ckpt 2>/dev/null | head -n 1)
cp "$CKPT_PROTO" "logs/insects_95_5_supproto_pretrain_last.ckpt"

python train.py \
    experiment=finetune \
    experiment/specs=insects \
    +base_model_path="logs/insects_95_5_supproto_pretrain_last.ckpt" \
    trainer.max_epochs=50 \
    module.optimizer_name=adam \
    module.lr=0.001 \
    train_transform._target_=data.augmentation.SimCLRValTransform \
    data.data_module.persistent_workers=True \
    name="insects-95_5-supproto-probe"

backup_to_gcs
echo "✅ Xong SupPrototypes Insects 95:5!"

# ------------------------------------------------------------------------------
# 3. STANDARD SUPCON (Baseline) - 50:50, 95:5, 99:1 (350ep + 50ep probe)
# ------------------------------------------------------------------------------
echo ""
echo "=================================================================="
echo "🐝 [3/4] CHẠY STANDARD SUPCON BASELINE (3 TỶ LỆ)"
echo "=================================================================="

for ITEM in "50_50:[0.5,0.5]" "95_5:[0.05,0.95]" "99_1:[0.01,0.99]"; do
    NAME_TAG="${ITEM%%:*}"
    RATIO_VAL="${ITEM##*:}"
    echo ">>> [Standard SupCon] Chạy 2 lớp Insects $NAME_TAG ($RATIO_VAL)..."
    python train.py \
        experiment=contrastive \
        experiment/specs=insects \
        class_ratios=$RATIO_VAL \
        module.ratio_supervised_majority=1.0 \
        batch_size=256 \
        trainer.max_epochs=350 \
        module.lr=0.0625 \
        trainer.precision=bf16-mixed \
        data.data_module.persistent_workers=True \
        trainer.check_val_every_n_epoch=5 \
        name="insects-$NAME_TAG-supcon-350ep"

    CKPT_SUPCON=$(ls -td logs/train/runs/*/checkpoints/last.ckpt 2>/dev/null | head -n 1)
    cp "$CKPT_SUPCON" "logs/insects_${NAME_TAG}_supcon_pretrain_last.ckpt"

    python train.py \
        experiment=finetune \
        experiment/specs=insects \
        +base_model_path="logs/insects_${NAME_TAG}_supcon_pretrain_last.ckpt" \
        trainer.max_epochs=50 \
        module.optimizer_name=adam \
        module.lr=0.001 \
        train_transform._target_=data.augmentation.SimCLRValTransform \
        data.data_module.persistent_workers=True \
        name="insects-$NAME_TAG-supcon-probe"

    backup_to_gcs
    echo "✅ Xong Standard SupCon Insects $NAME_TAG!"
done

# ------------------------------------------------------------------------------
# 4. STANDARD CROSS-ENTROPY (Unweighted Baseline) - 50:50, 95:5, 99:1 (350 epochs)
# ------------------------------------------------------------------------------
echo ""
echo "=================================================================="
echo "🐝 [4/4] CHẠY STANDARD CROSS-ENTROPY UNWEIGHTED (3 TỶ LỆ - 350 EPOCHS)"
echo "=================================================================="

for ITEM in "50_50:[0.5,0.5]" "95_5:[0.05,0.95]" "99_1:[0.01,0.99]"; do
    NAME_TAG="${ITEM%%:*}"
    RATIO_VAL="${ITEM##*:}"
    echo ">>> [Standard CE] Chạy Standard Cross-Entropy Insects $NAME_TAG ($RATIO_VAL)..."
    python train.py \
        experiment/specs=insects \
        class_ratios=$RATIO_VAL \
        batch_size=256 \
        trainer.max_epochs=350 \
        trainer.precision=bf16-mixed \
        data.data_module.persistent_workers=True \
        name="insects-$NAME_TAG-ce"

    backup_to_gcs
    echo "✅ Xong Standard CE Insects $NAME_TAG!"
done

echo ""
echo "=================================================================="
echo "🎉 TOÀN BỘ CÁC MÔ HÌNH INSECTS ĐÃ HOÀN TẤT THÀNH CÔNG RỰC RỠ TRÊN VM!"
echo "=================================================================="
