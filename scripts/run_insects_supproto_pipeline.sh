#!/bin/bash
# ==============================================================================
# Pipeline tự động chạy trọn gói SupPrototypes (2 lớp: Pretrain + Linear Probe)
# cho Insects trên các tỷ lệ: 50:50 và 99:1
# ==============================================================================

set -e
cd /home/tnpdung_79/TTC
git pull
source .venv/bin/activate

export INAT21_DATA_PATH="data/inat21"
export TMPDIR="${HOME}/tmp"
mkdir -p "$TMPDIR"

# Hàm đồng bộ checkpoint lên Google Cloud Storage để bảo toàn kết quả
backup_to_gcs() {
    echo ">>> Đồng bộ checkpoints lên GCS Bucket..."
    gcloud storage cp -r logs/ "gs://ttc-paper-datasets-2025/checkpoints/" 2>/dev/null || true
}

trap backup_to_gcs EXIT

# Danh sách tỷ lệ cần chạy (mặc định: 50:50 và 99:1)
RATIOS=("$@")
if [ ${#RATIOS[@]} -eq 0 ]; then
    RATIOS=("50_50:[0.5,0.5]" "99_1:[0.01,0.99]")
fi

echo "=================================================================="
echo "🚀 BẮT ĐẦU PIPELINE 2 LỚP CHO INSECTS SUP-PROTOTYPES"
echo " Danh sách tỷ lệ: ${RATIOS[*]}"
echo " Cấu hình: Physical Batch 256 | BF16 | 350 ep Pretrain + 50 ep Probe"
echo "=================================================================="

for ITEM in "${RATIOS[@]}"; do
    NAME_TAG="${ITEM%%:*}"
    RATIO_VAL="${ITEM##*:}"
    
    echo ""
    echo "=================================================================="
    echo ">>> [1/2] BẮT ĐẦU TỶ LỆ: INSECTS $NAME_TAG ($RATIO_VAL)"
    echo "=================================================================="

    RUN_NAME_PRETRAIN="insects-$NAME_TAG-supproto-350ep"
    NAMED_PRETRAIN_CKPT="logs/insects_${NAME_TAG}_supproto_pretrain_last.ckpt"

    # LỚP 1: Pre-training 350 epochs (SupPrototypes)
    echo ">>> [Lớp 1] Pre-training SupPrototypes 350 epochs..."
    python train.py \
        experiment=contrastive_sup_prototype \
        experiment/specs=insects \
        class_ratios=$RATIO_VAL \
        batch_size=256 \
        trainer.max_epochs=350 \
        module.lr=0.0625 \
        trainer.precision=bf16-mixed \
        data.data_module.persistent_workers=True \
        trainer.check_val_every_n_epoch=5 \
        name="$RUN_NAME_PRETRAIN"

    # Lấy checkpoint pre-train vừa hoàn thành và tạo bản sao định danh an toàn
    LATEST_CKPT=$(ls -td logs/train/runs/*/checkpoints/last.ckpt 2>/dev/null | head -n 1)
    if [ -z "$LATEST_CKPT" ] || [ ! -f "$LATEST_CKPT" ]; then
        echo "❌ Lỗi: Không tìm thấy checkpoint sau khi pretrain $NAME_TAG!"
        exit 1
    fi
    cp "$LATEST_CKPT" "$NAMED_PRETRAIN_CKPT"
    echo ">>> Pre-training $NAME_TAG hoàn tất! Checkpoint lưu tại: $NAMED_PRETRAIN_CKPT"
    backup_to_gcs

    # LỚP 2: Linear Probing 50 epochs (Resize + CenterCrop chuẩn S2.3)
    echo ">>> [Lớp 2] Linear Probing đánh giá đầu dò (50 epochs)..."
    python train.py \
        experiment=finetune \
        experiment/specs=insects \
        +base_model_path="$NAMED_PRETRAIN_CKPT" \
        trainer.max_epochs=50 \
        module.optimizer_name=adam \
        module.lr=0.001 \
        train_transform._target_=data.augmentation.SimCLRValTransform \
        data.data_module.persistent_workers=True \
        name="insects-$NAME_TAG-supproto-probe"

    echo ">>> Linear Probing $NAME_TAG hoàn tất!"
    backup_to_gcs

    echo "=================================================================="
    echo "🎉 HOÀN TẤT TRỌN VẸN CẢ 2 LỚP CHO INSECTS $NAME_TAG!"
    echo "=================================================================="
done

echo ""
echo "=================================================================="
echo "🏆 TOÀN BỘ CÁC TỶ LỆ (${RATIOS[*]}) ĐÃ HOÀN TẤT THÀNH CÔNG RỰC RỠ!"
echo "=================================================================="
