#!/bin/bash
# ==============================================================================
# Script chạy trên Kaggle GPU (P100 / Dual T4):
# Chạy song song phần còn lại của tập INSECTS:
# 1. Weighted CE (cả 3 tỷ lệ: 50:50, 95:5, 99:1 - đã tích hợp class_weights)
# 2. Supervised Minority (95:5 và 99:1 - gồm 2 lớp: 350 ep Pretrain + 50 ep Probe)
# ==============================================================================

set -e

echo "=================================================================="
echo "🚀 KHỞI ĐỘNG KAGGLE PIPELINE: INSECTS (WEIGHTED CE + SUP-MINORITY)"
echo " Cấu hình phần cứng: FP16 (16-mixed) | Physical Batch 256"
echo "=================================================================="

export INAT21_DATA_PATH="data/inat21"
export TMPDIR="${PWD}/tmp"
mkdir -p "$TMPDIR" logs checkpoints

# ------------------------------------------------------------------------------
# PHẦN 1: HUẤN LUYỆN WEIGHTED CROSS-ENTROPY (3 TỶ LỆ)
# Đã tích hợp class_weights nghịch đảo tần số, chạy 100 epochs chuẩn baseline
# ------------------------------------------------------------------------------
echo ""
echo "=================================================================="
echo ">>> [PHẦN 1] CHẠY 3 MÔ HÌNH WEIGHTED CROSS-ENTROPY (INSECTS)"
echo "=================================================================="

CE_RATIOS=("50_50:[0.5,0.5]" "95_5:[0.05,0.95]" "99_1:[0.01,0.99]")

for ITEM in "${CE_RATIOS[@]}"; do
    NAME_TAG="${ITEM%%:*}"
    RATIO_VAL="${ITEM##*:}"

    echo ""
    echo ">>> Đang chạy Weighted CE Insects: $NAME_TAG ($RATIO_VAL)..."
    python train.py \
        experiment=weighted_ce \
        experiment/specs=insects \
        class_ratios=$RATIO_VAL \
        batch_size=256 \
        trainer.max_epochs=100 \
        trainer.precision=16-mixed \
        data.data_module.persistent_workers=True \
        name="insects-$NAME_TAG-weightedce"

    echo "✅ Hoàn tất Weighted CE cho $NAME_TAG!"
done

# ------------------------------------------------------------------------------
# PHẦN 2: HUẤN LUYỆN SUPERVISED MINORITY (2 LỚP TRỌN GÓI CHO 95:5 VÀ 99:1)
# Lớp 1: Pretrain 350 epochs (SupMinority ratio_supervised_majority=0.0)
# Lớp 2: Linear Probing 50 epochs (Adam 0.001 + Resize/CenterCrop chuẩn S2.3)
# ------------------------------------------------------------------------------
echo ""
echo "=================================================================="
echo ">>> [PHẦN 2] CHẠY TRỌN GÓI 2 LỚP CHO SUPERVISED MINORITY (INSECTS)"
echo "=================================================================="

MIN_RATIOS=("95_5:[0.05,0.95]" "99_1:[0.01,0.99]")

for ITEM in "${MIN_RATIOS[@]}"; do
    NAME_TAG="${ITEM%%:*}"
    RATIO_VAL="${ITEM##*:}"

    RUN_NAME_PRETRAIN="insects-$NAME_TAG-supmin-350ep"
    NAMED_CKPT="checkpoints/insects_${NAME_TAG}_supmin_pretrain_last.ckpt"

    echo ""
    echo "=================================================================="
    echo ">>> Bắt đầu SupMinority Insects $NAME_TAG ($RATIO_VAL)"
    echo "=================================================================="

    # LỚP 1: Pre-training 350 epochs
    echo ">>> [Lớp 1] Pre-training SupMinority 350 epochs..."
    python train.py \
        experiment=contrastive \
        experiment/specs=insects \
        class_ratios=$RATIO_VAL \
        module.ratio_supervised_majority=0.0 \
        batch_size=256 \
        trainer.max_epochs=350 \
        module.lr=0.0625 \
        trainer.precision=16-mixed \
        data.data_module.persistent_workers=True \
        trainer.check_val_every_n_epoch=5 \
        name="$RUN_NAME_PRETRAIN"

    # Lưu checkpoint Lớp 1
    LATEST_CKPT=$(ls -td logs/train/runs/*/checkpoints/last.ckpt 2>/dev/null | head -n 1)
    if [ -z "$LATEST_CKPT" ] || [ ! -f "$LATEST_CKPT" ]; then
        echo "❌ Lỗi: Không tìm thấy checkpoint sau khi pretrain $NAME_TAG!"
        exit 1
    fi
    cp "$LATEST_CKPT" "$NAMED_CKPT"
    echo ">>> Checkpoint Lớp 1 đã lưu tại: $NAMED_CKPT"

    # LỚP 2: Linear Probing 50 epochs (Clean Evaluation S2.3)
    echo ">>> [Lớp 2] Linear Probing 50 epochs..."
    python train.py \
        experiment=finetune \
        experiment/specs=insects \
        +base_model_path="$NAMED_CKPT" \
        trainer.max_epochs=50 \
        module.optimizer_name=adam \
        module.lr=0.001 \
        train_transform._target_=data.augmentation.SimCLRValTransform \
        trainer.precision=16-mixed \
        data.data_module.persistent_workers=True \
        name="insects-$NAME_TAG-supmin-probe"

    echo "✅ Hoàn tất trọn gói 2 lớp SupMinority cho Insects $NAME_TAG!"
done

echo ""
echo "=================================================================="
echo "🎉 CHÚC MỪNG: KAGGLE ĐÃ HOÀN TẤT TRỌN VẸN TOÀN BỘ NHIỆM VỤ ĐƯỢC GIAO!"
echo "=================================================================="
