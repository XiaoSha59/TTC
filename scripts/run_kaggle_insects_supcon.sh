#!/bin/bash
# ==============================================================================
# Run Standard SupCon 95:5 on Kaggle GPU T4 (350 Epochs Pretrain + 50 Epochs Probe)
# Auto-releases GPU when finished
# ==============================================================================

set -e
echo "=================================================================="
echo "🐝 KHỞI ĐỘNG STANDARD SUPCON 95:5 CHO INSECTS (KAGGLE GPU T4)"
echo " Cấu hình: 350 Epochs Pretrain + 50 Epochs Probe | FP16-mixed | Batch Size 256"
echo "=================================================================="

# 1. Tìm kiếm và liên kết tự động dữ liệu iNat21 Natural từ /kaggle/input
mkdir -p data
export INAT21_DATA_PATH="data/inat21"

if [ ! -d "data/inat21/train_mini" ] && [ ! -d "data/inat21/train" ]; then
    rm -rf data/inat21
    TRAIN_MINI=$(find /kaggle/input -type d -name "train_mini" 2>/dev/null | head -n 1)
    if [ -n "$TRAIN_MINI" ]; then
        FOUND=$(dirname "$TRAIN_MINI")
        echo ">>> Đã tìm thấy dataset root tại: $FOUND. Đang tạo liên kết sang data/inat21..."
        ln -s "$FOUND" data/inat21
    else
        echo "⚠️ Cảnh báo: Đang tìm kiếm các thư mục dataset khả dụng..."
        find /kaggle/input -maxdepth 4 -type d 2>/dev/null || true
    fi
fi

# Giai đoạn 1: Pre-training 350 Epochs
echo ""
echo ">>> [STAGE 1/2] Contrastive Pretraining 350 Epochs (Standard SupCon)..."
python train.py \
    experiment=contrastive \
    experiment/specs=insects \
    class_ratios=[0.05,0.95] \
    module.ratio_supervised_majority=1.0 \
    batch_size=256 \
    trainer.max_epochs=350 \
    module.lr=0.0625 \
    trainer.precision=16-mixed \
    data.data_module.num_workers=2 \
    data.data_module.persistent_workers=False \
    trainer.check_val_every_n_epoch=5 \
    name="insects-95_5-supcon-350ep-full"

CKPT_PATH=$(ls -td logs/train/runs/*/checkpoints/last.ckpt 2>/dev/null | head -n 1)
TARGET_CKPT="logs/insects_95_5_supcon_last.ckpt"

if [ -f "${CKPT_PATH}" ]; then
    echo ">>> Saving backbone checkpoint to: ${TARGET_CKPT}"
    mkdir -p logs
    cp "${CKPT_PATH}" "${TARGET_CKPT}"
else
    echo "❌ Error: Could not locate last.ckpt!"
    exit 1
fi

# Giai đoạn 2: Linear Probing Evaluation 50 Epochs
echo ""
echo ">>> [STAGE 2/2] Linear Probe Evaluation 50 Epochs (Adam lr=1e-3)..."
python train.py \
    experiment=finetune \
    experiment/specs=insects \
    +base_model_path="${TARGET_CKPT}" \
    trainer.max_epochs=50 \
    module.optimizer_name=adam \
    module.lr=0.001 \
    train_transform._target_=data.augmentation.SimCLRValTransform \
    data.data_module.num_workers=2 \
    data.data_module.persistent_workers=False \
    name="insects-95_5-supcon-full-probe"

echo ""
echo "=================================================================="
echo "🎉 HOÀN THÀNH TOÀN BỘ PIPELINE INSECTS 95:5 STANDARD SUPCON!"
echo "=================================================================="
