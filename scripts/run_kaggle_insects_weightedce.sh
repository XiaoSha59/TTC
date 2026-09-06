#!/bin/bash
# ==============================================================================
# Run Weighted Cross-Entropy Baseline for Insects on Kaggle GPU T4 (350 Epochs)
# Focus: Insects 95:5 (and 50:50, 99:1)
# ==============================================================================

set -e
echo "=================================================================="
echo "🐝 KHỞI ĐỘNG WEIGHTED CROSS-ENTROPY CHO INSECTS (KAGGLE GPU T4)"
echo " Cấu hình: 350 Epochs | FP16-mixed | Batch Size 256 | Inverse Class Weights"
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

# Chạy ưu tiên Insects 95:5
echo ""
echo ">>> [1/1] Đang chạy Insects Weighted CE 95:5 (Target: 63.4% Paper)..."
python train.py \
    experiment=weighted_ce \
    experiment/specs=insects \
    class_ratios=[0.05,0.95] \
    batch_size=256 \
    trainer.max_epochs=350 \
    trainer.precision=16-mixed \
    data.data_module.num_workers=2 \
    data.data_module.persistent_workers=False \
    name="insects-95_5-weightedce-full"

echo ""
echo "=================================================================="
echo "🎉 HOÀN THÀNH INSECTS 95:5 WEIGHTED CE TRÊN KAGGLE!"
echo "=================================================================="
