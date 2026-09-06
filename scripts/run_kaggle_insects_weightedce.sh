#!/bin/bash
# ==============================================================================
# Run Weighted Cross-Entropy Baseline for Insects on Kaggle GPU T4 (350 Epochs)
# Covers: Insects 95:5 and Insects 99:1 (and 50:50)
# ==============================================================================

echo "=================================================================="
echo "🐝 KHỞI ĐỘNG WEIGHTED CROSS-ENTROPY CHO INSECTS (KAGGLE GPU T4)"
echo " Cấu hình: 350 Epochs | FP16-mixed | Batch Size 256 | Inverse Class Weights"
echo "=================================================================="

# 1. Tìm kiếm và liên kết tự động dữ liệu iNat21 Natural từ /kaggle/input
mkdir -p data
export INAT21_DATA_PATH="data/inat21"

rm -rf data/inat21
TRAIN_MINI=$(find /kaggle/input -type d -name "train_mini" 2>/dev/null | head -n 1)
if [ -n "$TRAIN_MINI" ]; then
    FOUND=$(dirname "$TRAIN_MINI")
    echo ">>> Đã tìm thấy dataset root tại: $FOUND. Đang tạo liên kết sang data/inat21..."
    ln -sf "$FOUND" data/inat21
else
    echo "⚠️ Đang tìm train thông thường..."
    TRAIN_DIR=$(find /kaggle/input -type d -name "train" 2>/dev/null | head -n 1)
    if [ -n "$TRAIN_DIR" ]; then
        FOUND=$(dirname "$TRAIN_DIR")
        ln -sf "$FOUND" data/inat21
    fi
fi

# ------------------------------------------------------------------------------
# 1. Insects 95:5 (Ưu tiên hàng đầu)
# ------------------------------------------------------------------------------
echo ""
echo "=================================================================="
echo ">>> [1/2] Đang chạy Insects Weighted CE 95:5 (Target: 63.4% Paper)..."
echo "=================================================================="
python train.py \
    experiment=weighted_ce \
    experiment/specs=insects \
    class_ratios=[0.05,0.95] \
    batch_size=256 \
    trainer.max_epochs=350 \
    trainer.precision=16-mixed \
    data.data_module.num_workers=2 \
    data.data_module.persistent_workers=False \
    name="insects-95_5-weightedce-full" || true

# ------------------------------------------------------------------------------
# 2. Insects 99:1 (Chạy nối tiếp trong phiên 12h)
# ------------------------------------------------------------------------------
echo ""
echo "=================================================================="
echo ">>> [2/2] Đang chạy Insects Weighted CE 99:1 (Target: 62.8% Paper)..."
echo "=================================================================="
python train.py \
    experiment=weighted_ce \
    experiment/specs=insects \
    class_ratios=[0.01,0.99] \
    batch_size=256 \
    trainer.max_epochs=350 \
    trainer.precision=16-mixed \
    data.data_module.num_workers=2 \
    data.data_module.persistent_workers=False \
    name="insects-99_1-weightedce-full" || true

echo ""
echo "=================================================================="
echo "🎉 HOÀN THÀNH TẤT CẢ CÁC TỶ LỆ INSECTS WEIGHTED CE TRÊN KAGGLE!"
echo "=================================================================="
