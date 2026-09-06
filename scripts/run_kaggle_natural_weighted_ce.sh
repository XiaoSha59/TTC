#!/bin/bash
# ==============================================================================
# Run Weighted Cross-Entropy for 3 Natural Datasets (Table 1):
# 1. Insects (50:50, 95:5, 99:1)
# 2. Plants  (50:50, 95:5, 99:1)
# 3. Animals (50:50, 95:5, 99:1)
# All runs use inverse frequency class_weights: ${class_ratios}
# ==============================================================================

echo "=================================================================="
echo "🌿 KHỞI ĐỘNG WEIGHTED CROSS-ENTROPY CHO 3 TẬP TỰ NHIÊN (TABLE 1)"
echo " Cấu hình: 100 Epochs | FP16 (16-mixed) | Batch Size 256"
echo "=================================================================="

# 1. Liên kết dữ liệu iNat21 Natural từ Kaggle Input
mkdir -p data
export INAT21_DATA_PATH="data/inat21"

if [ ! -d "data/inat21" ]; then
    if [ -d "/kaggle/input/inat21-natural/train_mini" ]; then
        echo ">>> Linking /kaggle/input/inat21-natural to data/inat21..."
        ln -s /kaggle/input/inat21-natural data/inat21
    elif [ -d "/kaggle/input/inat21-natural/inat21_natural/train_mini" ]; then
        echo ">>> Linking nested /kaggle/input/inat21-natural/inat21_natural to data/inat21..."
        ln -s /kaggle/input/inat21-natural/inat21_natural data/inat21
    elif [ -d "/kaggle/input/inat21-natural" ]; then
        echo ">>> Linking /kaggle/input/inat21-natural to data/inat21..."
        ln -s /kaggle/input/inat21-natural data/inat21
    elif [ -d "/kaggle/input/inat21" ]; then
        echo ">>> Linking /kaggle/input/inat21 to data/inat21..."
        ln -s /kaggle/input/inat21 data/inat21
    fi
fi

RATIOS=("50_50:[0.5,0.5]" "95_5:[0.05,0.95]" "99_1:[0.01,0.99]")

# ------------------------------------------------------------------------------
# 1. INSECTS (Họ Ong mật vs Họ Ong bắp cày)
# ------------------------------------------------------------------------------
echo ""
echo "=================================================================="
echo "🐝 [1/3] CHẠY WEIGHTED CE CHO TẬP INSECTS (3 TỶ LỆ)"
echo "=================================================================="

for ITEM in "${RATIOS[@]}"; do
    TAG="${ITEM%%:*}"
    RATIO="${ITEM##*:}"
    echo ">>> Đang chạy Insects Weighted CE $TAG ($RATIO)..."
    python train.py \
        experiment=weighted_ce \
        experiment/specs=insects \
        class_ratios=$RATIO \
        batch_size=256 \
        trainer.max_epochs=100 \
        trainer.precision=16-mixed \
        data.data_module.num_workers=2 \
        data.data_module.persistent_workers=False \
        name="insects-$TAG-weightedce" || true
    echo "✅ Xong Insects Weighted CE $TAG!"
done

# ------------------------------------------------------------------------------
# 2. PLANTS (Họ Cây sồi vs Bộ Tai hổ)
# ------------------------------------------------------------------------------
echo ""
echo "=================================================================="
echo "🌱 [2/3] CHẠY WEIGHTED CE CHO TẬP PLANTS (3 TỶ LỆ)"
echo "=================================================================="

for ITEM in "${RATIOS[@]}"; do
    TAG="${ITEM%%:*}"
    RATIO="${ITEM##*:}"
    echo ">>> Đang chạy Plants Weighted CE $TAG ($RATIO)..."
    python train.py \
        experiment=weighted_ce \
        experiment/specs=plants \
        class_ratios=$RATIO \
        batch_size=256 \
        trainer.max_epochs=100 \
        trainer.precision=16-mixed \
        data.data_module.num_workers=2 \
        data.data_module.persistent_workers=False \
        name="plants-$TAG-weightedce" || true
    echo "✅ Xong Plants Weighted CE $TAG!"
done

# ------------------------------------------------------------------------------
# 3. ANIMALS (Bộ Móng guốc chẵn vs Bộ Ăn thịt)
# ------------------------------------------------------------------------------
echo ""
echo "=================================================================="
echo "🐾 [3/3] CHẠY WEIGHTED CE CHO TẬP ANIMALS (3 TỶ LỆ)"
echo "=================================================================="

for ITEM in "${RATIOS[@]}"; do
    TAG="${ITEM%%:*}"
    RATIO="${ITEM##*:}"
    echo ">>> Đang chạy Animals Weighted CE $TAG ($RATIO)..."
    python train.py \
        experiment=weighted_ce \
        experiment/specs=animals \
        class_ratios=$RATIO \
        batch_size=256 \
        trainer.max_epochs=100 \
        trainer.precision=16-mixed \
        data.data_module.num_workers=2 \
        data.data_module.persistent_workers=False \
        name="animals-$TAG-weightedce" || true
    echo "✅ Xong Animals Weighted CE $TAG!"
done

echo ""
echo "=================================================================="
echo "🎉 TOÀN BỘ 9 MÔ HÌNH WEIGHTED CE TỰ NHIÊN ĐÃ HOÀN TẤT!"
echo "=================================================================="
