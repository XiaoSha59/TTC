#!/bin/bash
# ==============================================================================
# SCRIPT TẢI VÀ LƯU TRỮ TOÀN BỘ DATASET CỦA BÀI BÁO LÊN GCS BUCKET
# 
# Dataset gồm:
#   1. iNaturalist 2021 (Plants, Insects, Animals) - Table 1
#   2. MedMNIST (BreastMNIST, PneumoniaMNIST 224x224) - Table 2
#   3. FracAtlas (X-ray gãy xương) - Table 2
#
# Cách sử dụng:
#   bash scripts/setup_datasets_on_gcp.sh [TÊN_BUCKET]
# Ví dụ:
#   bash scripts/setup_datasets_on_gcp.sh ttc-paper-datasets-2025
# ==============================================================================

set -e

BUCKET_NAME=${1:-"ttc-paper-datasets-2025"}

echo "=========================================================="
echo " BẮT ĐẦU TẢI & ĐẨY TOÀN BỘ DATASET LÊN BUCKET: gs://$BUCKET_NAME"
echo "=========================================================="

# 0. Cài đặt các công cụ tối ưu tốc độ tải
echo "[*] Cài đặt aria2, unzip, python..."
sudo apt-get update && sudo apt-get install -y aria2 unzip wget python3-pip

DATA_DIR="$HOME/datasets"
mkdir -p "$DATA_DIR"

# ------------------------------------------------------------------------------
# 1. iNaturalist 2021 Mini (Tải đa luồng 16 kết nối)
# ------------------------------------------------------------------------------
echo ""
echo "=========================================================="
echo ">>> [1/3] Xử lý iNaturalist 2021 Dataset (Plants, Insects, Animals)..."
echo "=========================================================="
mkdir -p "$DATA_DIR/inat21" && cd "$DATA_DIR/inat21"

if [ ! -d "train_mini" ] && [ ! -d "train" ]; then
    echo "[*] Đang tải train_mini.tar.gz (16 luồng song song)..."
    aria2c -x 16 -s 16 -k 1M -c "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.tar.gz"
    
    echo "[*] Đang tải val.tar.gz (16 luồng song song)..."
    aria2c -x 16 -s 16 -k 1M -c "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz"

    echo "[*] Đang giải nén iNat21..."
    tar -xzf train_mini.tar.gz && rm -f train_mini.tar.gz
    tar -xzf val.tar.gz && rm -f val.tar.gz
fi

echo "[*] Đang đồng bộ iNat21 lên gs://$BUCKET_NAME/inat21/..."
gcloud storage cp -r "$DATA_DIR/inat21" "gs://$BUCKET_NAME/"
echo "[✓] iNaturalist 2021 đã được lưu an toàn trên Bucket!"

# ------------------------------------------------------------------------------
# 2. MedMNIST (BreastMNIST & PneumoniaMNIST 224x224)
# ------------------------------------------------------------------------------
echo ""
echo "=========================================================="
echo ">>> [2/3] Xử lý MedMNIST Datasets (BreastMNIST & PneumoniaMNIST)..."
echo "=========================================================="
pip install medmnist --break-system-packages 2>/dev/null || pip install medmnist
mkdir -p "$DATA_DIR/medmnist"

python3 -c "
import os
from medmnist import BreastMNIST, PneumoniaMNIST

data_dir = os.path.expanduser('$DATA_DIR/medmnist')
print('[*] Đang nạp BreastMNIST...')
for split in ['train', 'val', 'test']:
    BreastMNIST(split=split, download=True, root=data_dir, size=224)

print('[*] Đang nạp PneumoniaMNIST...')
for split in ['train', 'val', 'test']:
    PneumoniaMNIST(split=split, download=True, root=data_dir, size=224)
"

echo "[*] Đang đồng bộ MedMNIST lên gs://$BUCKET_NAME/medmnist/..."
gcloud storage cp -r "$DATA_DIR/medmnist" "gs://$BUCKET_NAME/"
echo "[✓] MedMNIST đã được lưu an toàn trên Bucket!"

# ------------------------------------------------------------------------------
# 3. FracAtlas Dataset (~2.5 GB)
# ------------------------------------------------------------------------------
echo ""
echo "=========================================================="
echo ">>> [3/3] Xử lý FracAtlas Dataset..."
echo "=========================================================="
mkdir -p "$DATA_DIR/fracatlas" && cd "$DATA_DIR/fracatlas"

if [ ! -d "FracAtlas" ]; then
    echo "[*] Đang tải FracAtlas từ Figshare API..."
    aria2c -x 16 -s 16 -k 1M -c -o FracAtlas.zip "https://api.figshare.com/v2/articles/22363012/files/41088458" || \
    curl -L -A "Mozilla/5.0" "https://api.figshare.com/v2/articles/22363012/files/41088458" -o FracAtlas.zip

    echo "[*] Đang giải nén FracAtlas..."
    unzip -q -o FracAtlas.zip
    rm -f FracAtlas.zip
fi

echo "[*] Đang đồng bộ FracAtlas lên gs://$BUCKET_NAME/fracatlas/..."
gcloud storage cp -r "$DATA_DIR/fracatlas" "gs://$BUCKET_NAME/"
echo "[✓] FracAtlas đã được lưu an toàn trên Bucket!"

# ------------------------------------------------------------------------------
# Hoàn tất
# ------------------------------------------------------------------------------
echo ""
echo "=========================================================="
echo "🎉 HOÀN TẤT 100%! TOÀN BỘ CÁC DATASET ĐÃ ĐƯỢC LƯU VÀO BUCKET:"
echo "   gs://$BUCKET_NAME/"
echo "   1. gs://$BUCKET_NAME/inat21/     (Plants, Insects, Animals)"
echo "   2. gs://$BUCKET_NAME/medmnist/   (BreastMNIST, PneumoniaMNIST)"
echo "   3. gs://$BUCKET_NAME/fracatlas/  (FracAtlas)"
echo "=========================================================="
