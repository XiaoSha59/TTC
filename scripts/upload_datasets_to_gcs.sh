#!/bin/bash
# ==============================================================================
# Script: Tải toàn bộ Dataset và Đẩy lên Google Cloud Storage Bucket
# Tự động kiểm tra: Dung lượng ổ đĩa, Quyền Bucket, Tải nối tiếp, Dọn rác
#
# Cách sử dụng:
#   bash scripts/upload_datasets_to_gcs.sh <TÊN_BUCKET>
# Ví dụ:
#   bash scripts/upload_datasets_to_gcs.sh ttc-paper-datasets-2025
# ==============================================================================

set -e

BUCKET_NAME=$1

# ------------------------------------------------------------------------------
# 0. Kiểm tra đối số đầu vào
# ------------------------------------------------------------------------------
if [ -z "$BUCKET_NAME" ]; then
    echo "[!] Lỗi: Bạn chưa cung cấp tên Bucket."
    echo "    Cú pháp: bash scripts/upload_datasets_to_gcs.sh <TÊN_BUCKET>"
    exit 1
fi

echo "=========================================================="
echo " KHỞI ĐỘNG TIẾN TRÌNH TẢI & ĐẨY DỮ LIỆU LÊN BUCKET"
echo " Đích đến: gs://$BUCKET_NAME"
echo "=========================================================="

# ------------------------------------------------------------------------------
# 1. Kiểm tra dung lượng ổ đĩa khả dụng (Cần tối thiểu 80GB)
# ------------------------------------------------------------------------------
FREE_GB=$(df -BG / | awk 'NR==2 {gsub("G",""); print $4}')
echo "[*] Dung lượng ổ đĩa khả dụng hiện tại: ${FREE_GB} GB"

if [ "$FREE_GB" -lt 70 ]; then
    echo "[!] CẢNH BÁO: Dung lượng ổ đĩa còn lại (${FREE_GB}GB) không đủ để tải và giải nén 50GB."
    echo "    Vui lòng tăng kích thước Boot Disk lên tối thiểu 100GB - 150GB trên GCP Console trước khi chạy tiếp."
    exit 1
fi

# ------------------------------------------------------------------------------
# 2. Kiểm tra quyền truy cập vào Bucket
# ------------------------------------------------------------------------------
echo "[*] Đang kiểm tra quyền ghi vào Bucket gs://$BUCKET_NAME..."
if ! gcloud storage cp /dev/null "gs://$BUCKET_NAME/.test_access" &> /dev/null; then
    echo "[!] Lỗi: Không có quyền ghi vào Bucket gs://$BUCKET_NAME."
    echo "    Vui lòng chạy lệnh sau để đăng nhập tài khoản của bạn:"
    echo "    >>> gcloud auth login"
    exit 1
fi
gcloud storage rm "gs://$BUCKET_NAME/.test_access" &> /dev/null || true
echo "[✓] Quyền truy cập Bucket hợp lệ!"

DATA_DIR="$HOME/datasets"
mkdir -p "$DATA_DIR"

# ------------------------------------------------------------------------------
# 3. Tải & Đẩy tập dữ liệu Y tế FracAtlas (~2.5 GB)
# ------------------------------------------------------------------------------
echo ""
echo "=========================================================="
echo ">>> [1/2] Đang xử lý FracAtlas Dataset..."
echo "=========================================================="
mkdir -p "$DATA_DIR/fracatlas"
cd "$DATA_DIR/fracatlas"

if [ ! -f "FracAtlas.zip" ] || [ $(stat -c%s "FracAtlas.zip" 2>/dev/null || echo 0) -lt 1000000 ]; then
    echo "[*] Đang tải FracAtlas từ Figshare API..."
    curl -L -A "Mozilla/5.0" "https://api.figshare.com/v2/articles/22363012/files/41088458" -o FracAtlas.zip
fi

if [ ! -d "FracAtlas" ]; then
    echo "[*] Đang giải nén FracAtlas.zip..."
    unzip -q -o FracAtlas.zip
fi

echo "[*] Đang tải FracAtlas lên gs://$BUCKET_NAME/fracatlas/..."
gcloud storage cp -r "$DATA_DIR/fracatlas" "gs://$BUCKET_NAME/"
echo "[✓] Đã hoàn thành tải và lưu trữ FracAtlas!"

# Dọn dẹp file zip để tiết kiệm dung lượng đĩa
rm -f FracAtlas.zip

# ------------------------------------------------------------------------------
# 4. Tải & Đẩy tập dữ liệu iNaturalist 2021 Mini (~42 GB)
# ------------------------------------------------------------------------------
echo ""
echo "=========================================================="
echo ">>> [2/2] Đang xử lý iNaturalist 2021 Dataset..."
echo "=========================================================="
mkdir -p "$DATA_DIR/inat21"
cd "$DATA_DIR/inat21"

echo "[*] Đang tải file train_mini.tar.gz (hỗ trợ resume)..."
wget -c "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.tar.gz"

echo "[*] Đang tải file val.tar.gz (hỗ trợ resume)..."
wget -c "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz"

echo "[*] Đang giải nén iNaturalist 2021 (vui lòng đợi 2-3 phút)..."
tar -xzf train_mini.tar.gz
tar -xzf val.tar.gz

echo "[*] Đang tải iNaturalist 2021 lên gs://$BUCKET_NAME/inat21/..."
gcloud storage cp -r "$DATA_DIR/inat21" "gs://$BUCKET_NAME/"
echo "[✓] Đã hoàn thành tải và lưu trữ iNaturalist 2021!"

# Dọn dẹp file nén
rm -f train_mini.tar.gz val.tar.gz

echo ""
echo "=========================================================="
echo "🎉 TẤT CẢ DỮ LIỆU ĐÃ ĐƯỢC LƯU TRỮ THÀNH CÔNG TRÊN BUCKET!"
echo "   Địa chỉ: gs://$BUCKET_NAME/"
echo "   Bạn có thể an tâm Xóa (Delete) máy ảo CPU này ngay bây giờ."
echo "=========================================================="
