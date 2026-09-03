#!/bin/bash
# ==============================================================================
# Script: Download Datasets directly on GCP VM and Upload to GCS Bucket
# Usage: bash scripts/download_and_upload_to_gcs.sh <YOUR_BUCKET_NAME>
# Example: bash scripts/download_and_upload_to_gcs.sh ttc-paper-datasets-2025
# ==============================================================================

set -e

BUCKET_NAME=$1

if [ -z "$BUCKET_NAME" ]; then
    echo "Error: Please provide your GCS Bucket name!"
    echo "Usage: bash scripts/download_and_upload_to_gcs.sh <YOUR_BUCKET_NAME>"
    exit 1
fi

echo "=========================================================="
echo " Starting Dataset Download & Upload to gs://$BUCKET_NAME"
echo "=========================================================="

DATA_DIR="$HOME/datasets"
mkdir -p "$DATA_DIR"

# ------------------------------------------------------------------------------
# 1. Download & Upload FracAtlas Dataset
# ------------------------------------------------------------------------------
echo ">>> [1/2] Processing FracAtlas Dataset..."
mkdir -p "$DATA_DIR/fracatlas"
cd "$DATA_DIR/fracatlas"

if [ ! -f "FracAtlas.zip" ]; then
    echo "Downloading FracAtlas from Figshare..."
    wget -c -O FracAtlas.zip "https://figshare.com/ndownloader/files/41088458"
fi

if [ ! -d "FracAtlas" ]; then
    echo "Extracting FracAtlas.zip..."
    unzip -q -o FracAtlas.zip
fi

echo "Uploading FracAtlas to GCS Bucket..."
gcloud storage cp -r "$DATA_DIR/fracatlas" "gs://$BUCKET_NAME/fracatlas"
echo "FracAtlas uploaded successfully!"

# ------------------------------------------------------------------------------
# 2. Download & Upload iNaturalist 2021 Mini Dataset
# ------------------------------------------------------------------------------
echo ">>> [2/2] Processing iNaturalist 2021 Dataset..."
mkdir -p "$DATA_DIR/inat21"
cd "$DATA_DIR/inat21"

if [ ! -f "train_mini.tar.gz" ]; then
    echo "Downloading iNat21 train_mini.tar.gz from AWS..."
    wget -c "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.tar.gz"
fi

if [ ! -f "train_mini.json.tar.gz" ]; then
    echo "Downloading iNat21 train_mini.json.tar.gz..."
    wget -c "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.json.tar.gz"
fi

echo "Extracting iNat21 archives..."
tar -xzf train_mini.tar.gz
tar -xzf train_mini.json.tar.gz

echo "Uploading iNat21 to GCS Bucket..."
gcloud storage cp -r "$DATA_DIR/inat21" "gs://$BUCKET_NAME/inat21"
echo "iNat21 uploaded successfully!"

echo "=========================================================="
echo " All datasets have been downloaded and uploaded to:"
echo " gs://$BUCKET_NAME/"
echo "=========================================================="
