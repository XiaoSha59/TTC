#!/bin/bash
# ==============================================================================
# Helper to download and verify iNaturalist 2021 Validation Dataset
# ==============================================================================
set -e

mkdir -p data/inat21

# Clean up broken nested files
rm -f data/inat21/inat21/val.tar.gz 2>/dev/null || true

# Test existing archive
if [ -f "data/inat21/val.tar.gz" ]; then
    echo "Testing existing data/inat21/val.tar.gz..."
    if ! tar -tf "data/inat21/val.tar.gz" >/dev/null 2>&1; then
        echo "⚠️ data/inat21/val.tar.gz is truncated or corrupted. Deleting..."
        rm -f "data/inat21/val.tar.gz"
    fi
fi

# Download if not present and val folder not extracted
if [ ! -d "data/inat21/val" ] || [ -z "$(ls -A data/inat21/val 2>/dev/null)" ]; then
    if [ ! -f "data/inat21/val.tar.gz" ]; then
        echo ">>> Downloading val.tar.gz from GCS Bucket..."
        if ! gcloud storage cp gs://ttc-paper-datasets-2025/inat21/val.tar.gz data/inat21/val.tar.gz; then
            echo ">>> GCS download failed, downloading from official AWS mirror..."
            wget -c https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz -O data/inat21/val.tar.gz
        fi
    fi

    echo ">>> Extracting val.tar.gz to data/inat21/ ..."
    tar -xzf data/inat21/val.tar.gz -C data/inat21/
fi

# Flatten if double-nested
if [ -d "data/inat21/val/val" ]; then
    echo "Flattening nested data/inat21/val/val..."
    mv data/inat21/val/val/* data/inat21/val/ 2>/dev/null || true
    rmdir data/inat21/val/val 2>/dev/null || true
fi

echo "=========================================================="
echo "✅ iNaturalist 2021 validation set is ready!"
echo "Sample count in val:" $(ls -d data/inat21/val/*/ 2>/dev/null | wc -l) "categories"
echo "=========================================================="
