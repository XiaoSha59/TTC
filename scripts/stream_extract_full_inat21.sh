#!/usr/bin/env bash
# ==============================================================================
# Stream Extract Full iNat2021 Natural Benchmark Subsets (8.8 GB) directly from AWS S3
# Pipe curl directly into GNU tar with wildcard filtering:
# - Zero raw 224GB tarball stored on disk.
# - Low CPU/IO priority (nice -n 15) to prevent any slowdown to active GPU training.
# - Captures all 6 taxonomic groups for Insects, Plants, Animals (~35,212 images).
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEST_DIR="${BASE_DIR}/data/inat21_full"
LOG_FILE="${BASE_DIR}/logs/stream_extract.log"

mkdir -p "${BASE_DIR}/logs"
mkdir -p "${DEST_DIR}"

echo "=================================================================" | tee -a "${LOG_FILE}"
echo "  Starting Stream Extraction: iNat2021 Full Natural Subsets" | tee -a "${LOG_FILE}"
echo "  Destination: ${DEST_DIR}" | tee -a "${LOG_FILE}"
echo "  Source: AWS S3 Open Data (2021/train.tar.gz - 224 GB stream)" | tee -a "${LOG_FILE}"
echo "  Time: $(date)" | tee -a "${LOG_FILE}"
echo "=================================================================" | tee -a "${LOG_FILE}"

# 1. Symlink official validation set from existing inat21 directory if present
VAL_SRC="${BASE_DIR}/data/inat21/val"
VAL_DST="${DEST_DIR}/val"
if [ -d "${VAL_SRC}" ] && [ ! -e "${VAL_DST}" ]; then
    echo ">>> Symlinking existing official val split from ${VAL_SRC} to ${VAL_DST}..." | tee -a "${LOG_FILE}"
    ln -s "${VAL_SRC}" "${VAL_DST}"
fi

# 2. Monitor extraction progress in background
monitor_progress() {
    while kill -0 "$1" 2>/dev/null; do
        sleep 60
        if [ -d "${DEST_DIR}/train" ]; then
            NUM_FILES=$(find "${DEST_DIR}/train" -type f -name "*.jpg" 2>/dev/null | wc -l || echo 0)
            DISK_SIZE=$(du -sh "${DEST_DIR}/train" 2>/dev/null | cut -f1 || echo "0")
            echo "[$(date +'%T')] Extracted images: ${NUM_FILES} | Disk used: ${DISK_SIZE}" >> "${LOG_FILE}"
        fi
    done
}

echo ">>> Launching curl | tar stream pipeline..." | tee -a "${LOG_FILE}"

# 3. Stream curl directly to tar with wildcard filters
# Note: nice -n 15 ensures background extraction doesn't preempt active training workers
(
    curl -sL "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.tar.gz" \
    | nice -n 15 tar -xz -C "${DEST_DIR}" --wildcards \
        'train/*_Animalia_Arthropoda_Insecta_Hymenoptera_Apidae*' \
        'train/*_Animalia_Arthropoda_Insecta_Hymenoptera_Vespidae*' \
        'train/*_Plantae_Tracheophyta_Magnoliopsida_Fagales_Fagaceae_Quercus*' \
        'train/*_Plantae_Tracheophyta_Magnoliopsida_Saxifragales*' \
        'train/*_Animalia_Chordata_Mammalia_Artiodactyla*' \
        'train/*_Animalia_Chordata_Mammalia_Carnivora*'
) &
TAR_PID=$!

# Start progress logger
monitor_progress "${TAR_PID}" &
MONITOR_PID=$!

# Wait for extraction to complete
wait "${TAR_PID}"
kill "${MONITOR_PID}" 2>/dev/null || true

echo "=================================================================" | tee -a "${LOG_FILE}"
echo ">>> Stream extraction completed successfully at $(date)!" | tee -a "${LOG_FILE}"
echo "=================================================================" | tee -a "${LOG_FILE}"

# 4. Count and verify image counts against paper Supplementary Table S1
if [ -d "${DEST_DIR}/train" ]; then
    INSECTS_COUNT=$(find "${DEST_DIR}/train" -type f -name "*.jpg" | grep -E 'Apidae|Vespidae' | wc -l || echo 0)
    PLANTS_COUNT=$(find "${DEST_DIR}/train" -type f -name "*.jpg" | grep -E 'Quercus|Saxifragales' | wc -l || echo 0)
    ANIMALS_COUNT=$(find "${DEST_DIR}/train" -type f -name "*.jpg" | grep -E 'Artiodactyla|Carnivora' | wc -l || echo 0)
    TOTAL_COUNT=$(find "${DEST_DIR}/train" -type f -name "*.jpg" | wc -l || echo 0)
    TOTAL_SIZE=$(du -sh "${DEST_DIR}/train" | cut -f1)

    echo "📊 VERIFICATION SUMMARY:" | tee -a "${LOG_FILE}"
    echo "  - Insects (Apidae + Vespidae): ${INSECTS_COUNT} images (Expected Table S1: ~9,438)" | tee -a "${LOG_FILE}"
    echo "  - Plants (Quercus + Saxifragales): ${PLANTS_COUNT} images (Expected Table S1: ~11,197)" | tee -a "${LOG_FILE}"
    echo "  - Animals (Artiodactyla + Carnivora): ${ANIMALS_COUNT} images (Expected Table S1: ~14,577)" | tee -a "${LOG_FILE}"
    echo "  - Total full dataset images: ${TOTAL_COUNT} (~35,212)" | tee -a "${LOG_FILE}"
    echo "  - Total disk space consumed: ${TOTAL_SIZE} (~8.8 GB)" | tee -a "${LOG_FILE}"
fi
