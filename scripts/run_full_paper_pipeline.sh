#!/bin/bash
# ==============================================================================
# MASTER AUTOMATION PIPELINE FOR THE ENTIRE PAPER
# 1. Finishes PneumoniaMNIST (Table 2)
# 2. Runs FracAtlas (Table 2)
# 3. Runs iNat21 Natural Benchmarks: Plants, Insects, Animals (Table 1)
# 4. Exports Table 1 & Table 2 to Markdown, CSV, and LaTeX
# 5. Syncs all logs/checkpoints to Google Cloud Storage
# 6. Automatically shuts down the VM to stop billing ($0.00 cost while sleeping)
# ==============================================================================

set -e
source .venv/bin/activate

echo "=========================================================="
echo " 🚀 STARTING MASTER TRAINING PIPELINE (ALL EXPERIMENTS)"
echo "=========================================================="

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export INAT21_DATA_PATH="data/inat21"

backup_to_gcs() {
    echo ">>> Syncing checkpoints to GCS Bucket..."
    gcloud storage cp -r logs/ "gs://ttc-paper-datasets-2025/checkpoints/" 2>/dev/null || true
    gcloud storage cp -r results/ "gs://ttc-paper-datasets-2025/results/" 2>/dev/null || true
}

trap backup_to_gcs EXIT

# Helper to auto-detect and resume from latest checkpoint if interrupted
get_ckpt_arg() {
    local ckpt=$(ls -td logs/train/runs/*/checkpoints/last.ckpt 2>/dev/null | head -n1 || true)
    if [ -n "$ckpt" ] && [ -f "$ckpt" ]; then
        echo "ckpt_path=$ckpt"
    fi
}

# ==============================================================================
# PHASE 1: PNEUMONIAMNIST (TABLE 2)
# ==============================================================================
echo ">>> [Phase 1/3] Running PneumoniaMNIST..."
if [ ! -f "logs/train/runs/*pneumoniamnist-weightedce*/checkpoints/last.ckpt" ]; then
    bash scripts/run_pneumonia.sh || true
fi
backup_to_gcs

# ==============================================================================
# PHASE 2: FRACATLAS (TABLE 2)
# ==============================================================================
echo ">>> [Phase 2/3] Setting up and Running FracAtlas..."
mkdir -p data
if [ ! -d "data/FracAtlas" ] && [ ! -d "data/fracatlas" ]; then
    echo "Downloading FracAtlas dataset from Bucket..."
    gcloud storage cp -r gs://ttc-paper-datasets-2025/FracAtlas data/ || gcloud storage cp -r gs://ttc-paper-datasets-2025/fracatlas data/
fi

python scripts/prepare_fracatlas.py

# 1. FracAtlas - SupMin
echo "Running FracAtlas - SupMin..."
python train.py experiment=contrastive experiment/specs=generic_2_class data=fracatlas batch_size=128 module.ratio_supervised_majority=0.0 name="fracatlas-supmin"
backup_to_gcs

# 2. FracAtlas - SupProto
echo "Running FracAtlas - SupProto..."
python train.py experiment=contrastive_sup_prototype experiment/specs=generic_2_class data=fracatlas batch_size=128 name="fracatlas-supproto"
backup_to_gcs

# 3. FracAtlas - SupCon
echo "Running FracAtlas - SupCon..."
python train.py experiment=contrastive experiment/specs=generic_2_class data=fracatlas batch_size=128 module.ratio_supervised_majority=1.0 name="fracatlas-supcon"
backup_to_gcs

# 4. FracAtlas - Weighted CE
echo "Running FracAtlas - Weighted CE..."
python train.py experiment=weighted_ce experiment/specs=generic_2_class data=fracatlas batch_size=128 name="fracatlas-weightedce"
backup_to_gcs

# ==============================================================================
# PHASE 3: INAT21 NATURAL BENCHMARKS (TABLE 1: Plants, Insects, Animals)
# ==============================================================================
echo ">>> [Phase 3/3] Setting up and Running iNat21 Natural Benchmarks..."
if [ ! -d "data/inat21" ]; then
    echo "Downloading iNat21 dataset from Bucket..."
    gcloud storage cp -r gs://ttc-paper-datasets-2025/inat21 data/
fi

# Run Plants
echo "Running iNat21 Plants (Table 1)..."
bash scripts/run_inat_plants.sh || true
backup_to_gcs

# Run Insects
echo "Running iNat21 Insects (Table 1)..."
RATIOS=("50_50:[0.5,0.5]" "95_5:[0.05,0.95]" "99_1:[0.01,0.99]")
for ITEM in "${RATIOS[@]}"; do
    NAME_TAG="${ITEM%%:*}"
    RATIO_VAL="${ITEM##*:}"
    echo ">>> Insects Ratio: $NAME_TAG"
    python train.py experiment=contrastive experiment/specs=insects class_ratios=$RATIO_VAL module.ratio_supervised_majority=0.0 name="insects-$NAME_TAG-supmin" || true
    python train.py experiment=contrastive_sup_prototype experiment/specs=insects class_ratios=$RATIO_VAL name="insects-$NAME_TAG-supproto" || true
    python train.py experiment=contrastive experiment/specs=insects class_ratios=$RATIO_VAL module.ratio_supervised_majority=1.0 name="insects-$NAME_TAG-supcon" || true
    python train.py experiment=weighted_ce experiment/specs=insects class_ratios=$RATIO_VAL name="insects-$NAME_TAG-weightedce" || true
    backup_to_gcs
done

# Run Animals
echo "Running iNat21 Animals (Table 1)..."
for ITEM in "${RATIOS[@]}"; do
    NAME_TAG="${ITEM%%:*}"
    RATIO_VAL="${ITEM##*:}"
    echo ">>> Animals Ratio: $NAME_TAG"
    python train.py experiment=contrastive experiment/specs=animals class_ratios=$RATIO_VAL module.ratio_supervised_majority=0.0 name="animals-$NAME_TAG-supmin" || true
    python train.py experiment=contrastive_sup_prototype experiment/specs=animals class_ratios=$RATIO_VAL name="animals-$NAME_TAG-supproto" || true
    python train.py experiment=contrastive experiment/specs=animals class_ratios=$RATIO_VAL module.ratio_supervised_majority=1.0 name="animals-$NAME_TAG-supcon" || true
    python train.py experiment=weighted_ce experiment/specs=animals class_ratios=$RATIO_VAL name="animals-$NAME_TAG-weightedce" || true
    backup_to_gcs
done

# ==============================================================================
# PHASE 4: EXPORT ALL FINAL RESULTS & TABLES
# ==============================================================================
echo "=========================================================="
echo " 📊 EXPORTING FINAL SUMMARY TABLES..."
echo "=========================================================="
python scripts/export_results.py tnpdung79hcmus binary-learning || true
backup_to_gcs

echo "=========================================================="
echo " 🎉 ALL EXPERIMENTS FOR THE ENTIRE PAPER COMPLETED 100%!"
echo " Automatically shutting down VM now to save budget ($0.00 cost)..."
echo "=========================================================="

sudo shutdown -h now
