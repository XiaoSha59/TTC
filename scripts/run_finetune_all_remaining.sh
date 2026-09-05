#!/bin/bash
# ==============================================================================
# Master Runner: Fine-tune Plants and Animals Sequentially
# Completes Table 1 (Natural Benchmarks) 100% via Official SGD 50 Epochs Protocol
# ==============================================================================

set -e

echo "=========================================================="
echo " 🌿 STAGE 1: FINE-TUNING PLANTS (9 MODELS)"
echo "=========================================================="
bash scripts/run_finetune_plants.sh

echo ""
echo "=========================================================="
echo " 🐾 STAGE 2: FINE-TUNING ANIMALS (9 MODELS)"
echo "=========================================================="
bash scripts/run_finetune_animals.sh

echo ""
echo "=========================================================="
echo " 🎉 ALL TABLE 1 BENCHMARKS (INSECTS, PLANTS, ANIMALS) 100% COMPLETED!"
echo " Generating Final Paper Tables..."
python scripts/generate_official_paper_tables.py || true
echo "=========================================================="
