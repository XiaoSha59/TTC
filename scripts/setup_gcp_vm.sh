#!/bin/bash
# ==============================================================================
# Setup Script for GCP GPU Instance using Python standard venv (No Miniconda)
# Usage: bash scripts/setup_gcp_vm.sh
# ==============================================================================

set -e

echo "=== 1. Updating System & Installing Python 3 Venv ==="
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git tmux htop

echo "=== 2. Creating Python Virtual Environment (.venv) ==="
python3 -m venv .venv
source .venv/bin/activate

echo "=== 3. Upgrading pip and installing PyTorch with CUDA ==="
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "=== 4. Installing All Project Requirements ==="
pip install -r requirements.txt

echo "=== 5. Setting up WandB ==="
if [ -n "$WANDB_API_KEY" ]; then
    wandb login "$WANDB_API_KEY"
    echo "WandB logged in successfully!"
fi

echo "=========================================================="
echo " Setup Completed! Run: source .venv/bin/activate"
echo "=========================================================="
