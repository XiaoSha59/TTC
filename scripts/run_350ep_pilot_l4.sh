#!/bin/bash
# ==============================================================================
# PILOT EXPERIMENT ON 1x NVIDIA L4 (24GB VRAM)
# Target: insects-99_1-supmin (350 Epochs, Effective Batch Size 256 via Accumulation=2)
# Runs directly on existing VM with ZERO setup, ZERO quota issues!
# ==============================================================================

set -e
source .venv/bin/activate

echo "======================================================================"
echo " 🚀 STARTING 350-EPOCH PILOT TEST ON 1x L4: INSECTS 99:1 SUPMIN"
echo " Protocol: 350 Epochs | Effective Batch Size: 256 (128 x Accumulation=2)"
echo " GPU Check:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo "======================================================================"

export INAT21_DATA_PATH="data/inat21"

PRETRAIN_RUN_NAME="insects-99_1-supmin-350ep-accum"
FINETUNE_RUN_NAME="finetune-insects-99_1-supmin-350ep-accum"

START_TIME=$(date +%s)

echo ""
echo ">>> [STAGE 1/2] Contrastive Pre-training (350 Epochs, Effective Batch 256)..."
echo ">>> Run Name: $PRETRAIN_RUN_NAME"

python train.py experiment=contrastive \
    experiment/specs=insects \
    class_ratios=[0.01,0.99] \
    module.ratio_supervised_majority=0.0 \
    trainer.max_epochs=350 \
    batch_size=128 \
    trainer.accumulate_grad_batches=2 \
    name="$PRETRAIN_RUN_NAME" \
    trainer.accelerator=gpu \
    trainer.devices=1

echo ""
echo ">>> Pre-training completed! Locating checkpoint..."

CKPT_PATH=$(python3 -c "
import os, glob, yaml
found = None
for cfg_file in glob.glob('logs/**/.hydra/config.yaml', recursive=True):
    try:
        with open(cfg_file, 'r') as f:
            cfg = yaml.safe_load(f)
        if cfg.get('name', '').strip() == '$PRETRAIN_RUN_NAME':
            run_dir = os.path.dirname(os.path.dirname(cfg_file))
            last_ckpt = os.path.join(run_dir, 'checkpoints', 'last.ckpt')
            if os.path.exists(last_ckpt):
                found = last_ckpt
                break
            ckpts = glob.glob(os.path.join(run_dir, 'checkpoints', '*.ckpt'))
            if ckpts:
                ckpts.sort(key=os.path.getmtime, reverse=True)
                found = ckpts[0]
                break
    except:
        pass
if not found:
    ckpts = glob.glob('logs/**/last.ckpt', recursive=True)
    if ckpts:
        ckpts.sort(key=os.path.getmtime, reverse=True)
        found = ckpts[0]
print(found or '')
")

if [ -z "$CKPT_PATH" ] || [ ! -f "$CKPT_PATH" ]; then
    echo "❌ Error: Could not find checkpoint for $PRETRAIN_RUN_NAME!"
    exit 1
fi

echo ">>> Found Checkpoint: $CKPT_PATH"

echo ""
echo ">>> [STAGE 2/2] Official Linear Probing (50 Epochs, SGD, lr=3e-4)..."
echo ">>> Run Name: $FINETUNE_RUN_NAME"

python train.py experiment=finetune \
    experiment/specs=insects \
    +base_model_path="$CKPT_PATH" \
    trainer.max_epochs=50 \
    module.optimizer_name=sgd \
    module.lr=3e-4 \
    name="$FINETUNE_RUN_NAME" \
    trainer.accelerator=gpu \
    trainer.devices=1

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo ""
echo "======================================================================"
echo " 🎉 350-EPOCH PILOT TEST COMPLETED SUCCESSFULLY!"
echo " Total Elapsed Time: ${HOURS}h ${MINUTES}m"
echo " Pre-train Run: $PRETRAIN_RUN_NAME"
echo " Fine-tune Run: $FINETUNE_RUN_NAME"
echo "======================================================================"
