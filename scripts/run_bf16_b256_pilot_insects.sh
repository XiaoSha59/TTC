#!/bin/bash
# ==============================================================================
# Full Paper Replication: Insects 95:5 SupProto
# Protocol: 350 Epochs (No Early Stopping, Full Cosine Annealing to 0)
# Batch 256 (Physical) | Precision: BF16-mixed | LR: 0.0625 (Official)
# Phase 2: Official Linear Probing (50 epochs, SGD, lr=3e-4)
# ==============================================================================

set -e
source .venv/bin/activate

export INAT21_DATA_PATH="data/inat21"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

NAME="insects-95_5-supproto-b256-350ep"
FINETUNE_NAME="finetune-${NAME}"

echo "======================================================================"
echo "🚀 [STAGE 1/2] Full Contrastive Pre-training (Exact 350 Epochs)"
echo "Specs: Insects 95:5 | Loss: SupPrototypes | Batch: 256 | Precision: BF16"
echo "Learning Rate: 0.0625 (SGD + Cosine Warmup 10ep -> Full 350ep decay)"
echo "======================================================================"

python train.py experiment=contrastive_sup_prototype experiment/specs=insects \
    class_ratios=[0.05,0.95] \
    batch_size=256 \
    data.data_module.batch_size=256 \
    trainer.precision=bf16-mixed \
    module.lr=0.0625 \
    trainer.max_epochs=350 \
    name="$NAME" \
    trainer.accelerator=gpu \
    trainer.devices=1

echo "======================================================================"
echo "✅ [STAGE 1/2] Pretraining Finished! Locating Checkpoint..."
echo "======================================================================"

CKPT_PATH=$(python3 -c "
import os, glob, yaml
found = None
for cfg_file in glob.glob('logs/**/.hydra/config.yaml', recursive=True):
    try:
        with open(cfg_file, 'r') as f:
            cfg = yaml.safe_load(f)
        if cfg.get('name', '').strip() == '$NAME':
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
print(found or '')
")

if [ -z "$CKPT_PATH" ] || [ ! -f "$CKPT_PATH" ]; then
    echo "❌ Error: Could not find checkpoint for $NAME"
    exit 1
fi

echo ">>> Found Checkpoint: $CKPT_PATH"
echo ""

# Explicit VRAM cleanup before Phase 2
python3 -c "import torch; torch.cuda.is_available() and torch.cuda.empty_cache()" 2>/dev/null || true
sleep 3

echo "======================================================================"
echo "🎯 [STAGE 2/2] Official Linear Probing: $FINETUNE_NAME"
echo "Protocol: 50 Epochs SGD, lr=3e-4, Frozen Backbone on 1% Balanced Train Set"
echo "======================================================================"

python train.py experiment=finetune experiment/specs=insects \
    +base_model_path="$CKPT_PATH" \
    trainer.max_epochs=50 \
    module.optimizer_name=sgd \
    module.lr=3e-4 \
    trainer.precision=bf16-mixed \
    name="$FINETUNE_NAME" \
    trainer.accelerator=gpu \
    trainer.devices=1

# Explicit VRAM cleanup after Phase 2
python3 -c "import torch; torch.cuda.is_available() and torch.cuda.empty_cache()" 2>/dev/null || true

echo "======================================================================"
echo "🎉 ALL 2 STAGES COMPLETED SUCCESSFULLY FOR $NAME!"
echo "======================================================================"
