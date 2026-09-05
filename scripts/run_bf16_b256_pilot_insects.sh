#!/bin/bash
# ==============================================================================
# Pilot Experiment: BF16 + Batch Size 256 on Insects 95:5 SupProto
# End-to-End 2-Layer Protocol (Phase 1 Pretrain + Phase 2 Linear Probing)
# ==============================================================================

set -e
source .venv/bin/activate

export INAT21_DATA_PATH="data/inat21"

NAME="bf16-b256-insects-95_5-supproto"
FINETUNE_NAME="finetune-${NAME}"

echo "=========================================================="
echo "🚀 [LỚP 1] Starting Phase 1 Pretraining: $NAME"
echo "Specs: Insects 95:5 | Loss: SupPrototypes | Batch: 256 | Precision: BF16"
echo "=========================================================="

python train.py experiment=contrastive_sup_prototype experiment/specs=insects \
    class_ratios=[0.05,0.95] \
    batch_size=256 \
    data.data_module.batch_size=256 \
    trainer.precision=bf16-mixed \
    module.lr=0.5 \
    trainer.max_epochs=100 \
    name="$NAME" \
    trainer.accelerator=gpu \
    trainer.devices=1

echo "=========================================================="
echo "✅ [LỚP 1] Pretraining Finished! Locating Checkpoint..."
echo "=========================================================="

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
echo "=========================================================="
echo "🎯 [LỚP 2] Starting Phase 2 Official Linear Probing: $FINETUNE_NAME"
echo "Protocol: 50 Epochs SGD, lr=3e-4, Frozen Backbone on 1% Balanced Train Set"
echo "=========================================================="

python train.py experiment=finetune experiment/specs=insects \
    +base_model_path="$CKPT_PATH" \
    trainer.max_epochs=50 \
    module.optimizer_name=sgd \
    module.lr=3e-4 \
    trainer.precision=bf16-mixed \
    name="$FINETUNE_NAME" \
    trainer.accelerator=gpu \
    trainer.devices=1

echo "=========================================================="
echo "🎉 ALL 2 LAYERS COMPLETED FOR $NAME!"
echo "=========================================================="
