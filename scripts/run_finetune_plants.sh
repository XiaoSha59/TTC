#!/bin/bash
# ==============================================================================
# Run Official Linear Probing (Fine-tune) for Plants (Table 1)
# Uses 100% original author's train.py and Hydra experiment=finetune protocol
# ==============================================================================

set -e
source .venv/bin/activate

echo "=========================================================="
echo " Starting Official Paper Fine-tune Probing: PLANTS"
echo " Protocol: 50 Epochs, SGD, lr=3e-4, Frozen Backbone"
echo "=========================================================="

export INAT21_DATA_PATH="data/inat21"

# Helper python script to find checkpoint path for a given run name
get_ckpt_path() {
    python3 -c "
import os, glob, yaml, sys
target_name = sys.argv[1].lower().strip()
found = None
for cfg_file in glob.glob('logs/**/.hydra/config.yaml', recursive=True):
    try:
        with open(cfg_file, 'r') as f:
            cfg = yaml.safe_load(f)
        if cfg.get('name', '').lower().strip() == target_name:
            run_dir = os.path.dirname(os.path.dirname(cfg_file))
            ckpt_dir = os.path.join(run_dir, 'checkpoints')
            last_ckpt = os.path.join(ckpt_dir, 'last.ckpt')
            if os.path.exists(last_ckpt):
                found = last_ckpt
                break
            ckpts = glob.glob(os.path.join(ckpt_dir, '*.ckpt'))
            if ckpts:
                ckpts.sort(key=os.path.getmtime, reverse=True)
                found = ckpts[0]
                break
    except:
        pass
if not found:
    for c in glob.glob('logs/**/*.ckpt', recursive=True):
        if target_name in c.lower():
            found = c
            break
if found:
    print(found)
" "$1"
}

MODELS=(
    "plants-50_50-supproto"
    "plants-50_50-supmin"
    "plants-50_50-supcon"
    "plants-95_5-supproto"
    "plants-95_5-supmin"
    "plants-95_5-supcon"
    "plants-99_1-supproto"
    "plants-99_1-supmin"
    "plants-99_1-supcon"
)

TOTAL=${#MODELS[@]}
COUNT=0

for RUN_NAME in "${MODELS[@]}"; do
    COUNT=$((COUNT + 1))
    echo ""
    echo "=========================================================="
    echo "[$COUNT/$TOTAL] Fine-tuning model: $RUN_NAME"
    echo "=========================================================="
    
    ALREADY_DONE=$(python3 -c "
import os, glob, yaml, sys
target_name = sys.argv[1].lower().strip()
found = False
for cfg_file in glob.glob('logs/**/.hydra/config.yaml', recursive=True):
    try:
        with open(cfg_file, 'r') as f:
            cfg = yaml.safe_load(f)
        if cfg.get('name', '').lower().strip() == target_name:
            run_dir = os.path.dirname(os.path.dirname(cfg_file))
            if os.path.exists(os.path.join(run_dir, 'checkpoints', 'last.ckpt')) or glob.glob(os.path.join(run_dir, 'checkpoints', '*.ckpt')):
                found = True
                break
    except:
        pass
print('YES' if found else 'NO')
" "finetune-$RUN_NAME")

    if [ "$ALREADY_DONE" = "YES" ]; then
        echo "⏩ Run finetune-$RUN_NAME already completed with checkpoint, skipping..."
        continue
    fi
    
    CKPT_PATH=$(get_ckpt_path "$RUN_NAME")
    
    if [ -z "$CKPT_PATH" ] || [ ! -f "$CKPT_PATH" ]; then
        echo "⚠️ Checkpoint not found for $RUN_NAME, skipping..."
        continue
    fi
    
    echo ">>> Found checkpoint: $CKPT_PATH"
    
    # Run 100% original author's train.py with experiment=finetune
    python train.py experiment=finetune experiment/specs=plants \
        +base_model_path="$CKPT_PATH" \
        trainer.max_epochs=50 \
        module.optimizer_name=adam \
        module.lr=0.001 \
        data.data_module.persistent_workers=True \
        name="finetune-$RUN_NAME" \
        trainer.accelerator=gpu \
        trainer.devices=1
        
    echo ">>> Completed fine-tuning for $RUN_NAME"
done

echo ""
echo "=========================================================="
echo "✅ All Plants Fine-tuning Completed!"
echo "=========================================================="
