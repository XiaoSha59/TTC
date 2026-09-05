#!/bin/bash
# ==============================================================================
# MASTER 24/7 REPRODUCTION RUNNER (ALL DATASETS - BATCH 256 - BF16)
# Fully resilient: Auto-Skip completed runs, Auto-Resume from checkpoints
# ==============================================================================

set -e
source .venv/bin/activate

export INAT21_DATA_PATH="data/inat21"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 1. Disable Ubuntu automatic reboot in background
sudo systemctl stop unattended-upgrades 2>/dev/null || true
sudo systemctl disable unattended-upgrades 2>/dev/null || true

echo "======================================================================"
echo "🛡️  STARTING MASTER 24/7 RESILIENT RUNNER (BATCH 256 + BF16)"
echo "======================================================================"

DATASETS=("insects" "animals" "plants")
RATIOS=("50_50:[0.5,0.5]" "95_5:[0.05,0.95]" "99_1:[0.01,0.99]")

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "######################################################################"
    echo ">>> PROCESSING DATASET: $DATASET"
    echo "######################################################################"
    
    for ITEM in "${RATIOS[@]}"; do
        NAME_TAG="${ITEM%%:*}"
        RATIO_VAL="${ITEM##*:}"
        
        # We run the two proposed TTC contrastive methods + SupCon baseline
        MODELS=(
            "supproto:experiment=contrastive_sup_prototype"
            "supmin:experiment=contrastive:module.ratio_supervised_majority=0.0"
            "supcon:experiment=contrastive"
        )
        
        for M_INFO in "${MODELS[@]}"; do
            M_NAME="${M_INFO%%:*}"
            M_ARGS="${M_INFO#*:}"
            
            PRETRAIN_NAME="b256-bf16-${DATASET}-${NAME_TAG}-${M_NAME}"
            FINETUNE_NAME="finetune-${PRETRAIN_NAME}"
            
            echo "------------------------------------------------------------------"
            echo "Target: $DATASET | Ratio: $NAME_TAG | Method: $M_NAME"
            echo "------------------------------------------------------------------"
            
            # --- CHECK IF LAYER 2 IS ALREADY COMPLETED ---
            DONE_L2=$(python3 -c "
import os, glob, yaml, sys
target = sys.argv[1].strip()
found = False
for cfg in glob.glob('logs/**/.hydra/config.yaml', recursive=True):
    try:
        with open(cfg) as f:
            c = yaml.safe_load(f)
        if c.get('name', '').strip() == target:
            rdir = os.path.dirname(os.path.dirname(cfg))
            if os.path.exists(os.path.join(rdir, 'checkpoints', 'last.ckpt')) or glob.glob(os.path.join(rdir, 'checkpoints', '*.ckpt')):
                found = True
                break
    except:
        pass
print('YES' if found else 'NO')
" "$FINETUNE_NAME")

            if [ "$DONE_L2" = "YES" ]; then
                echo "⏩ [$FINETUNE_NAME] Both Layer 1 and Layer 2 already completed! Skipping..."
                continue
            fi
            
            # --- CHECK LAYER 1 CHECKPOINT ---
            L1_CKPT=$(python3 -c "
import os, glob, yaml, sys
target = sys.argv[1].strip()
found = ''
for cfg in glob.glob('logs/**/.hydra/config.yaml', recursive=True):
    try:
        with open(cfg) as f:
            c = yaml.safe_load(f)
        if c.get('name', '').strip() == target:
            rdir = os.path.dirname(os.path.dirname(cfg))
            last = os.path.join(rdir, 'checkpoints', 'last.ckpt')
            if os.path.exists(last):
                found = last
                break
            ckpts = glob.glob(os.path.join(rdir, 'checkpoints', '*.ckpt'))
            if ckpts:
                ckpts.sort(key=os.path.getmtime, reverse=True)
                found = ckpts[0]
                break
    except:
        pass
print(found)
" "$PRETRAIN_NAME")

            # --- RUN LAYER 1 PRETRAINING IF NOT FINISHED ---
            if [ -z "$L1_CKPT" ] || [ ! -f "$L1_CKPT" ]; then
                echo "🚀 [LỚP 1] Launching Phase 1 Pretraining: $PRETRAIN_NAME"
                
                # Format experiment argument and extra args
                EXP_ARG=$(echo "$M_ARGS" | cut -d':' -f1)
                EXTRA_ARG=$(echo "$M_ARGS" | cut -s -d':' -f2)
                
                python train.py $EXP_ARG experiment/specs=$DATASET \
                    class_ratios=$RATIO_VAL \
                    batch_size=256 \
                    data.data_module.batch_size=256 \
                    trainer.precision=bf16-mixed \
                    module.lr=0.5 \
                    trainer.max_epochs=100 \
                    name="$PRETRAIN_NAME" \
                    trainer.accelerator=gpu \
                    trainer.devices=1 $EXTRA_ARG
                    
                # Re-locate newly saved checkpoint
                L1_CKPT=$(python3 -c "
import os, glob, yaml, sys
target = sys.argv[1].strip()
found = ''
for cfg in glob.glob('logs/**/.hydra/config.yaml', recursive=True):
    try:
        with open(cfg) as f:
            c = yaml.safe_load(f)
        if c.get('name', '').strip() == target:
            rdir = os.path.dirname(os.path.dirname(cfg))
            last = os.path.join(rdir, 'checkpoints', 'last.ckpt')
            if os.path.exists(last):
                found = last
                break
            ckpts = glob.glob(os.path.join(rdir, 'checkpoints', '*.ckpt'))
            if ckpts:
                ckpts.sort(key=os.path.getmtime, reverse=True)
                found = ckpts[0]
                break
    except:
        pass
print(found)
" "$PRETRAIN_NAME")
            else
                echo "✅ [LỚP 1] Pretraining checkpoint already exists: $L1_CKPT"
            fi
            
            # --- RUN LAYER 2 LINEAR PROBING ---
            echo "🎯 [LỚP 2] Launching Phase 2 Linear Probing: $FINETUNE_NAME"
            python train.py experiment=finetune experiment/specs=$DATASET \
                +base_model_path="$L1_CKPT" \
                trainer.max_epochs=50 \
                module.optimizer_name=sgd \
                module.lr=3e-4 \
                trainer.precision=bf16-mixed \
                name="$FINETUNE_NAME" \
                trainer.accelerator=gpu \
                trainer.devices=1
                
            echo "🎉 Successfully finished both layers for: $PRETRAIN_NAME"
            
            # --- EXPLICIT VRAM & PROCESS CLEANUP ---
            python3 -c "import torch; torch.cuda.is_available() and torch.cuda.empty_cache()" 2>/dev/null || true
            sleep 3
        done
    done
done

echo ""
echo "======================================================================"
echo "🏆 ALL EXPERIMENTS IN TABLE 1 (INSECTS, ANIMALS, PLANTS) 100% FINISHED!"
echo "======================================================================"
