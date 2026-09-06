#!/bin/bash
set -e
source /home/tnpdung_79/TTC/.venv/bin/activate
cd /home/tnpdung_79/TTC

CKPT="/home/tnpdung_79/TTC/logs/train/runs/2026-09-05_16-58-10/checkpoints/last.ckpt"

python train.py experiment=finetune experiment/specs=insects \
    +base_model_path="$CKPT" \
    trainer.max_epochs=50 \
    module.optimizer_name=adam \
    module.lr=3e-4 \
    trainer.precision=bf16-mixed \
    name="finetune-insects-adam-3e4" \
    trainer.accelerator=gpu \
    trainer.devices=1
