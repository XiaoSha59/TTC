#!/bin/bash
# ==============================================================================
# Script tự động chạy trọn gói Insects 95:5 SupPrototypes (Stage 1 + Stage 2)
# ==============================================================================

set -e
cd /home/tnpdung_79/TTC
git pull
source .venv/bin/activate

export INAT21_DATA_PATH="data/inat21"
export TMPDIR="${HOME}/tmp"
mkdir -p "$TMPDIR"

echo "=================================================================="
echo ">>> GIAI ĐOẠN 1 (Lớp 1): PRE-TRAINING 350 EPOCHS SUP-PROTOTYPES"
echo "=================================================================="

RUN_NAME="insects-95_5-supproto-fixed-350ep"

python train.py \
    experiment=contrastive_sup_prototype \
    experiment/specs=insects \
    class_ratios=[0.05,0.95] \
    batch_size=256 \
    trainer.max_epochs=350 \
    module.lr=0.0625 \
    trainer.precision=bf16-mixed \
    data_module.persistent_workers=True \
    trainer.check_val_every_n_epoch=5 \
    name="$RUN_NAME"

# Tìm checkpoint vừa train xong
LATEST_CKPT=$(ls -td logs/train/runs/*/checkpoints/last.ckpt 2>/dev/null | head -n 1)

if [ -z "$LATEST_CKPT" ] || [ ! -f "$LATEST_CKPT" ]; then
    echo "Lỗi: Không tìm thấy checkpoint last.ckpt!"
    exit 1
fi

echo "=================================================================="
echo ">>> Checkpoint đã lưu thành công tại: $LATEST_CKPT"
echo "=================================================================="

echo "=================================================================="
echo ">>> GIAI ĐOẠN 2 (Lớp 2): LINEAR PROBING EVALUATION (50 EPOCHS)"
echo "=================================================================="

python train.py \
    experiment=finetune \
    experiment/specs=insects \
    ckpt_path="$LATEST_CKPT" \
    trainer.max_epochs=50 \
    module.optimizer_name=adam \
    module.lr=0.0003 \
    name="insects-95_5-supproto-fixed-probe"

echo "=================================================================="
echo "🎉 HOÀN THÀNH TOÀN BỘ 2 GIAI ĐOẠN CHO INSECTS 95:5 SUP-PROTOTYPES!"
echo "=================================================================="
