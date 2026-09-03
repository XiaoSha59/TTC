#!/bin/bash
# ==============================================================================
# Run iNaturalist Plants Experiments (Table 1 in Paper)
# Tests across 3 imbalance ratios: 50%-50%, 95%-5%, 99%-1%
# ==============================================================================

set -e
source .venv/bin/activate

echo "=========================================================="
echo " Starting iNat21 Plants Experiments (Table 1)"
echo "=========================================================="

RATIOS=("50_50:[0.5,0.5]" "95_5:[0.05,0.95]" "99_1:[0.01,0.99]")

for ITEM in "${RATIOS[@]}"; do
    NAME_TAG="${ITEM%%:*}"
    RATIO_VAL="${ITEM##*:}"
    
    echo ">>> Running Plants Ratio: $NAME_TAG ($RATIO_VAL)"

    # 1. Supervised Minority (Ours)
    python train.py experiment=contrastive experiment/specs=plants class_ratios=$RATIO_VAL module.ratio_supervised_majority=0.0 name="plants-$NAME_TAG-supmin"

    # 2. Supervised Prototypes (Ours)
    python train.py experiment=contrastive_sup_prototype experiment/specs=plants class_ratios=$RATIO_VAL name="plants-$NAME_TAG-supproto"

    # 3. Standard SupCon
    python train.py experiment=contrastive experiment/specs=plants class_ratios=$RATIO_VAL module.ratio_supervised_majority=1.0 name="plants-$NAME_TAG-supcon"

    # 4. Weighted Cross-Entropy
    python train.py experiment=weighted_ce experiment/specs=plants class_ratios=$RATIO_VAL name="plants-$NAME_TAG-weightedce"
done

echo "=========================================================="
echo " Plants Experiments Completed!"
echo "=========================================================="
