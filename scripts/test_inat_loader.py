import os
from data.iNatData import INaturalistNClasses

print("Testing iNaturalist data loader...")
try:
    train_ds = INaturalistNClasses(root="data/inat21", split="train", classes=["tracheophyta"])
    print(f"✅ TRAIN SET SUCCESS: {len(train_ds)} samples loaded!")
    val_ds = INaturalistNClasses(root="data/inat21", split="val", classes=["tracheophyta"])
    print(f"✅ VAL SET SUCCESS: {len(val_ds)} samples loaded!")
except Exception as e:
    print(f"❌ ERROR: {e}")
