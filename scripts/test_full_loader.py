import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.iNatData import INaturalistNClasses
from data.data_module import create_subsampled_dataset
import torch
from torch.utils.data import random_split

print(">>> Testing INaturalistNClasses + create_subsampled_dataset with data/inat21_full...")
classes = [
    "Animalia_Arthropoda_Insecta_Hymenoptera_Apidae",
    "Animalia_Arthropoda_Insecta_Hymenoptera_Vespidae"
]

data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "inat21_full")

total_dataset = INaturalistNClasses(
    root=data_root,
    split="train",
    classes=classes
)
print(f"Total raw matching images in data/inat21_full/train: {len(total_dataset)}")

generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(total_dataset, [0.95, 0.05], generator=generator)

for name, ratios in [("50_50", [0.5, 0.5]), ("95_5", [0.05, 0.95]), ("99_1", [0.01, 0.99])]:
    sub, counts = create_subsampled_dataset(train_dataset, ratios, is_test_val=False)
    print(f"[{name}] Subsampled Dataset Size: {len(sub)} | Class Counts: {counts}")
