import os
import sys

# Ensure root dir in PYTHONPATH
sys.path.append(os.getcwd())

from data.iNatData import INaturalistNClasses
from torch.utils.data import DataLoader

SPECS = {
    "plants": [
        "Plantae_Tracheophyta_Magnoliopsida_Fagales_Fagaceae_Quercus",
        "Plantae_Tracheophyta_Magnoliopsida_Saxifragales"
    ],
    "insects": [
        "Animalia_Arthropoda_Insecta_Hymenoptera_Apidae",
        "Animalia_Arthropoda_Insecta_Hymenoptera_Vespidae"
    ],
    "animals": [
        "Animalia_Chordata_Mammalia_Artiodactyla",
        "Animalia_Chordata_Mammalia_Carnivora"
    ]
}

print("==========================================================")
print("🔍 DIAGNOSTIC & VERIFICATION FOR iNATURALIST 2021 DATASETS")
print("==========================================================")

all_passed = True

for dataset_name, classes in SPECS.items():
    print(f"\n--- Testing Dataset: {dataset_name.upper()} ---")
    for split in ["train", "val"]:
        try:
            ds = INaturalistNClasses(root="data/inat21", split=split, classes=classes)
            if len(ds) == 0:
                print(f"❌ FAIL: {dataset_name} [{split}] loaded 0 samples!")
                all_passed = False
                continue
            
            # Test actual item fetching
            img, label = ds[0]
            print(f"✅ PASS: {dataset_name} [{split}] -> Total: {len(ds)} samples, First sample size: {img.size}, Label: {label}")
        except Exception as e:
            print(f"❌ ERROR on {dataset_name} [{split}]: {e}")
            all_passed = False

print("\n==========================================================")
if all_passed:
    print("🎉 ALL iNATURALIST DATASETS & SPLITS VERIFIED SUCCESSFULLY!")
else:
    print("⚠️ SOME DATASETS FAILED VERIFICATION. CHECK PATHS ABOVE.")
print("==========================================================")
