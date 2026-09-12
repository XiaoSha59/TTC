import os
import sys
from kaggle.api.kaggle_api_extended import KaggleApi

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# 1. Inspect Account 1
os.environ['KAGGLE_USERNAME'] = 'salala1706'
os.environ['KAGGLE_KEY'] = 'KGAT_c50dd809cbcb96fb725040dee59239f5'
if 'KAGGLE_API_TOKEN' in os.environ: del os.environ['KAGGLE_API_TOKEN']
api1 = KaggleApi()
api1.authenticate()

print("\n--- Account 1: salala1706 ---")
for k in ["ttc-insects-weighted-ce-all-ratios-full", "ttc-insects-supcon-all-ratios-full", "ttc-insects-supcon-50-50"]:
    ref = f"salala1706/{k}"
    try:
        s = api1.kernels_status(ref)
        print(f"• {ref:<50} | Status: {s.status}")
    except Exception as e:
        print(f"• {ref:<50} | Status: Not created / {e}")

# 2. Inspect Account 2
if 'KAGGLE_USERNAME' in os.environ: del os.environ['KAGGLE_USERNAME']
if 'KAGGLE_KEY' in os.environ: del os.environ['KAGGLE_KEY']
os.environ['KAGGLE_API_TOKEN'] = 'KGAT_d8bfa22b3323e9d1a32a52737cc6a634'
api2 = KaggleApi()
api2.authenticate()

print("\n--- Account 2: ngocuyennhitran ---")
for k in ["ttc-insects-supcon-all-ratios-full", "ttc-insects-supcon-95-5-full", "ttc-insects-supcon-99-1"]:
    ref = f"ngocuyennhitran/{k}"
    try:
        s = api2.kernels_status(ref)
        print(f"• {ref:<50} | Status: {s.status}")
    except Exception as e:
        print(f"• {ref:<50} | Status: Not created / {e}")
