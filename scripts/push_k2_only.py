import os
import json
from kaggle.api.kaggle_api_extended import KaggleApi

k2_dir = r"d:\TTC\kaggle_account_2\kernel_supcon_all"

os.environ['KAGGLE_API_TOKEN'] = 'KGAT_d8bfa22b3323e9d1a32a52737cc6a634'
api = KaggleApi()
api.authenticate()
print(">>> Authenticated as Account 2:", api.get_config_value(api.CONFIG_NAME_USER))
print(">>> Pushing Kernel 2 (SupCon All Ratios) to Account 2...")
res2 = api.kernels_push(folder=k2_dir)
print("Result:", res2)
