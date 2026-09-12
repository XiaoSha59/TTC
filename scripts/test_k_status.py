from kaggle.api.kaggle_api_extended import KaggleApi
import os

try:
    k = KaggleApi()
    k.authenticate()
    st = k.kernels_status('salala1706/ttc-insects-weighted-ce-95-5')
    print("Type:", type(st))
    print("Dir:", dir(st))
    print("Dict/Attrs:", {attr: getattr(st, attr) for attr in dir(st) if not attr.startswith('_')})
except Exception as e:
    print("Error:", e)
