import os
import sys
from kaggle.api.kaggle_api_extended import KaggleApi

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

def audit_and_stop_old(username, key, token_mode=False):
    print(f"\n========================================================")
    print(f"🔍 AUDITING RUNNING KERNELS FOR: {username}")
    print(f"========================================================")
    
    if token_mode:
        if 'KAGGLE_USERNAME' in os.environ: del os.environ['KAGGLE_USERNAME']
        if 'KAGGLE_KEY' in os.environ: del os.environ['KAGGLE_KEY']
        os.environ['KAGGLE_API_TOKEN'] = key
    else:
        if 'KAGGLE_API_TOKEN' in os.environ: del os.environ['KAGGLE_API_TOKEN']
        os.environ['KAGGLE_USERNAME'] = username
        os.environ['KAGGLE_KEY'] = key
        
    api = KaggleApi()
    api.authenticate()
    
    kernels = api.kernels_list(user=username, page_size=20)
    for k in kernels:
        kernel_str = getattr(k, 'ref', None) or f"{getattr(k, 'author', username)}/{getattr(k, 'slug', '')}"
        try:
            status_info = api.kernels_status(kernel_str)
            status = status_info.get("status")
            print(f"• {kernel_str:<50} | Status: {status}")
        except Exception as e:
            print(f"• {kernel_str:<50} | Error: {e}")
        
    return api

if __name__ == "__main__":
    audit_and_stop_old("salala1706", "KGAT_c50dd809cbcb96fb725040dee59239f5", False)
    audit_and_stop_old("ngocuyennhitran", "KGAT_d8bfa22b3323e9d1a32a52737cc6a634", True)
