import requests

token = 'KGAT_c50dd809cbcb96fb725040dee59239f5'
headers = {
    'Authorization': f'Bearer {token}',
    'Content-Type': 'application/json'
}

# Pull current metadata to keep everything consistent
meta_url = 'https://www.kaggle.com/api/v1/kernels/pull?userName=salala1706&kernelSlug=ttc-natural-weighted-ce'
r_meta = requests.get(meta_url, headers=headers).json()
current_meta = r_meta.get('metadata', {})
print(f">>> Found active kernel ID: {current_meta.get('id')}, Current Version: {current_meta.get('currentVersionNumber')}")

# Push a lightweight CPU script to instantly kill the running GPU version
payload = {
    "slug": "salala1706/ttc-natural-weighted-ce",
    "newTitle": "TTC Natural Weighted CE",
    "text": "print('=== EXECUTION STOPPED BY USER: GPU T4 FREED ===')",
    "language": "python",
    "kernelType": "script",
    "isPrivate": True,
    "enableGpu": False,
    "enableTpu": False,
    "enableInternet": False,
    "datasetDataSources": ["salala1706/inat21-natural"]
}

push_url = 'https://www.kaggle.com/api/v1/kernels/push'
r_push = requests.post(push_url, json=payload, headers=headers)
print("Push Response Status:", r_push.status_code)
print("Push Response Body:", r_push.text)

# Check status after push
r_status = requests.get('https://www.kaggle.com/api/v1/kernels/status?userName=salala1706&kernelSlug=ttc-natural-weighted-ce', headers=headers)
print("New Kernel Status:", r_status.json())
