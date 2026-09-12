import urllib.request
import urllib.error
import json
import base64
import os

api_key = os.environ.get('WANDB_API_KEY', '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')
# Wandb basic auth is username 'api' and password is the api key
auth = base64.b64encode(f'api:{api_key}'.encode('utf-8')).decode('utf-8')

# Let's test standard WandB REST API endpoint: https://api.wandb.ai/api/v1/runs/tnpdung79hcmus/binary-learning
url = 'https://api.wandb.ai/api/v1/runs/tnpdung79hcmus/binary-learning?per_page=100'
req = urllib.request.Request(
    url,
    headers={
        'Authorization': f'Basic {auth}',
        'Content-Type': 'application/json'
    }
)

try:
    with urllib.request.urlopen(req) as response:
        res = json.loads(response.read().decode('utf-8'))
        runs = res.get('objects', [])
        print(f"Total runs fetched via REST: {len(runs)}")
        with open('wandb_full_dump.json', 'w') as f:
            json.dump(runs, f, indent=2)
except urllib.error.HTTPError as e:
    err_body = e.read().decode('utf-8')
    print(f"HTTPError {e.code}: {err_body}")
