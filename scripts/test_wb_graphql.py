import urllib.request
import urllib.error
import json
import base64
import os

api_key = os.environ.get('WANDB_API_KEY', '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')
auth = base64.b64encode(f'api:{api_key}'.encode('utf-8')).decode('utf-8')

query = """
query {
  viewer {
    username
  }
}
"""

req = urllib.request.Request(
    'https://api.wandb.ai/graphql',
    data=json.dumps({'query': query}).encode('utf-8'),
    headers={
        'Authorization': f'Basic {auth}',
        'Content-Type': 'application/json'
    }
)

try:
    with urllib.request.urlopen(req) as response:
        res = json.loads(response.read().decode('utf-8'))
        print("Viewer response:", res)
except urllib.error.HTTPError as e:
    print(f"HTTPError {e.code}: {e.read().decode('utf-8')}")
