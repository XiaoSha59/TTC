import os
import json
import urllib.request
import base64

# 1. Fetch all WandB runs
api_key = os.environ.get('WANDB_API_KEY', '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')
auth = base64.b64encode(f'api:{api_key}'.encode('utf-8')).decode('utf-8')

query = """
query ModelRuns($entityName: String!, $projectName: String!) {
  project(name: $projectName, entityName: $entityName) {
    runs(first: 100) {
      edges {
        node {
          id
          name
          displayName
          state
          createdAt
          updatedAt
          summaryMetrics
        }
      }
    }
  }
}
"""

req = urllib.request.Request(
    'https://api.wandb.ai/graphql',
    data=json.dumps({
        'query': query,
        'variables': {'entityName': 'tnpdung79hcmus', 'projectName': 'binary-learning'}
    }).encode('utf-8'),
    headers={
        'Authorization': f'Basic {auth}',
        'Content-Type': 'application/json'
    }
)

with urllib.request.urlopen(req) as resp:
    res = json.loads(resp.read().decode('utf-8'))
    runs_edges = res.get('data', {}).get('project', {}).get('runs', {}).get('edges', [])

all_runs = []
for r in runs_edges:
    node = r['node']
    name = node.get('displayName') or node.get('name')
    state = node.get('state')
    created = node.get('createdAt', '')
    updated = node.get('updatedAt', '')
    sm_str = node.get('summaryMetrics') or '{}'
    try:
        sm = json.loads(sm_str)
    except Exception:
        sm = {}
    all_runs.append({
        'name': name,
        'state': state,
        'created': created,
        'updated': updated,
        'summary': sm
    })

# Kaggle status checking
from kaggle.api.kaggle_api_extended import KaggleApi

kaggle_status = {}
print(">>> Checking Kaggle Account 1...")
try:
    os.environ.pop("KAGGLE_API_TOKEN", None)
    k1 = KaggleApi()
    k1.authenticate()
    for kid in ['salala1706/ttc-insects-weighted-ce-95-5', 'salala1706/ttc-insects-supcon-95-5', 'salala1706/ttc-medical-weighted-ce']:
        st = k1.kernels_status(kid)
        kaggle_status[kid] = f"Status: {st.status}, Failure: {st.failure_message}"
except Exception as e:
    kaggle_status["Account1_err"] = str(e)

print(">>> Checking Kaggle Account 2...")
try:
    os.environ["KAGGLE_API_TOKEN"] = "KGAT_d8bfa22b3323e9d1a32a52737cc6a634"
    k2 = KaggleApi()
    k2.authenticate()
    for kid in ['ngocuyennhitran/ttc-insects-supmin-50-50', 'ngocuyennhitran/ttc-insects-supcon-50-50-and-99-1']:
        st = k2.kernels_status(kid)
        kaggle_status[kid] = f"Status: {st.status}, Failure: {st.failure_message}"
except Exception as e:
    kaggle_status["Account2_err"] = str(e)

print("\n--- KAGGLE KERNELS STATUS ---")
for k, v in kaggle_status.items():
    print(f"{k}: {v}")

with open('all_evaluation_runs.json', 'w', encoding='utf-8') as f:
    json.dump({'kaggle': kaggle_status, 'runs': all_runs}, f, indent=2, ensure_ascii=False)

print(f"\nSaved {len(all_runs)} runs to all_evaluation_runs.json")
