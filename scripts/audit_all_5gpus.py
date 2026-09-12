import os
import json
import urllib.request
import base64
import subprocess

# 1. WandB Check
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
          tags
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

print("=" * 100)
print("1. WANDB RECENT RUNS (ALL EXPERIMENTS)")
print("=" * 100)
wb_runs = []
try:
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode('utf-8'))
        runs = res.get('data', {}).get('project', {}).get('runs', {}).get('edges', [])
        for r in runs:
            node = r['node']
            name = node.get('displayName') or node.get('name')
            state = node.get('state')
            created = node.get('createdAt', '')[:19]
            updated = node.get('updatedAt', '')[:19]
            sm_str = node.get('summaryMetrics') or '{}'
            try:
                sm = json.loads(sm_str)
            except Exception:
                sm = {}
            wb_runs.append({
                'name': name,
                'state': state,
                'created': created,
                'updated': updated,
                'summary': sm
            })
            
        for r in wb_runs[:25]:
            sm = r['summary']
            ep = sm.get('epoch')
            val_acc = sm.get('online_val_acc') or sm.get('test.acc') or sm.get('val.acc') or sm.get('test/balanced_accuracy') or sm.get('test_balanced_acc') or sm.get('val/balanced_accuracy')
            auc = sm.get('test.auc') or sm.get('val.auroc') or sm.get('test/auc')
            m_str = []
            if ep is not None: m_str.append(f"ep={ep}")
            if val_acc is not None: m_str.append(f"Acc={val_acc*100:.2f}%" if isinstance(val_acc, (int, float)) and val_acc <= 1.0 else f"Acc={val_acc}")
            if auc is not None: m_str.append(f"AUC={auc*100:.2f}%" if isinstance(auc, (int, float)) and auc <= 1.0 else f"AUC={auc}")
            print(f"{r['name']:<42} | {r['state']:<10} | {', '.join(m_str) if m_str else 'N/A':<25} | {r['updated']}")
except Exception as e:
    print(f"Error checking WandB: {e}")

# 2. Check Kaggle Account 1
print("\n" + "=" * 100)
print("2. KAGGLE ACCOUNT 1 (salala1706)")
print("=" * 100)
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    # If default kaggle.json is account 1
    api1 = KaggleApi()
    api1.authenticate()
    for kid in [
        'salala1706/ttc-insects-weighted-ce-95-5',
        'salala1706/ttc-insects-supcon-95-5',
        'salala1706/ttc-medical-weighted-ce'
    ]:
        try:
            st = api1.kernels_status(kid)
            print(f"Kernel: {kid:<45} | Status: {st.get('status')} | Message: {st.get('failureMessage')}")
        except Exception as ex:
            print(f"Kernel: {kid:<45} | Error: {ex}")
except Exception as e:
    print(f"Error checking Kaggle 1: {e}")

# 3. Check Kaggle Account 2
print("\n" + "=" * 100)
print("3. KAGGLE ACCOUNT 2 (ngocuyennhitran)")
print("=" * 100)
try:
    os.environ["KAGGLE_API_TOKEN"] = "KGAT_d8bfa22b3323e9d1a32a52737cc6a634"
    api2 = KaggleApi()
    api2.authenticate()
    for kid in [
        'ngocuyennhitran/ttc-insects-supmin-50-50',
        'ngocuyennhitran/ttc-insects-supcon-50-50-and-99-1'
    ]:
        try:
            st = api2.kernels_status(kid)
            print(f"Kernel: {kid:<45} | Status: {st.get('status')} | Message: {st.get('failureMessage')}")
        except Exception as ex:
            print(f"Kernel: {kid:<45} | Error: {ex}")
except Exception as e:
    print(f"Error checking Kaggle 2: {e}")
