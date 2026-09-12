import urllib.request
import urllib.error
import json
import base64
import os

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
          summaryMetrics
          config
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

try:
    with urllib.request.urlopen(req) as response:
        res = json.loads(response.read().decode('utf-8'))
        if 'errors' in res:
            print("GraphQL Errors:", res['errors'])
        else:
            runs = res.get('data', {}).get('project', {}).get('runs', {}).get('edges', [])
            print(f"Total runs fetched: {len(runs)}")
            with open('wandb_dump.json', 'w') as f:
                json.dump(runs, f, indent=2)
            for r in runs:
                node = r['node']
                name = node.get('displayName') or node.get('name')
                state = node.get('state')
                created = node.get('createdAt', '')[:19]
                sm_str = node.get('summaryMetrics') or '{}'
                try:
                    sm = json.loads(sm_str)
                except Exception:
                    sm = {}
                epoch = sm.get('epoch', sm.get('trainer/global_step'))
                val_acc = sm.get('online_val_acc') or sm.get('test.acc') or sm.get('val.acc') or sm.get('test/balanced_accuracy') or sm.get('test_balanced_acc') or sm.get('val/balanced_accuracy')
                auc = sm.get('test.auc') or sm.get('val.auroc') or sm.get('test/auc')
                print(f"{name:<45} | {state:<10} | ep={str(epoch):<5} | acc={str(val_acc):<18} | auc={str(auc):<18} | {created}")
except urllib.error.HTTPError as e:
    print(f"HTTPError {e.code}: {e.read().decode('utf-8')}")
