import os
import wandb
from kaggle.api.kaggle_api_extended import KaggleApi

os.environ['WANDB_API_KEY'] = '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'

print("=" * 80)
print("1. WANDB LATEST RUNS")
print("=" * 80)
try:
    api = wandb.Api()
    runs = api.runs('tnpdung79hcmus/binary-learning', order='-created_at')
    for r in runs[:12]:
        name = r.name
        state = r.state
        epoch = r.summary.get('epoch', 'N/A')
        test_acc = r.summary.get('test/balanced_accuracy', r.summary.get('test_balanced_acc', 'N/A'))
        val_acc = r.summary.get('val/balanced_accuracy', r.summary.get('val_balanced_acc', 'N/A'))
        created = r.created_at
        print(f"{name:<35} | State: {state:<9} | Epoch: {str(epoch):<5} | Test BalAcc: {str(test_acc):<8} | Val BalAcc: {str(val_acc):<8} | Created: {created}")
except Exception as e:
    print(f"Error checking WandB: {e}")

print("\n" + "=" * 80)
print("2. KAGGLE KERNELS STATUS")
print("=" * 80)
try:
    kapi = KaggleApi()
    kapi.authenticate()
    for kid in ['salala1706/ttc-insects-weighted-ce-95-5', 'salala1706/ttc-insects-supcon-95-5']:
        st = kapi.kernels_status(kid)
        print(f"{kid:<45} -> Status: {st.get('status')} (Failure: {st.get('failureMessage')})")
except Exception as e:
    print(f"Error checking Kaggle: {e}")
