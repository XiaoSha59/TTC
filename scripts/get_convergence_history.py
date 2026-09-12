import os
import sys
import wandb

sys.stdout.reconfigure(encoding='utf-8')
os.environ['WANDB_API_KEY'] = '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'

api = wandb.Api()
runs = {
    "95:5 Sup-Prototypes (Ours)": "sdlgl2l8",
    "95:5 Sup-Minority (Ours)"  : "9n7czfsh",
    "50:50 Sup-Prototypes (Ours)": "ow3bgs15",
    "99:1 Sup-Prototypes (Ours)": "werm1svc"
}

for name, run_id in runs.items():
    print(f"\n{'='*70}")
    print(f"📊 MODEL: {name} (WandB ID: {run_id})")
    print(f"{'='*70}")
    try:
        r = api.run(f"tnpdung79hcmus/binary-learning/{run_id}")
        h = r.history(samples=500)
        
        # Look for loss columns
        cols = [c for c in h.columns if 'loss' in c.lower()]
        epoch_col = 'epoch' if 'epoch' in h.columns else '_step'
        
        print(f"Final Epoch: {r.summary.get('epoch', 'N/A')}")
        print(f"Final Train Loss: {r.summary.get('train.loss', r.summary.get('train.loss.epoch', 'N/A'))}")
        print(f"Final Val Loss  : {r.summary.get('val.loss', 'N/A')}")
        print(f"Online Val Acc  : {r.summary.get('online_val_acc', 'N/A')}")
        
        # Sample history
        print("\nTiến trình giảm Loss qua các mốc huấn luyện:")
        step_col = '_step'
        if 'train.loss' in h.columns:
            h_sub = h[[epoch_col, 'train.loss']].dropna().drop_duplicates(subset=[epoch_col])
            # Pick 8 evenly spaced rows
            idx = range(0, len(h_sub), max(1, len(h_sub)//8))
            for i in idx:
                row = h_sub.iloc[i]
                ep = int(row[epoch_col]) if epoch_col in row else i
                loss_val = row['train.loss']
                print(f"  • Mốc Epoch {ep:<3}: Loss = {loss_val:.4f}")
    except Exception as e:
        print(f"Error reading run {run_id}: {e}")
