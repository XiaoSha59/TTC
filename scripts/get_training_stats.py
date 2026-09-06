import os
import sys
import wandb
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding='utf-8')

os.environ['WANDB_API_KEY'] = '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'
api = wandb.Api()
runs = api.runs('tnpdung79hcmus/binary-learning', order='created_at', per_page=120)

tz_vn = timezone(timedelta(hours=7))

dataset_groups = {
    "BreastMNIST (Medical)": [],
    "PneumoniaMNIST (Medical)": [],
    "FracAtlas (Medical)": [],
    "Plants (Natural - iNat21)": [],
    "Insects (Natural - iNat21)": [],
    "Animals (Natural - iNat21)": []
}

for r in runs:
    if r.state not in ['finished', 'running']:
        continue
    name = r.name.lower()
    
    group_name = None
    if "breast" in name:
        group_name = "BreastMNIST (Medical)"
    elif "pneumonia" in name:
        group_name = "PneumoniaMNIST (Medical)"
    elif "fracatlas" in name:
        group_name = "FracAtlas (Medical)"
    elif "plants" in name:
        group_name = "Plants (Natural - iNat21)"
    elif "insects" in name:
        group_name = "Insects (Natural - iNat21)"
    elif "animals" in name:
        group_name = "Animals (Natural - iNat21)"
        
    if not group_name:
        continue
        
    created_dt = datetime.fromisoformat(r.created_at.replace('Z', '+00:00')).astimezone(tz_vn)
    runtime_sec = r.summary.get('_runtime', 0)
    if not runtime_sec and r.heartbeat_at:
        hb_dt = datetime.fromisoformat(r.heartbeat_at.replace('Z', '+00:00')).astimezone(tz_vn)
        runtime_sec = (hb_dt - created_dt).total_seconds()
        
    duration_min = runtime_sec / 60.0
    
    dataset_groups[group_name].append({
        "name": r.name,
        "start": created_dt,
        "duration_min": duration_min,
        "state": r.state
    })

print("=== THONG KE THOI GIAN HUAN LUYEN 3 NGAY QUA ===")
total_all_minutes = 0

for group, r_list in dataset_groups.items():
    if not r_list:
        continue
    total_group_min = sum(item["duration_min"] for item in r_list)
    total_all_minutes += total_group_min
    start_str = r_list[0]["start"].strftime("%d/%m %H:%M")
    end_dt = r_list[-1]["start"] + timedelta(minutes=r_list[-1]["duration_min"])
    end_str = end_dt.strftime("%d/%m %H:%M")
    hours = int(total_group_min // 60)
    mins = int(total_group_min % 60)
    
    print(f"\n📂 {group} - Tổng {len(r_list)} runs | Thời gian: {hours}h {mins}m ({total_group_min:.1f} phút)")
    print(f"   Khung giờ: {start_str}  --->  {end_str}")
    for item in r_list:
        st = item["start"].strftime("%H:%M")
        status_sym = "✅" if item["state"] == "finished" else "⏳"
        print(f"   {status_sym} {item['name']:<35} | Bắt đầu: {st} | Chạy: {item['duration_min']:>5.1f} phút")

all_hours = int(total_all_minutes // 60)
all_mins = int(total_all_minutes % 60)
print("\n" + "=" * 75)
print(f"🔥 TỔNG THỜI GIAN GPU HOẠT ĐỘNG: {all_hours} Giờ {all_mins} Phút ({total_all_minutes:.1f} phút)")
print("=" * 75)
