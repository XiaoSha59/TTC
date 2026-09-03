# ==============================================================================
# Script đồng bộ code từ máy Local (Windows) lên GCP VM
# Usage: .\scripts\sync_to_gcp.ps1 -InstanceName <TÊN_VM> -Zone <ZONE>
# ==============================================================================
param (
    [Parameter(Mandatory=$true)]
    [string]$InstanceName,
    
    [Parameter(Mandatory=$true)]
    [string]$Zone
)

Write-Host ">>> Đang đồng bộ mã nguồn lên GCP VM: $InstanceName ($Zone)..." -ForegroundColor Cyan

# Sử dụng gcloud compute scp để đồng bộ các thư mục code, loại trừ .venv, datasets và cache
gcloud compute scp --recurse `
    --zone=$Zone `
    configs data models utils metrics scripts loss.py train.py requirements.txt `
    "${InstanceName}:~/TTC/"

Write-Host ">>> Đồng bộ code hoàn tất!" -ForegroundColor Green
