import os
import subprocess

# Configure Kaggle credentials on VM
setup_cmd = (
    "export KAGGLE_USERNAME=salala1706 && "
    "export KAGGLE_KEY=KGAT_c50dd809cbcb96fb725040dee59239f5 && "
    "cd /home/XiaoSha/TTC && "
    "/home/XiaoSha/TTC/.venv/bin/kaggle datasets create -p /home/XiaoSha/TTC/data/kaggle_insects_full_pkg --dir-mode zip -u || "
    "/home/XiaoSha/TTC/.venv/bin/kaggle datasets version -p /home/XiaoSha/TTC/data/kaggle_insects_full_pkg -m 'Full Insects dataset' --dir-mode zip"
)

gcloud_cmd = [
    "gcloud", "compute", "ssh", "instance-20260903-164213",
    "--zone=us-central1-a",
    f"--command={setup_cmd}"
]

print(">>> Executing Kaggle dataset push from GCP VM...")
res = subprocess.run(gcloud_cmd, capture_output=True, text=True, shell=True)
print("STDOUT:", res.stdout)
print("STDERR:", res.stderr)
