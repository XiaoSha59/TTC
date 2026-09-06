#!/bin/bash
# Watcher script: Waits for the current insects 99:1 run to complete, then immediately launches the remaining models.

LOG_FILE="/home/tnpdung_79/TTC/logs/watcher_insects_remaining.log"
mkdir -p /home/tnpdung_79/TTC/logs

echo "[$(date)] >>> Watcher started. Monitoring current Insects pipeline..." | tee -a "$LOG_FILE"

# Wait as long as the current python train.py (PID 5480) or bash (PID 4354) is running
while ps aux | grep -E "python train.py.*insects.*supproto" | grep -v grep > /dev/null 2>&1; do
    sleep 20
done

# Wait another 15s to ensure linear probe finishes and GPU memory is clean
while ps aux | grep -E "python train.py.*insects" | grep -v grep > /dev/null 2>&1; do
    sleep 15
done

echo "[$(date)] >>> Insects 50:50 and 99:1 SupPrototypes have completely finished!" | tee -a "$LOG_FILE"
echo "[$(date)] >>> Launching remaining Insects models in tmux session 'insects_remaining'..." | tee -a "$LOG_FILE"

cd /home/tnpdung_79/TTC
git pull
chmod +x scripts/run_insects_remaining.sh

tmux new-session -d -s insects_remaining "bash scripts/run_insects_remaining.sh 2>&1 | tee -a /home/tnpdung_79/TTC/logs/insects_remaining.log"

echo "[$(date)] >>> Started tmux session 'insects_remaining' successfully!" | tee -a "$LOG_FILE"
