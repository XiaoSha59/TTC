#!/bin/bash
# ==============================================================================
# Auto-Queue Master 24/7 Runner immediately after Pilot finishes
# Zero downtime, zero human intervention needed
# ==============================================================================

set -e
cd /home/tnpdung_79/TTC

echo "======================================================================"
echo "⏳ Auto-Watcher Active: Monitoring session 'pilot_insects_audited'..."
echo "Will automatically launch master 24/7 runner when pilot concludes."
echo "======================================================================"

while tmux has-session -t pilot_insects_audited 2>/dev/null; do
    sleep 20
done

echo ""
echo "======================================================================"
echo "🎉 Pilot run finished! Cooling down 10s and starting Master 24/7 Runner..."
echo "======================================================================"
sleep 10

python3 -c "import torch; torch.cuda.is_available() and torch.cuda.empty_cache()" 2>/dev/null || true

bash scripts/run_master_reproduce_24_7.sh
