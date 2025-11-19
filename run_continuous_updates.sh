#!/bin/bash

# Continuous Update Script for Bitcoin Dashboard
# Runs forever, updating dashboard every 5 minutes
# Run in background: nohup ./run_continuous_updates.sh &

UPDATE_INTERVAL=300  # 5 minutes

cd "$(dirname "$0")"

echo "🚀 Starting continuous Bitcoin dashboard updates..."
echo "📊 Update interval: $UPDATE_INTERVAL seconds (5 minutes)"
echo "📝 Logs: /tmp/bitcoin-dashboard-update.log"
echo ""
echo "To stop: pkill -f run_continuous_updates.sh"
echo "To view logs: tail -f /tmp/bitcoin-dashboard-update.log"
echo ""

while true; do
    echo "[$(date)] Running update..."
    ./auto_update_dashboard.sh

    echo "[$(date)] Update complete. Waiting $UPDATE_INTERVAL seconds..."
    sleep $UPDATE_INTERVAL
done
