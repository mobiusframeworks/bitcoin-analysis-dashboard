#!/bin/bash

# Continuous Quick Update Script for Bitcoin Dashboard
# Runs forever, updating dashboard every 5 minutes
# Run in background: ./run_continuous_quick_updates.sh &

UPDATE_INTERVAL=300  # 5 minutes (300 seconds)

cd "$(dirname "$0")"

echo "🚀 Starting continuous Bitcoin dashboard updates..."
echo "📊 Update interval: $UPDATE_INTERVAL seconds (5 minutes)"
echo "📝 Logs: /tmp/bitcoin-dashboard-quick-update.log"
echo ""
echo "To stop: pkill -f run_continuous_quick_updates.sh"
echo "To view logs: tail -f /tmp/bitcoin-dashboard-quick-update.log"
echo ""

# Make the update script executable
chmod +x run_quick_update.sh

# Run initial update
echo "[$(date)] Running initial update..."
./run_quick_update.sh

# Continuous loop
while true; do
    echo "[$(date)] Waiting $UPDATE_INTERVAL seconds before next update..."
    sleep $UPDATE_INTERVAL

    echo "[$(date)] Running update..."
    ./run_quick_update.sh
done
