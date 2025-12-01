#!/bin/bash

# Quick Bitcoin Dashboard Update Script
# Just fetches data and regenerates main dashboard (no heavy analyses)

set -e

LOG_FILE="/tmp/bitcoin-dashboard-quick-update.log"

cd "$(dirname "$0")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "======================================"
log "Bitcoin Dashboard Quick Update"
log "======================================"

# Fetch latest Bitcoin data
log "Fetching latest Bitcoin price data..."
if python3 fetch_live_bitcoin_data.py >> "$LOG_FILE" 2>&1; then
    log "✅ Bitcoin data updated successfully"
else
    log "⚠️  Warning: Bitcoin data fetch failed, using existing data"
fi

# Generate dashboard (this should work with just bitcoin_live.csv)
log "Generating dashboard..."
if python3 generate_comprehensive_bitcoin_website.py >> "$LOG_FILE" 2>&1; then
    log "✅ Dashboard generated successfully"
    cp reports/Bitcoin_Comprehensive_Dashboard.html reports/index.html
    log "✅ Dashboard available at http://localhost:8000"
else
    log "❌ ERROR: Dashboard generation failed - check log for details"
    tail -50 "$LOG_FILE"
    exit 1
fi

log "======================================"
log "Quick Update Complete!"
log "======================================"
