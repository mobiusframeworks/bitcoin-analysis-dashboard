#!/bin/bash

# Local Bitcoin Dashboard Auto-Update Script
# Fetches latest data and regenerates dashboard (no Vercel deployment)
# For localhost development

set -e  # Exit on error

# Configuration
LOG_FILE="/tmp/bitcoin-dashboard-local-update.log"

# Change to script directory
cd "$(dirname "$0")"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "======================================"
log "Bitcoin Dashboard Local Update Starting"
log "======================================"

# Step 1: Check for virtual environment
if [ -d "../venv" ]; then
    log "Activating virtual environment..."
    source ../venv/bin/activate
elif [ -d "venv" ]; then
    log "Activating virtual environment..."
    source venv/bin/activate
else
    log "⚠️  No virtual environment found, using system Python"
fi

# Step 2: Fetch latest Bitcoin data
log "Fetching latest Bitcoin price data..."
if python3 fetch_live_bitcoin_data.py >> "$LOG_FILE" 2>&1; then
    log "✅ Bitcoin data updated successfully"
else
    log "⚠️  Warning: Bitcoin data fetch failed, using existing data"
fi

# Step 3: Regenerate analyses (optional - can comment out for faster updates)
log "Regenerating M2 analysis..."
if python3 generate_m2_interest_rate_bitcoin_study.py >> "$LOG_FILE" 2>&1; then
    log "✅ M2 analysis complete"
else
    log "⚠️  Warning: M2 analysis failed, using existing results"
fi

log "Regenerating trading strategy analysis..."
if python3 generate_trading_strategy_analysis.py >> "$LOG_FILE" 2>&1; then
    log "✅ Trading strategy analysis complete"
else
    log "⚠️  Warning: Trading strategy analysis failed, using existing results"
fi

# Step 4: Generate comprehensive dashboard
log "Generating comprehensive dashboard..."
if python3 generate_comprehensive_bitcoin_website.py >> "$LOG_FILE" 2>&1; then
    log "✅ Dashboard generated successfully"
else
    log "❌ ERROR: Dashboard generation failed!"
    exit 1
fi

# Step 5: Copy to index.html for local serving
log "Updating index.html..."
cp reports/Bitcoin_Comprehensive_Dashboard.html reports/index.html

log "======================================"
log "Local Update Complete!"
log "Dashboard available at http://localhost:8000"
log "======================================"
log ""
