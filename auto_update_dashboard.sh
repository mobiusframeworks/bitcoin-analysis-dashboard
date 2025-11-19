#!/bin/bash

# Bitcoin Dashboard Auto-Update Script
# Fetches latest data, regenerates dashboard, and deploys to Vercel
# Run this script with cron for automatic updates

set -e  # Exit on error

# Configuration
UPDATE_INTERVAL=300  # 5 minutes in seconds
LOG_FILE="/tmp/bitcoin-dashboard-update.log"

# Change to script directory
cd "$(dirname "$0")"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "======================================"
log "Bitcoin Dashboard Auto-Update Starting"
log "======================================"

# Step 1: Activate virtual environment
log "Activating virtual environment..."
source ../venv/bin/activate

# Step 2: Fetch latest Bitcoin data
log "Fetching latest Bitcoin price data..."
if python3 fetch_live_bitcoin_data.py >> "$LOG_FILE" 2>&1; then
    log "✅ Bitcoin data updated successfully"
else
    log "⚠️  Warning: Bitcoin data fetch failed, using existing data"
fi

# Step 3: Regenerate all analyses
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

# Step 5: Copy to index.html
log "Preparing for deployment..."
cp reports/Bitcoin_Comprehensive_Dashboard.html reports/index.html

# Step 6: Deploy to Vercel
log "Deploying to Vercel..."
cd reports
if vercel --prod --yes >> "$LOG_FILE" 2>&1; then
    log "✅ Deployment successful!"
else
    log "❌ ERROR: Deployment failed!"
    exit 1
fi

log "======================================"
log "Update Complete - Dashboard is Live!"
log "======================================"
log ""
