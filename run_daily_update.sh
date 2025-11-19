#!/bin/bash
################################################################################
# Bitcoin Analysis - Daily Auto-Update Script
# Runs daily to fetch latest data and regenerate all reports
################################################################################

# Configuration
BASE_DIR="/Users/alexhorton/quant connect dev environment"
ML_PIPELINE_DIR="$BASE_DIR/ml_pipeline"
LOG_DIR="$ML_PIPELINE_DIR/logs"
LOG_FILE="$LOG_DIR/daily_update_$(date +%Y%m%d).log"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Start logging
echo "================================================================================" | tee -a "$LOG_FILE"
echo "Bitcoin Analysis Daily Update" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Change to ml_pipeline directory
cd "$ML_PIPELINE_DIR" || {
    echo "❌ ERROR: Could not change to $ML_PIPELINE_DIR" | tee -a "$LOG_FILE"
    exit 1
}

# Activate virtual environment
source "$BASE_DIR/venv/bin/activate" || {
    echo "❌ ERROR: Could not activate virtual environment" | tee -a "$LOG_FILE"
    exit 1
}

echo "✅ Virtual environment activated" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Run the daily update script
echo "🔄 Running daily data update..." | tee -a "$LOG_FILE"
python3 update_daily_data.py 2>&1 | tee -a "$LOG_FILE"

# Check if update was successful
if [ $? -eq 0 ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "✅ Daily update completed successfully!" | tee -a "$LOG_FILE"

    # Read the update log to get latest price
    if [ -f "daily_update_log.json" ]; then
        LATEST_PRICE=$(python3 -c "import json; data=json.load(open('daily_update_log.json')); print('{:,.2f}'.format(data['latest_btc_price']))" 2>/dev/null)
        if [ -n "$LATEST_PRICE" ]; then
            echo "📊 Latest Bitcoin Price: \$$LATEST_PRICE" | tee -a "$LOG_FILE"
        fi
    fi

    EXIT_CODE=0
else
    echo "" | tee -a "$LOG_FILE"
    echo "❌ ERROR: Daily update failed!" | tee -a "$LOG_FILE"
    EXIT_CODE=1
fi

# Cleanup old logs (keep last 30 days)
echo "" | tee -a "$LOG_FILE"
echo "🧹 Cleaning up old logs (keeping last 30 days)..." | tee -a "$LOG_FILE"
find "$LOG_DIR" -name "daily_update_*.log" -type f -mtime +30 -delete 2>/dev/null
REMAINING_LOGS=$(find "$LOG_DIR" -name "daily_update_*.log" -type f | wc -l | xargs)
echo "   Remaining log files: $REMAINING_LOGS" | tee -a "$LOG_FILE"

# End logging
echo "" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "Completed: $(date)" | tee -a "$LOG_FILE"
echo "Exit Code: $EXIT_CODE" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"

exit $EXIT_CODE
