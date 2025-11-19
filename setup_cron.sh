#!/bin/bash
################################################################################
# Bitcoin Analysis - Cron Job Setup Script
# This script helps set up automated daily updates via cron
################################################################################

echo "================================================================================"
echo "Bitcoin Analysis - Automated Daily Update Setup"
echo "================================================================================"
echo ""

# Configuration
SCRIPT_PATH="/Users/alexhorton/quant connect dev environment/ml_pipeline/run_daily_update.sh"
CRON_TIME="0 0 * * *"  # Midnight every day (0:00 AM)

# Check if script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "❌ ERROR: Update script not found at $SCRIPT_PATH"
    exit 1
fi

echo "📋 Current Configuration:"
echo "   Script: $SCRIPT_PATH"
echo "   Schedule: $CRON_TIME (Midnight daily)"
echo ""

# Show current crontab
echo "📅 Current crontab entries:"
crontab -l 2>/dev/null | grep -v "^#" | grep -v "^$" || echo "   (No cron jobs configured)"
echo ""

# Ask for confirmation
read -p "Do you want to add this cron job? (y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Setup cancelled."
    exit 0
fi

# Backup current crontab
echo "💾 Backing up current crontab..."
crontab -l > /tmp/crontab_backup_$(date +%Y%m%d_%H%M%S).txt 2>/dev/null
echo "   Backup saved to /tmp/crontab_backup_$(date +%Y%m%d_%H%M%S).txt"
echo ""

# Create new crontab entry
CRON_ENTRY="$CRON_TIME \"$SCRIPT_PATH\" >> /tmp/btc_cron.log 2>&1"

# Check if entry already exists
if crontab -l 2>/dev/null | grep -q "run_daily_update.sh"; then
    echo "⚠️  Cron job already exists. Updating..."
    # Remove old entry and add new one
    (crontab -l 2>/dev/null | grep -v "run_daily_update.sh"; echo "$CRON_ENTRY") | crontab -
else
    echo "➕ Adding new cron job..."
    (crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -
fi

# Verify installation
echo ""
echo "✅ Cron job installed successfully!"
echo ""
echo "📅 New crontab:"
crontab -l | grep "run_daily_update.sh"
echo ""

echo "================================================================================"
echo "Setup Complete!"
echo "================================================================================"
echo ""
echo "Your Bitcoin analysis will now update automatically every day at midnight."
echo ""
echo "📊 Useful Commands:"
echo "   View cron jobs:     crontab -l"
echo "   Edit cron jobs:     crontab -e"
echo "   Remove cron job:    crontab -e (then delete the line)"
echo "   View cron log:      tail -f /tmp/btc_cron.log"
echo "   View update logs:   ls -lh \"$ML_PIPELINE_DIR/logs/\""
echo ""
echo "🧪 Test the update script manually:"
echo "   \"$SCRIPT_PATH\""
echo ""
