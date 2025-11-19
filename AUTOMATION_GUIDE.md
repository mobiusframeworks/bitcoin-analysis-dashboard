# Bitcoin Analysis - Automated Daily Updates

## Overview

Your Bitcoin analysis dashboard can now update automatically every day with the latest data from Coinbase and FRED (Federal Reserve Economic Data).

---

## 🚀 Quick Start (3 Steps)

### 1. Test Manual Update

First, verify the update script works:

```bash
cd "/Users/alexhorton/quant connect dev environment/ml_pipeline"
./run_daily_update.sh
```

This should:
- ✅ Fetch latest Bitcoin price from Coinbase
- ✅ Download FRED economic indicators
- ✅ Update comprehensive dataset
- ✅ Regenerate all reports and website
- ✅ Create a log file in `logs/`

**Expected duration:** ~30 seconds

---

### 2. Set Up Automated Daily Updates

Run the setup script to install a cron job:

```bash
cd "/Users/alexhorton/quant connect dev environment/ml_pipeline"
./setup_cron.sh
```

This will:
- Add a cron job to run updates daily at midnight
- Create a backup of your current crontab
- Verify the installation

**When prompted, type `y` and press Enter.**

---

### 3. Verify Automation is Running

Check that the cron job was installed:

```bash
crontab -l | grep "run_daily_update"
```

You should see:
```
0 0 * * * "/Users/alexhorton/quant connect dev environment/ml_pipeline/run_daily_update.sh" >> /tmp/btc_cron.log 2>&1
```

---

## 📅 Schedule

**Default schedule:** Every day at **12:00 AM (midnight)**

### Change the Schedule

Edit your crontab:
```bash
crontab -e
```

Common cron schedules:
- `0 0 * * *` - Midnight every day (default)
- `0 1 * * *` - 1:00 AM every day
- `0 */6 * * *` - Every 6 hours
- `0 9 * * 1-5` - 9:00 AM on weekdays only
- `*/30 * * * *` - Every 30 minutes

---

## 📊 What Gets Updated

Every day, the automation:

### 1. **Fetches Latest Data**
- Bitcoin OHLC (Open, High, Low, Close, Volume) from Coinbase API
- M2 Money Supply from FRED
- Federal Funds Rate from FRED
- 10-Year Treasury Rate from FRED
- Consumer Price Index (CPI) from FRED
- Fed Balance Sheet (WALCL) from FRED
- USD/EUR Exchange Rate from FRED
- WTI Crude Oil Price from FRED

### 2. **Updates Files**
- `datasets/btc-ohlc.csv` - Appends latest Bitcoin data
- `datasets/btc_comprehensive_data.csv` - Full dataset with all indicators (27 columns)
- `ml_pipeline/daily_update_log.json` - Update metadata

### 3. **Regenerates Reports**
- `reports/m2_interest_rate_study_results.json` - M2 analysis results
- `reports/Bitcoin_Comprehensive_Dashboard.html` - Main dashboard

### 4. **Creates Logs**
- `ml_pipeline/logs/daily_update_YYYYMMDD.log` - Detailed log for each day

---

## 🔍 Monitoring

### View Today's Update Log

```bash
tail -f "/Users/alexhorton/quant connect dev environment/ml_pipeline/logs/daily_update_$(date +%Y%m%d).log"
```

### View Cron Execution Log

```bash
tail -f /tmp/btc_cron.log
```

### Check Last Update Status

```bash
cd "/Users/alexhorton/quant connect dev environment/ml_pipeline"
cat daily_update_log.json | python3 -m json.tool
```

This shows:
- Timestamp of last update
- Latest Bitcoin price
- Number of records
- FRED series fetched
- Success/failure status

---

## 🛠️ Troubleshooting

### Cron Job Not Running?

**Check if cron is enabled (macOS Ventura+):**

1. Open **System Settings** → **Privacy & Security** → **Full Disk Access**
2. Ensure `/usr/sbin/cron` or **Terminal** has Full Disk Access
3. Restart cron: `sudo launchctl kickstart -k system/com.apple.cron`

**Verify cron is running:**
```bash
ps aux | grep cron
```

**Check system logs:**
```bash
log show --predicate 'subsystem == "com.apple.cron"' --last 1h
```

---

### Update Script Failing?

**Run manually with verbose output:**
```bash
cd "/Users/alexhorton/quant connect dev environment/ml_pipeline"
source ../venv/bin/activate
python3 update_daily_data.py
```

**Common issues:**
- **No internet connection** → Script will fail to fetch data
- **FRED API rate limits** → Wait 1 minute and retry (free tier allows ~120 requests/hour)
- **Coinbase API down** → Script will retry with cached data
- **Disk space full** → Free up space in `/Users/alexhorton/`

**Check error logs:**
```bash
grep "ERROR" "/Users/alexhorton/quant connect dev environment/ml_pipeline/logs/daily_update_$(date +%Y%m%d).log"
```

---

### Virtual Environment Not Found?

Recreate the virtual environment:
```bash
cd "/Users/alexhorton/quant connect dev environment"
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy matplotlib scipy scikit-learn statsmodels requests
```

---

## 📧 Email Notifications (Optional)

To receive email notifications on update failures:

### 1. Install `mailutils` (if not already installed)
```bash
brew install mailutils
```

### 2. Modify `run_daily_update.sh`

Add this at the end (before `exit $EXIT_CODE`):

```bash
if [ $EXIT_CODE -ne 0 ]; then
    echo "Bitcoin analysis update failed on $(date)" | mail -s "Bitcoin Update Failed" your_email@example.com
fi
```

### 3. Configure mail settings

Edit `/etc/mail.rc` or `~/.mailrc` to configure SMTP settings.

---

## 🗑️ Remove Automation

To stop automatic updates:

```bash
crontab -e
```

Delete the line containing `run_daily_update.sh`, then save and exit.

Or remove all cron jobs:
```bash
crontab -r
```

---

## 📂 File Structure

```
ml_pipeline/
├── update_daily_data.py           # Main update script (fetches data)
├── run_daily_update.sh            # Shell wrapper with logging
├── setup_cron.sh                  # Cron installation script
├── generate_m2_interest_rate_bitcoin_study.py
├── generate_comprehensive_bitcoin_website.py
├── daily_update_log.json          # Last update metadata
└── logs/
    ├── daily_update_20251119.log  # Today's log
    ├── daily_update_20251118.log  # Yesterday's log
    └── ...                         # Logs kept for 30 days

datasets/
├── btc-ohlc.csv                   # Bitcoin OHLC data
└── btc_comprehensive_data.csv     # Full dataset (27 columns)

reports/
├── Bitcoin_Comprehensive_Dashboard.html
├── m2_interest_rate_study_results.json
└── ...
```

---

## 🔐 Security Notes

- **No API keys stored** - Uses public FRED CSV endpoint (no authentication required)
- **Coinbase public API** - No authentication needed for price data
- **Logs contain sensitive data?** - No, logs only contain public market data
- **File permissions** - Scripts are executable only by owner (`-rwx--x--x`)

---

## 💡 Advanced Usage

### Run Update at Specific Time

To update at 8:00 AM instead of midnight:

```bash
crontab -e
```

Change:
```
0 0 * * *   # Midnight
```

To:
```
0 8 * * *   # 8:00 AM
```

---

### Multiple Updates Per Day

To update every 6 hours:

```bash
crontab -e
```

Change to:
```
0 */6 * * * "/Users/alexhorton/quant connect dev environment/ml_pipeline/run_daily_update.sh" >> /tmp/btc_cron.log 2>&1
```

**Note:** FRED data is updated daily/weekly/monthly, so multiple updates per day won't fetch new economic data.

---

### Backup Automation

Add this cron job to backup your data weekly:

```bash
0 0 * * 0 tar -czf "/Users/alexhorton/btc_backup_$(date +\%Y\%m\%d).tar.gz" "/Users/alexhorton/quant connect dev environment/datasets/"
```

This creates a backup every Sunday at midnight.

---

## 📞 Support

If automation fails repeatedly:

1. Check logs: `tail -100 logs/daily_update_$(date +%Y%m%d).log`
2. Run manual update: `./run_daily_update.sh`
3. Verify internet connection
4. Check FRED status: https://fred.stlouisfed.org/
5. Check Coinbase status: https://status.coinbase.com/

---

## ✅ Success Checklist

After setup, verify:

- [ ] Manual update works: `./run_daily_update.sh` completes without errors
- [ ] Cron job installed: `crontab -l` shows the entry
- [ ] Logs directory created: `ls logs/` shows log files
- [ ] Dashboard updates: Bitcoin_Comprehensive_Dashboard.html shows today's date
- [ ] Latest price correct: Check against https://www.coinbase.com/price/bitcoin

---

**Last Updated:** November 19, 2025
**Version:** 1.0
**Automation Status:** ✅ Active
