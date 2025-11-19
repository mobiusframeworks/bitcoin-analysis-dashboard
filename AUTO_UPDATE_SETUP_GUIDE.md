# Bitcoin Dashboard Auto-Update Setup Guide

## ✅ What's Been Updated

Your dashboard now has:

1. **₿ Bitcoin Logo Favicon** - Shows the Bitcoin symbol in browser tabs
2. **Auto-Refresh** - Page refreshes every 5 minutes automatically
3. **Updated Title** - "₿ Bitcoin Analysis - Live Dashboard"
4. **Automated Update Scripts** - Ready to deploy fresh data continuously

## 🌐 Your Live Dashboard

**Production URL:**
https://bitcoin-analysis-dashboard-6092pn9cd-alexs-projects-543ee4d6.vercel.app

The page now automatically refreshes every 5 minutes, BUT to get fresh data from the blockchain, you need to run the auto-update scripts.

## 🚀 Auto-Update Options

Choose one of these methods to keep your dashboard updated:

### Option 1: Run Continuously (Recommended for Testing)

This runs in the foreground so you can see what's happening:

```bash
cd "/Users/alexhorton/quant connect dev environment/ml_pipeline"
./run_continuous_updates.sh
```

**What it does:**
- Fetches latest Bitcoin price every 5 minutes
- Regenerates all analyses
- Deploys to Vercel automatically
- Shows live progress in terminal

**To stop:** Press `Ctrl+C`

### Option 2: Run in Background

Run it in the background and continue working:

```bash
cd "/Users/alexhorton/quant connect dev environment/ml_pipeline"
nohup ./run_continuous_updates.sh > /tmp/bitcoin-update-console.log 2>&1 &
```

**View logs:**
```bash
# Watch updates in real-time
tail -f /tmp/bitcoin-dashboard-update.log

# Or console output
tail -f /tmp/bitcoin-update-console.log
```

**To stop:**
```bash
pkill -f run_continuous_updates.sh
```

### Option 3: macOS LaunchAgent (Auto-Start on Boot)

To make it start automatically when you log in:

```bash
# Copy the plist to LaunchAgents
cp com.bitcoin.dashboard.plist ~/Library/LaunchAgents/

# Load it
launchctl load ~/Library/LaunchAgents/com.bitcoin.dashboard.plist

# Start it now
launchctl start com.bitcoin.dashboard
```

**Check if it's running:**
```bash
launchctl list | grep bitcoin
```

**View logs:**
```bash
tail -f /tmp/bitcoin-dashboard-update.log
```

**Stop it:**
```bash
launchctl stop com.bitcoin.dashboard
launchctl unload ~/Library/LaunchAgents/com.bitcoin.dashboard.plist
```

### Option 4: Cron Job (Traditional Unix Way)

For a traditional cron job (runs every 5 minutes):

```bash
# Edit crontab
crontab -e

# Add this line (runs every 5 minutes):
*/5 * * * * cd "/Users/alexhorton/quant connect dev environment/ml_pipeline" && ./auto_update_dashboard.sh >> /tmp/bitcoin-cron.log 2>&1
```

**Note:** Cron may not work well with GUI apps on macOS. Use LaunchAgent instead.

## 📊 What Gets Updated

Each update cycle:

1. **Fetches latest Bitcoin price** from Yahoo Finance
2. **Regenerates M2 analysis** with latest economic data
3. **Updates trading strategy** with current signals
4. **Generates new dashboard** with all fresh data
5. **Deploys to Vercel** automatically

## ⚡ Update Frequency

Currently set to update every **5 minutes**. To change:

Edit the interval in `run_continuous_updates.sh`:
```bash
UPDATE_INTERVAL=300  # Change this number (in seconds)
```

Recommendations:
- **5 minutes (300)** - Good for testing, may hit API rate limits
- **10 minutes (600)** - Balanced, recommended for production
- **15 minutes (900)** - Conservative, less API calls
- **30 minutes (1800)** - Safe for 24/7 operation

## 🔍 Monitoring Updates

### View Real-Time Logs

```bash
tail -f /tmp/bitcoin-dashboard-update.log
```

### Check Last Update Time

Visit your dashboard and check the "Last Updated" timestamp in the header.

### Vercel Deployment Logs

Visit: https://vercel.com/alexs-projects-543ee4d6/bitcoin-analysis-dashboard

Click on any deployment to see logs and analytics.

## 🛠 Troubleshooting

### Updates Not Running?

```bash
# Check if process is running
ps aux | grep run_continuous_updates

# Check logs for errors
tail -100 /tmp/bitcoin-dashboard-update.log
```

### API Rate Limits?

If you see errors about rate limits, increase the update interval to 10-15 minutes.

### Deployment Failing?

```bash
# Check Vercel auth
vercel whoami

# Re-login if needed
vercel login
```

### Out of Memory?

Long-running Python scripts may accumulate memory. The continuous script restarts the Python process each update to avoid this.

## 📈 One-Time Manual Update

To manually trigger an update without waiting:

```bash
cd "/Users/alexhorton/quant connect dev environment/ml_pipeline"
./auto_update_dashboard.sh
```

## 🎯 Recommended Setup for You

For the best experience:

1. **Start with Option 1** (foreground) to verify everything works
2. **Once confirmed, use Option 3** (LaunchAgent) for automatic startup
3. **Set interval to 10 minutes** to balance freshness with API limits

**Quick Start Command:**

```bash
cd "/Users/alexhorton/quant connect dev environment/ml_pipeline"

# Test it once manually first
./auto_update_dashboard.sh

# If that works, set up auto-start
cp com.bitcoin.dashboard.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.bitcoin.dashboard.plist
launchctl start com.bitcoin.dashboard

# Watch it work
tail -f /tmp/bitcoin-dashboard-update.log
```

## 🌐 What Users See

When someone visits your dashboard:

1. **Initial Load:** Shows the most recent data
2. **After 5 Minutes:** Page auto-refreshes
3. **Fresh Data:** If auto-update is running, they see newly fetched data

The combination of:
- Auto-update scripts (backend) every 5-10 min
- Auto-refresh meta tag (frontend) every 5 min

Ensures users always see fresh Bitcoin data!

## 💰 Cost Considerations

**Vercel Free Tier:**
- 100 deployments/day
- With 5-min updates: 288 deployments/day (over limit!)
- With 10-min updates: 144 deployments/day (over limit!)
- With 15-min updates: 96 deployments/day (within limit)

**Recommendation:**
- Use **15-minute intervals** for 24/7 operation on free tier
- Or upgrade to Vercel Pro ($20/month) for unlimited deployments

**To change to 15 minutes:**

```bash
# Edit run_continuous_updates.sh
UPDATE_INTERVAL=900  # 15 minutes
```

## 📝 Files Created

1. `auto_update_dashboard.sh` - Single update run
2. `run_continuous_updates.sh` - Continuous update loop
3. `com.bitcoin.dashboard.plist` - macOS LaunchAgent config
4. This guide - `AUTO_UPDATE_SETUP_GUIDE.md`

All logs go to `/tmp/bitcoin-dashboard-*.log`

## ✅ Next Steps

1. Test manual update first
2. Choose your preferred auto-update method
3. Monitor logs to ensure it's working
4. Adjust update interval based on your needs
5. Consider Vercel Pro if you want sub-15-minute updates

Your dashboard is now fully automated! 🎉
