# 🚀 GitHub Actions Auto-Update Setup

## Overview

This setup creates a **fully automated cloud-based system** that:

1. ✅ **Runs every 10 minutes** in GitHub's cloud (not your computer!)
2. ✅ **Fetches latest Bitcoin data** from APIs
3. ✅ **Regenerates all analyses** (M2, trading strategy, ML predictions)
4. ✅ **Commits data to GitHub** (version controlled, tracked)
5. ✅ **Auto-deploys to Vercel** (your live website updates automatically)

**Best part:** Completely free and runs 24/7 without your computer!

---

## 📋 Prerequisites

- GitHub account (free)
- Vercel account (free)
- Your Bitcoin analysis code

---

## 🔧 Step-by-Step Setup

### Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `bitcoin-analysis-dashboard`
3. Description: "Automated Bitcoin analysis dashboard with ML predictions"
4. **Make it Public** (or Private if you prefer)
5. ✅ Click "Create repository"

### Step 2: Initialize Git in Your Project

```bash
cd "/Users/alexhorton/quant connect dev environment/ml_pipeline"

# Initialize git
git init

# Add .gitignore
git add .gitignore

# Add all files
git add .

# First commit
git commit -m "Initial commit: Bitcoin Analysis Dashboard"

# Link to your GitHub repo (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/bitcoin-analysis-dashboard.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Step 3: Get Vercel Credentials

Run this script to get your credentials:

```bash
chmod +x get_vercel_credentials.sh
./get_vercel_credentials.sh
```

This will give you three values:
- `VERCEL_TOKEN` (you need to create this)
- `VERCEL_ORG_ID` (auto-extracted)
- `VERCEL_PROJECT_ID` (auto-extracted)

**To create VERCEL_TOKEN:**

1. Visit https://vercel.com/account/tokens
2. Click "Create Token"
3. Name: `GitHub Actions`
4. Scope: Full Account
5. Expiration: No Expiration
6. **Copy the token** (you'll only see it once!)

### Step 4: Add Secrets to GitHub

1. Go to your GitHub repository
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **"New repository secret"**
4. Add these three secrets:

**Secret 1:**
```
Name: VERCEL_TOKEN
Value: [paste token from Vercel]
```

**Secret 2:**
```
Name: VERCEL_ORG_ID
Value: [from get_vercel_credentials.sh output]
```

**Secret 3:**
```
Name: VERCEL_PROJECT_ID
Value: [from get_vercel_credentials.sh output]
```

### Step 5: Push GitHub Actions Workflow

```bash
cd "/Users/alexhorton/quant connect dev environment/ml_pipeline"

# Add the workflow file
git add .github/workflows/update-dashboard.yml

# Commit
git commit -m "Add GitHub Actions auto-update workflow"

# Push
git push
```

### Step 6: Verify It's Working

1. Go to your GitHub repo
2. Click **Actions** tab
3. You should see the workflow running!
4. Click on the workflow run to see live logs

---

## 🎯 How It Works

### Automatic Schedule

The workflow runs **every 10 minutes** automatically:

```yaml
schedule:
  - cron: '*/10 * * * *'
```

### What Happens Each Run

```
[Every 10 minutes]
    ↓
1. Fetch latest Bitcoin data (yfinance API)
    ↓
2. Generate M2 analysis
    ↓
3. Generate trading strategy analysis
    ↓
4. Generate comprehensive dashboard
    ↓
5. Commit changes to GitHub
    ↓
6. Deploy to Vercel
    ↓
[Your website is updated! 🎉]
```

### Manual Trigger

You can also trigger updates manually:

1. Go to **Actions** tab
2. Click **"Update Bitcoin Dashboard"**
3. Click **"Run workflow"**
4. Click **"Run workflow"** button

---

## 📊 Data Storage Strategy

### What Gets Stored in GitHub

✅ **Committed to Git** (version controlled):
- `datasets/btc-ohlc.csv` - Bitcoin price history
- `ml_pipeline/reports/*.json` - Analysis results
- `ml_pipeline/reports/*.html` - Generated dashboards

❌ **Not committed** (.gitignore):
- PDFs
- Old dashboard versions
- Temporary files
- Virtual environment
- Logs

### Benefits

1. **Full history** - Every data update is tracked
2. **Rollback capability** - Can revert to any previous version
3. **Audit trail** - See when prices were fetched
4. **Collaboration** - Others can fork your analysis
5. **Free hosting** - GitHub stores it all for free

---

## ⚙️ Customizing Update Frequency

Edit `.github/workflows/update-dashboard.yml`:

```yaml
schedule:
  # Every 5 minutes (may hit API limits)
  - cron: '*/5 * * * *'

  # Every 10 minutes (recommended)
  - cron: '*/10 * * * *'

  # Every 15 minutes (safe for 24/7)
  - cron: '*/15 * * * *'

  # Every 30 minutes (very conservative)
  - cron: '*/30 * * * *'

  # Every hour
  - cron: '0 * * * *'
```

**Recommendation:** Start with 10 minutes, adjust based on your needs.

---

## 🔍 Monitoring

### View Workflow Runs

GitHub Actions Dashboard:
- https://github.com/YOUR_USERNAME/bitcoin-analysis-dashboard/actions

See:
- ✅ Successful runs (green checkmark)
- ❌ Failed runs (red X)
- ⏱️ Duration of each run
- 📊 Logs for debugging

### View Commits

See all automatic data updates:
- https://github.com/YOUR_USERNAME/bitcoin-analysis-dashboard/commits

### View Deployments

Vercel deployments:
- https://vercel.com/YOUR_USERNAME/bitcoin-analysis-dashboard

---

## 🛠️ Troubleshooting

### Workflow Not Running?

1. **Check GitHub Actions is enabled:**
   - Settings → Actions → General
   - Allow all actions

2. **Check schedule syntax:**
   - Visit https://crontab.guru to validate cron expression

3. **Check secrets are set:**
   - Settings → Secrets and variables → Actions
   - Verify all 3 secrets exist

### Data Not Updating?

1. **Check workflow logs:**
   - Actions tab → Latest run → View logs
   - Look for error messages

2. **Common issues:**
   - API rate limits (reduce frequency)
   - Invalid Vercel token (regenerate)
   - Git conflicts (fix manually)

### Deployment Failing?

1. **Verify Vercel credentials:**
   ```bash
   ./get_vercel_credentials.sh
   ```

2. **Check Vercel project exists:**
   - Visit https://vercel.com/dashboard
   - Ensure project is there

3. **Re-deploy manually:**
   ```bash
   cd reports
   vercel --prod --yes
   ```

---

## 💰 Cost & Limits

### GitHub Actions

**Free Tier:**
- ✅ 2,000 minutes/month for private repos
- ✅ Unlimited for public repos
- ✅ Each run takes ~3-5 minutes

**With 10-min intervals:**
- ~4,320 runs/month
- ~13-22 hours of runner time
- ✅ **Well within free tier limits!**

### Vercel

**Free Tier:**
- ✅ 100 GB bandwidth/month
- ⚠️ 100 deployments/day limit

**With 10-min intervals:**
- 144 deployments/day
- ❌ **Exceeds free tier!**

**Solutions:**
1. **Use 15-min intervals** (96 deployments/day = within limit)
2. **Upgrade to Vercel Pro** ($20/month = unlimited deployments)
3. **Deploy only on data changes** (modify workflow)

### Recommended Free Setup

**For completely free 24/7 operation:**

```yaml
# Change to 15-minute intervals
schedule:
  - cron: '*/15 * * * *'  # 96 deployments/day ✅
```

---

## 🎨 Architecture Diagram

```
┌─────────────────────────────────────────────┐
│         GitHub Actions (Cloud)              │
│                                             │
│  ┌──────────────────────────────────────┐  │
│  │  Every 10 Minutes:                   │  │
│  │                                      │  │
│  │  1. Fetch Bitcoin Data (yfinance)   │  │
│  │  2. Fetch M2 Data (FRED API)        │  │
│  │  3. Run ML Analysis                 │  │
│  │  4. Generate Trading Signals        │  │
│  │  5. Create Dashboard HTML           │  │
│  └──────────────────────────────────────┘  │
│                    ↓                        │
│  ┌──────────────────────────────────────┐  │
│  │  Commit to GitHub Repository         │  │
│  │  - datasets/btc-ohlc.csv            │  │
│  │  - reports/*.json                    │  │
│  │  - reports/*.html                    │  │
│  └──────────────────────────────────────┘  │
│                    ↓                        │
│  ┌──────────────────────────────────────┐  │
│  │  Deploy to Vercel                    │  │
│  │  - Upload static files               │  │
│  │  - Update production site            │  │
│  └──────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
                     ↓
         🌐 Live Dashboard Updated!
```

---

## 🚀 Advanced: Deploy Only on Changes

To avoid hitting Vercel's deployment limit, modify the workflow to deploy only when data actually changes:

```yaml
- name: Check for changes
  id: check_changes
  run: |
    if git diff --quiet HEAD; then
      echo "changed=false" >> $GITHUB_OUTPUT
    else
      echo "changed=true" >> $GITHUB_OUTPUT
    fi

- name: Deploy to Vercel
  if: steps.check_changes.outputs.changed == 'true'
  uses: amondnet/vercel-action@v25
  # ... rest of deployment config
```

---

## 🎯 Next Steps

### After Setup is Complete:

1. **Star your repo** ⭐ (it's your project!)
2. **Watch Actions run** 👀 (see it work in real-time)
3. **Share your dashboard** 🌐 (send the Vercel URL to friends)
4. **Monitor performance** 📊 (check GitHub Actions usage)
5. **Customize analysis** 🔧 (add more features!)

### Future Enhancements:

- 📧 Email alerts for significant price moves
- 📱 Mobile-responsive improvements
- 🤖 Twitter bot for sharing insights
- 📈 More technical indicators
- 🔔 Discord/Slack notifications
- 💾 Historical data export to CSV
- 🎨 Dark mode toggle
- 🌍 Multiple cryptocurrency support

---

## 📖 Additional Resources

- **GitHub Actions Docs:** https://docs.github.com/en/actions
- **Vercel Docs:** https://vercel.com/docs
- **Cron Syntax:** https://crontab.guru
- **Your Workflow Runs:** https://github.com/YOUR_USERNAME/bitcoin-analysis-dashboard/actions

---

## ✅ Checklist

Before going live, verify:

- [ ] GitHub repository created
- [ ] Code pushed to GitHub
- [ ] Vercel token created
- [ ] All 3 secrets added to GitHub
- [ ] Workflow file pushed
- [ ] First workflow run successful
- [ ] Dashboard deployed to Vercel
- [ ] Auto-refresh working
- [ ] Bitcoin logo showing
- [ ] Data updating automatically

---

## 🎉 Congratulations!

Your Bitcoin Analysis Dashboard is now **fully automated** and running in the cloud!

- ✅ No computer needed
- ✅ Updates every 10-15 minutes
- ✅ Completely free
- ✅ Professional and reliable
- ✅ Version controlled with full history

**Your live dashboard:**
- Production: https://bitcoin-analysis-dashboard.vercel.app
- GitHub: https://github.com/YOUR_USERNAME/bitcoin-analysis-dashboard
- Actions: https://github.com/YOUR_USERNAME/bitcoin-analysis-dashboard/actions

Enjoy your automated Bitcoin analysis system! 🚀📊₿
