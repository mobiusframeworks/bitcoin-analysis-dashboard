# Vercel Deployment Guide for Bitcoin Analysis Dashboard

## Quick Start - Deploy Now!

### Method 1: One-Command Deployment (Recommended)

```bash
cd "/Users/alexhorton/quant connect dev environment/ml_pipeline"
./deploy_to_vercel.sh
```

This script will:
1. Regenerate the dashboard with latest data
2. Copy it to `index.html`
3. Deploy to Vercel

### Method 2: Manual Deployment

```bash
cd "/Users/alexhorton/quant connect dev environment/ml_pipeline/reports"
vercel --prod
```

## First-Time Setup

When you run the deployment for the first time, Vercel will ask you a few questions:

1. **"Set up and deploy?"** → Yes
2. **"Which scope?"** → Select your personal account or team
3. **"Link to existing project?"** → No (unless you already created one)
4. **"What's your project's name?"** → `bitcoin-analysis-dashboard` (or your preferred name)
5. **"In which directory is your code located?"** → `.` (current directory)
6. **"Want to override settings?"** → No

After answering these questions, Vercel will:
- Upload your files
- Deploy your site
- Give you a live URL (e.g., `https://bitcoin-analysis-dashboard.vercel.app`)

## Updating Your Deployed Site

After the first deployment, simply run:

```bash
./deploy_to_vercel.sh
```

Or manually:

```bash
# 1. Regenerate with latest data
python3 generate_comprehensive_bitcoin_website.py

# 2. Copy to index.html
cp reports/Bitcoin_Comprehensive_Dashboard.html reports/index.html

# 3. Deploy
cd reports
vercel --prod
```

## What Gets Deployed

The deployment includes:
- `index.html` - Your main dashboard (943 KB with embedded charts)
- `Bitcoin_Comprehensive_Dashboard.html` - Backup copy
- `ML_Analysis_Report.html` - ML analysis report
- `Lead_Lag_Cointegration_Report.html` - Lead-lag analysis
- `Reports_Index.html` - Reports index page
- `*.json` - Data files for reference

Files excluded (via `.vercelignore`):
- PDF reports
- Old dashboard versions
- Documentation files

## Custom Domain (Optional)

To use your own domain:

1. Go to https://vercel.com/dashboard
2. Select your project
3. Go to Settings → Domains
4. Add your custom domain
5. Follow the DNS configuration instructions

## Environment Variables (If Needed)

If you later add server-side features, you can set environment variables:

1. Go to your project in Vercel Dashboard
2. Settings → Environment Variables
3. Add variables as needed

## Automatic Deployments (Optional)

To enable automatic deployments when you push to GitHub:

1. Push your `ml_pipeline/reports` directory to a GitHub repository
2. Import the project in Vercel from GitHub
3. Vercel will automatically deploy on every push to main branch

### Quick GitHub Setup

```bash
cd "/Users/alexhorton/quant connect dev environment/ml_pipeline/reports"

# Initialize git (if not already done)
git init

# Create .gitignore
echo "*.pdf" > .gitignore
echo "Bitcoin_Dashboard_*.html" >> .gitignore

# Add files
git add .
git commit -m "Initial commit - Bitcoin Analysis Dashboard"

# Connect to GitHub (create repo first at github.com)
git remote add origin https://github.com/YOUR_USERNAME/bitcoin-dashboard.git
git push -u origin main
```

Then import the repository in Vercel dashboard.

## Troubleshooting

### "Command not found: vercel"
Install Vercel CLI:
```bash
npm install -g vercel
```

### "File size too large"
Your HTML file (943 KB) should be fine. Vercel supports files up to 100 MB.

### "Deployment failed"
Check the deployment logs in Vercel Dashboard for specific errors.

### Need to redeploy without changes?
```bash
cd reports
vercel --prod --force
```

## Monitoring & Analytics

View your deployment metrics:
- Visit https://vercel.com/dashboard
- Select your project
- See visitor stats, performance metrics, and deployment history

## Security Notes

- Your dashboard is publicly accessible (no authentication required)
- All data is embedded in the HTML (no backend)
- Consider adding password protection if needed (Vercel supports this)

## Next Steps

After deployment:
1. Save your deployment URL
2. Set up a cron job to auto-update (see below)
3. Share your dashboard!

### Auto-Update with Cron (Optional)

To automatically update and redeploy daily:

```bash
# Edit crontab
crontab -e

# Add this line (runs at 2 AM daily):
0 2 * * * cd "/Users/alexhorton/quant connect dev environment/ml_pipeline" && ./deploy_to_vercel.sh >> /tmp/bitcoin-deploy.log 2>&1
```

## Support

- Vercel Documentation: https://vercel.com/docs
- Vercel CLI Reference: https://vercel.com/docs/cli
- Community Support: https://github.com/vercel/vercel/discussions
