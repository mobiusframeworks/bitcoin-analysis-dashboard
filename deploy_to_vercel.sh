#!/bin/bash

# Bitcoin Analysis Dashboard - Vercel Deployment Script
# This script regenerates the dashboard and deploys it to Vercel

set -e  # Exit on error

echo "=============================================="
echo "Bitcoin Analysis Dashboard - Vercel Deploy"
echo "=============================================="
echo ""

# Step 1: Activate virtual environment
echo "📦 Activating virtual environment..."
cd "$(dirname "$0")"
source ../venv/bin/activate

# Step 2: Regenerate the dashboard with latest data
echo ""
echo "🔄 Regenerating dashboard with latest data..."
python3 generate_comprehensive_bitcoin_website.py

# Step 3: Copy to index.html for Vercel
echo ""
echo "📋 Copying dashboard to index.html..."
cp reports/Bitcoin_Comprehensive_Dashboard.html reports/index.html

# Step 4: Deploy to Vercel
echo ""
echo "🚀 Deploying to Vercel..."
cd reports

# Check if this is the first deployment or an update
if [ -f ".vercel/project.json" ]; then
    echo "Deploying update to production..."
    vercel --prod
else
    echo "First time deployment..."
    echo "You'll need to:"
    echo "1. Link to your Vercel account"
    echo "2. Choose a project name"
    echo "3. Accept the defaults for directory settings"
    echo ""
    vercel --prod
fi

echo ""
echo "=============================================="
echo "✅ Deployment complete!"
echo "=============================================="
echo ""
echo "Your dashboard is now live on Vercel!"
echo "Visit your Vercel dashboard to see the URL: https://vercel.com/dashboard"
