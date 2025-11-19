#!/bin/bash

# Script to get Vercel credentials for GitHub Actions

echo "======================================================"
echo "Vercel Credentials for GitHub Actions Setup"
echo "======================================================"
echo ""

# Get Vercel token
echo "1️⃣  VERCEL_TOKEN"
echo "-------------------"
echo "You need to create a Vercel token at:"
echo "https://vercel.com/account/tokens"
echo ""
echo "Steps:"
echo "  a) Visit the URL above"
echo "  b) Click 'Create Token'"
echo "  c) Name it: 'GitHub Actions'"
echo "  d) Set expiration: No Expiration"
echo "  e) Copy the token (you'll only see it once!)"
echo ""
read -p "Press ENTER when you have your token..."
echo ""

# Get project info from .vercel/project.json
echo "2️⃣  VERCEL_ORG_ID and VERCEL_PROJECT_ID"
echo "-----------------------------------------"

if [ -f "reports/.vercel/project.json" ]; then
    echo "✅ Found project.json"
    echo ""

    ORG_ID=$(cat reports/.vercel/project.json | grep -o '"orgId":"[^"]*"' | cut -d'"' -f4)
    PROJECT_ID=$(cat reports/.vercel/project.json | grep -o '"projectId":"[^"]*"' | cut -d'"' -f4)

    echo "Your credentials:"
    echo ""
    echo "VERCEL_ORG_ID:     $ORG_ID"
    echo "VERCEL_PROJECT_ID: $PROJECT_ID"
    echo ""
else
    echo "❌ Error: reports/.vercel/project.json not found"
    echo ""
    echo "Run 'vercel --prod --yes' first in the reports/ directory"
    exit 1
fi

# Summary
echo "======================================================"
echo "GitHub Secrets to Add"
echo "======================================================"
echo ""
echo "Go to your GitHub repository:"
echo "Settings → Secrets and variables → Actions → New repository secret"
echo ""
echo "Add these 3 secrets:"
echo ""
echo "Name: VERCEL_TOKEN"
echo "Value: [paste your token from step 1]"
echo ""
echo "Name: VERCEL_ORG_ID"
echo "Value: $ORG_ID"
echo ""
echo "Name: VERCEL_PROJECT_ID"
echo "Value: $PROJECT_ID"
echo ""
echo "======================================================"
echo ""
echo "💡 TIP: Save these values in a secure place (like 1Password)"
echo ""
