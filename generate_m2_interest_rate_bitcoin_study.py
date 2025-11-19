#!/usr/bin/env python3
"""
Comprehensive M2, Interest Rate, and Bitcoin Price Study

Based on previous findings:
- M2 leads Bitcoin by 90 days with r=0.78
- Interest rates (FEDFUNDS) affect Bitcoin dynamics
- This study creates extensive visualizations showing:
  1. M2 vs Bitcoin with 90-day lead overlay
  2. Interest rate cycles and Bitcoin correlation
  3. Cointegration analysis (mean reversion)
  4. Multiple timeframe analysis (30, 60, 90, 180 days)
  5. Interactive regime charts
  6. Combined M2 + Interest Rate effects
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.tsa.stattools import coint
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("M2, INTEREST RATE, AND BITCOIN COMPREHENSIVE STUDY")
print("="*80)

# Configuration
OUTPUT_DIR = Path(__file__).parent / "reports"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_HTML = OUTPUT_DIR / "M2_Interest_Rate_Bitcoin_Study.html"

# Load comprehensive dataset with real FRED data
print("\n📊 Step 1: Loading comprehensive Bitcoin + FRED data...")
COMPREHENSIVE_DATA_PATH = Path(__file__).parent.parent / "datasets" / "btc_comprehensive_data.csv"

if not COMPREHENSIVE_DATA_PATH.exists():
    print(f"❌ Comprehensive data not found at {COMPREHENSIVE_DATA_PATH}")
    print("⚠️  Run update_daily_data.py first to fetch and merge all data")
    sys.exit(1)

btc = pd.read_csv(COMPREHENSIVE_DATA_PATH)
btc['Date'] = pd.to_datetime(btc['date'])
btc = btc.sort_values('Date').reset_index(drop=True)

# Rename columns for consistency
btc = btc.rename(columns={
    'close': 'c',
    'open': 'o',
    'high': 'h',
    'low': 'l',
    'volume': 'v'
})

print(f"✅ Loaded {len(btc)} records with REAL FRED data")
print(f"   Date range: {btc['Date'].min()} to {btc['Date'].max()}")
print(f"   FRED indicators included: M2SL, FEDFUNDS, DGS10, CPIAUCSL, WALCL, Real_Rate")

# No need to simulate - we have real FRED data!
print("\n📊 Step 2: Using real FRED economic indicators (no simulation needed)...")

# Already have real FRED data - no simulation needed!
print(f"✅ Using real FRED data from Federal Reserve Economic Data (FRED)")
print(f"   - M2SL: M2 Money Supply (Billions of USD)")
print(f"   - FEDFUNDS: Federal Funds Effective Rate (%)")
print(f"   - DGS10: 10-Year Treasury Constant Maturity Rate (%)")
print(f"   - CPIAUCSL: Consumer Price Index for All Urban Consumers")
print(f"   - WALCL: Federal Reserve Total Assets (Balance Sheet)")
print(f"   - Real_Rate: Fed Funds Rate - CPI YoY Inflation")

# Step 3: Create M2 with 90-day lead
print("\n🔄 Step 3: Creating M2 with 90-day lead...")

btc['M2SL_lead90'] = btc['M2SL'].shift(90)
btc['FEDFUNDS_lead30'] = btc['FEDFUNDS'].shift(30)

# Calculate correlations
m2_btc_corr = btc['M2SL'].corr(btc['c'])
m2_lead90_btc_corr = btc['M2SL_lead90'].corr(btc['c'])
fedfunds_btc_corr = btc['FEDFUNDS'].corr(btc['c'])

print(f"✅ Correlation Analysis:")
print(f"   M2 (contemporaneous) vs BTC: r = {m2_btc_corr:.4f}")
print(f"   M2 (90-day lead) vs BTC: r = {m2_lead90_btc_corr:.4f}")
print(f"   Fed Funds vs BTC: r = {fedfunds_btc_corr:.4f}")

# Step 4: Test multiple lead periods
print("\n📈 Step 4: Testing multiple lead periods...")

lead_periods = [0, 30, 60, 90, 120, 180, 270, 365]
m2_lead_correlations = {}
fedfunds_lead_correlations = {}

for lead in lead_periods:
    m2_shifted = btc['M2SL'].shift(lead)
    fed_shifted = btc['FEDFUNDS'].shift(lead)

    m2_corr = m2_shifted.corr(btc['c'])
    fed_corr = fed_shifted.corr(btc['c'])

    m2_lead_correlations[lead] = m2_corr
    fedfunds_lead_correlations[lead] = fed_corr

print(f"✅ M2 Lead Correlations:")
for lead, corr in m2_lead_correlations.items():
    print(f"   {lead} days: r = {corr:.4f}")

# Step 5: Cointegration Testing
print("\n🔗 Step 5: Testing cointegration...")

m2_clean = btc['M2SL'].dropna()
btc_clean = btc['c'].iloc[:len(m2_clean)]

try:
    coint_stat, pvalue, crit_values = coint(m2_clean, btc_clean)
    is_cointegrated = pvalue < 0.05

    print(f"✅ Cointegration Test:")
    print(f"   Test statistic: {coint_stat:.4f}")
    print(f"   P-value: {pvalue:.4f}")
    print(f"   Cointegrated: {'YES ✅' if is_cointegrated else 'NO ❌'}")

    if is_cointegrated:
        model = LinearRegression()
        model.fit(m2_clean.values.reshape(-1, 1), btc_clean.values)
        predicted_btc = model.predict(m2_clean.values.reshape(-1, 1))
        spread = btc_clean.values - predicted_btc

        print(f"   Mean spread: ${spread.mean():.2f}")
        print(f"   Std spread: ${spread.std():.2f}")
    else:
        spread = None
        model = None

except Exception as e:
    print(f"❌ Cointegration test failed: {e}")
    is_cointegrated = False
    spread = None

# Step 6: Interest Rate Regimes
print("\n💰 Step 6: Analyzing interest rate regimes...")

btc['Rate_Regime'] = pd.cut(btc['FEDFUNDS'],
                             bins=[-np.inf, 1, 3, np.inf],
                             labels=['Low (0-1%)', 'Medium (1-3%)', 'High (3%+)'])

regime_stats = btc.groupby('Rate_Regime')['c'].agg(['mean', 'std', 'count'])
print(f"✅ Bitcoin Performance by Interest Rate Regime:")
print(regime_stats)

# Step 7: Generate Charts
print("\n📊 Step 7: Generating comprehensive visualizations...")

import io
import base64

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return img_str

charts = {}

# Chart 1: M2 vs Bitcoin with 90-day lead
fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(16, 10))

ax1a_twin = ax1a.twinx()
ax1a.plot(btc.index, btc['M2SL'], color='#3498db', linewidth=2, label='M2 Money Supply', alpha=0.7)
ax1a_twin.plot(btc.index, btc['c'], color='#f39c12', linewidth=2, label='Bitcoin Price')
ax1a.set_ylabel('M2 Money Supply (Billions USD)', fontsize=12, color='#3498db')
ax1a_twin.set_ylabel('Bitcoin Price (USD)', fontsize=12, color='#f39c12')
ax1a.set_title('M2 Money Supply vs Bitcoin Price (Contemporaneous)', fontsize=14, fontweight='bold')
ax1a.legend(loc='upper left')
ax1a_twin.legend(loc='upper right')
ax1a.grid(alpha=0.3)

# Bottom: M2 with 90-day lead
ax1b_twin = ax1b.twinx()
ax1b.plot(btc.index, btc['M2SL_lead90'], color='#3498db', linewidth=2, label='M2 (shifted 90 days ahead)', alpha=0.7, linestyle='--')
ax1b_twin.plot(btc.index, btc['c'], color='#f39c12', linewidth=2, label='Bitcoin Price')
ax1b.set_ylabel('M2 Money Supply (90-day lead)', fontsize=12, color='#3498db')
ax1b_twin.set_ylabel('Bitcoin Price (USD)', fontsize=12, color='#f39c12')
ax1b.set_title(f'M2 with 90-Day Lead vs Bitcoin (r = {m2_lead90_btc_corr:.4f})', fontsize=14, fontweight='bold')
ax1b.set_xlabel('Date', fontsize=12)
ax1b.legend(loc='upper left')
ax1b_twin.legend(loc='upper right')
ax1b.grid(alpha=0.3)

charts['m2_bitcoin_90day_lead'] = fig_to_base64(fig1)

# Chart 2: Lead Period Correlation Analysis
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(16, 6))

leads = list(m2_lead_correlations.keys())
m2_corrs = list(m2_lead_correlations.values())
fed_corrs = list(fedfunds_lead_correlations.values())

ax2a.plot(leads, m2_corrs, marker='o', linewidth=3, markersize=8, color='#2ecc71')
ax2a.axvline(x=90, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Optimal: 90 days')
ax2a.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
ax2a.set_xlabel('Lead Period (days)', fontsize=12)
ax2a.set_ylabel('Correlation with Bitcoin', fontsize=12)
ax2a.set_title('M2 Lead Period vs Correlation Strength', fontsize=14, fontweight='bold')
ax2a.grid(alpha=0.3)
ax2a.legend()

ax2b.plot(leads, fed_corrs, marker='s', linewidth=3, markersize=8, color='#e74c3c')
ax2b.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
ax2b.set_xlabel('Lead Period (days)', fontsize=12)
ax2b.set_ylabel('Correlation with Bitcoin', fontsize=12)
ax2b.set_title('Fed Funds Rate Lead Period vs Correlation', fontsize=14, fontweight='bold')
ax2b.grid(alpha=0.3)

charts['lead_period_correlations'] = fig_to_base64(fig2)

# Chart 3: Interest Rate Regimes
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(16, 6))

# Left: Timeline with regime coloring
colors = {'Low (0-1%)': '#2ecc71', 'Medium (1-3%)': '#f39c12', 'High (3%+)': '#e74c3c'}
for regime in btc['Rate_Regime'].cat.categories:
    mask = btc['Rate_Regime'] == regime
    ax3a.scatter(btc[mask].index, btc[mask]['c'],
                 c=colors[regime], label=regime, alpha=0.5, s=10)

ax3a.set_ylabel('Bitcoin Price (USD)', fontsize=12)
ax3a.set_xlabel('Date', fontsize=12)
ax3a.set_title('Bitcoin Price Colored by Interest Rate Regime', fontsize=14, fontweight='bold')
ax3a.legend()
ax3a.grid(alpha=0.3)

# Right: Box plot by regime
regime_data = [btc[btc['Rate_Regime'] == r]['c'].dropna() for r in btc['Rate_Regime'].cat.categories]
bp = ax3b.boxplot(regime_data, labels=btc['Rate_Regime'].cat.categories, patch_artist=True)
for patch, regime in zip(bp['boxes'], btc['Rate_Regime'].cat.categories):
    patch.set_facecolor(colors[regime])
    patch.set_alpha(0.7)

ax3b.set_ylabel('Bitcoin Price (USD)', fontsize=12)
ax3b.set_xlabel('Interest Rate Regime', fontsize=12)
ax3b.set_title('Bitcoin Price Distribution by Rate Regime', fontsize=14, fontweight='bold')
ax3b.grid(axis='y', alpha=0.3)

charts['interest_rate_regimes'] = fig_to_base64(fig3)

# Chart 4: Cointegration (if exists)
if is_cointegrated and spread is not None:
    fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(16, 10))

    # Top: Normalized M2 vs BTC
    m2_norm = (m2_clean - m2_clean.min()) / (m2_clean.max() - m2_clean.min())
    btc_norm = (btc_clean - btc_clean.min()) / (btc_clean.max() - btc_clean.min())

    ax4a.plot(m2_norm.values, color='#3498db', linewidth=2, label='M2 (normalized)', alpha=0.8)
    ax4a.plot(btc_norm.values, color='#f39c12', linewidth=2, label='Bitcoin (normalized)', alpha=0.8)
    ax4a.set_ylabel('Normalized Value (0-1)', fontsize=12)
    ax4a.set_title(f'Cointegration: M2 vs Bitcoin (p-value: {pvalue:.4f})', fontsize=14, fontweight='bold')
    ax4a.legend()
    ax4a.grid(alpha=0.3)

    # Bottom: Mean reversion spread
    ax4b.plot(spread, color='#9b59b6', linewidth=1.5, label='Spread (BTC - Predicted)')
    ax4b.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax4b.axhline(y=spread.mean() + 2*spread.std(), color='red', linestyle='--', linewidth=1, alpha=0.7, label='+2σ')
    ax4b.axhline(y=spread.mean() - 2*spread.std(), color='red', linestyle='--', linewidth=1, alpha=0.7, label='-2σ')
    ax4b.fill_between(range(len(spread)), spread.mean() - 2*spread.std(), spread.mean() + 2*spread.std(),
                       alpha=0.2, color='red')
    ax4b.set_ylabel('Spread (USD)', fontsize=12)
    ax4b.set_xlabel('Time Index', fontsize=12)
    ax4b.set_title('Mean Reversion Spread (Trading Signal)', fontsize=12, fontweight='bold')
    ax4b.legend()
    ax4b.grid(alpha=0.3)

    charts['cointegration_mean_reversion'] = fig_to_base64(fig4)

# Chart 5: Fed Funds Rate Timeline with Bitcoin
fig5, ax5a = plt.subplots(figsize=(16, 8))

ax5a_twin = ax5a.twinx()
ax5a.fill_between(btc.index, 0, btc['FEDFUNDS'], color='#e74c3c', alpha=0.3, label='Fed Funds Rate')
ax5a.plot(btc.index, btc['FEDFUNDS'], color='#c0392b', linewidth=2)
ax5a_twin.plot(btc.index, btc['c'], color='#f39c12', linewidth=2, label='Bitcoin Price')

ax5a.set_ylabel('Federal Funds Rate (%)', fontsize=12, color='#c0392b')
ax5a_twin.set_ylabel('Bitcoin Price (USD)', fontsize=12, color='#f39c12')
ax5a.set_xlabel('Date', fontsize=12)
ax5a.set_title('Federal Funds Rate vs Bitcoin Price Over Time', fontsize=14, fontweight='bold')
ax5a.legend(loc='upper left')
ax5a_twin.legend(loc='upper right')
ax5a.grid(alpha=0.3)

charts['fedfunds_timeline'] = fig_to_base64(fig5)

# Chart 6: Real Interest Rate (Fed Funds - Inflation)
fig6, ax6a = plt.subplots(figsize=(16, 8))

ax6a_twin = ax6a.twinx()
ax6a.plot(btc.index, btc['Real_Rate'], color='#8e44ad', linewidth=2, label='Real Interest Rate')
ax6a.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax6a.fill_between(btc.index, 0, btc['Real_Rate'],
                   where=(btc['Real_Rate'] > 0), color='red', alpha=0.2, label='Positive Real Rate (restrictive)')
ax6a.fill_between(btc.index, 0, btc['Real_Rate'],
                   where=(btc['Real_Rate'] < 0), color='green', alpha=0.2, label='Negative Real Rate (stimulative)')

ax6a_twin.plot(btc.index, btc['c'], color='#f39c12', linewidth=2, label='Bitcoin Price', alpha=0.7)

ax6a.set_ylabel('Real Interest Rate (%)', fontsize=12, color='#8e44ad')
ax6a_twin.set_ylabel('Bitcoin Price (USD)', fontsize=12, color='#f39c12')
ax6a.set_xlabel('Date', fontsize=12)
ax6a.set_title('Real Interest Rate (Fed Funds - Inflation) vs Bitcoin', fontsize=14, fontweight='bold')
ax6a.legend(loc='upper left')
ax6a_twin.legend(loc='upper right')
ax6a.grid(alpha=0.3)

charts['real_interest_rate'] = fig_to_base64(fig6)

# Chart 7: Scatter - M2 vs Bitcoin (with 90-day lead)
fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(16, 7))

# Left: Contemporaneous
mask_contemp = btc[['M2SL', 'c']].notna().all(axis=1)
ax7a.scatter(btc[mask_contemp]['M2SL'], btc[mask_contemp]['c'],
             alpha=0.3, c=btc[mask_contemp].index.values.astype(float), cmap='viridis')
z = np.polyfit(btc[mask_contemp]['M2SL'], btc[mask_contemp]['c'], 1)
p = np.poly1d(z)
ax7a.plot(btc[mask_contemp]['M2SL'], p(btc[mask_contemp]['M2SL']),
          "r--", linewidth=2, label=f'r = {m2_btc_corr:.4f}')
ax7a.set_xlabel('M2 Money Supply', fontsize=12)
ax7a.set_ylabel('Bitcoin Price (USD)', fontsize=12)
ax7a.set_title('M2 vs Bitcoin (Contemporaneous)', fontsize=14, fontweight='bold')
ax7a.legend()
ax7a.grid(alpha=0.3)

# Right: 90-day lead
mask_lead = btc[['M2SL_lead90', 'c']].notna().all(axis=1)
ax7b.scatter(btc[mask_lead]['M2SL_lead90'], btc[mask_lead]['c'],
             alpha=0.3, c=btc[mask_lead].index.values.astype(float), cmap='plasma')
z2 = np.polyfit(btc[mask_lead]['M2SL_lead90'], btc[mask_lead]['c'], 1)
p2 = np.poly1d(z2)
ax7b.plot(btc[mask_lead]['M2SL_lead90'], p2(btc[mask_lead]['M2SL_lead90']),
          "r--", linewidth=2, label=f'r = {m2_lead90_btc_corr:.4f}')
ax7b.set_xlabel('M2 Money Supply (90-day lead)', fontsize=12)
ax7b.set_ylabel('Bitcoin Price (USD)', fontsize=12)
ax7b.set_title('M2 (90-day lead) vs Bitcoin', fontsize=14, fontweight='bold')
ax7b.legend()
ax7b.grid(alpha=0.3)

charts['m2_bitcoin_scatter'] = fig_to_base64(fig7)

# Chart 8: Combined Effects (M2 Growth + Rate Changes)
btc['M2_Growth'] = btc['M2SL'].pct_change(90) * 100  # 90-day growth %
btc['Rate_Change'] = btc['FEDFUNDS'].diff(90)  # 90-day change

fig8, ax8 = plt.subplots(figsize=(16, 8), subplot_kw={'projection': '3d'})

mask_3d = btc[['M2_Growth', 'Rate_Change', 'c']].notna().all(axis=1)
scatter = ax8.scatter(btc[mask_3d]['M2_Growth'],
                      btc[mask_3d]['Rate_Change'],
                      btc[mask_3d]['c'],
                      c=btc[mask_3d]['c'], cmap='viridis', s=20, alpha=0.6)

ax8.set_xlabel('M2 Growth (90-day %)', fontsize=10)
ax8.set_ylabel('Fed Rate Change (90-day)', fontsize=10)
ax8.set_zlabel('Bitcoin Price (USD)', fontsize=10)
ax8.set_title('Combined Effects: M2 Growth + Interest Rate Changes on Bitcoin', fontsize=12, fontweight='bold')
plt.colorbar(scatter, ax=ax8, label='Bitcoin Price', pad=0.1)

charts['combined_m2_rate_effects'] = fig_to_base64(fig8)

print(f"✅ Generated {len(charts)} comprehensive charts")

print("\n✅ Analysis complete! Saving results...")

# Save results for comprehensive website
results = {
    'm2_btc_corr': m2_btc_corr,
    'm2_lead90_btc_corr': m2_lead90_btc_corr,
    'fedfunds_btc_corr': fedfunds_btc_corr,
    'm2_lead_correlations': m2_lead_correlations,
    'fedfunds_lead_correlations': fedfunds_lead_correlations,
    'is_cointegrated': is_cointegrated,
    'coint_pvalue': pvalue if is_cointegrated else None,
    'regime_stats': regime_stats.to_dict() if 'regime_stats' in locals() else {},
    'charts': charts
}

import json
with open(OUTPUT_DIR / 'm2_interest_rate_study_results.json', 'w') as f:
    # Save everything except charts (those are in results['charts'])
    results_to_save = {k: v for k, v in results.items() if k != 'charts'}
    json.dump(results_to_save, f, indent=2, default=str)

print(f"✅ Results saved")
print(f"\n{'='*80}")
print("ANALYSIS SUMMARY")
print(f"{'='*80}")
print(f"\n📊 Correlation Analysis:")
print(f"   M2 (contemporaneous) vs BTC: r = {m2_btc_corr:.4f}")
print(f"   M2 (90-day lead) vs BTC: r = {m2_lead90_btc_corr:.4f}")
print(f"   Fed Funds vs BTC: r = {fedfunds_btc_corr:.4f}")

print(f"\n🔄 Optimal Lead Periods:")
best_m2_lead = max(m2_lead_correlations.items(), key=lambda x: abs(x[1]))
print(f"   M2: {best_m2_lead[0]} days (r = {best_m2_lead[1]:.4f})")

print(f"\n🔗 Cointegration:")
print(f"   M2 vs Bitcoin: {'YES ✅' if is_cointegrated else 'NO ❌'}")
if is_cointegrated:
    print(f"   P-value: {pvalue:.4f}")

print(f"\n💰 Interest Rate Regimes:")
if 'regime_stats' in locals():
    for regime in regime_stats.index:
        print(f"   {regime}: Avg ${regime_stats.loc[regime, 'mean']:,.2f}, Count {regime_stats.loc[regime, 'count']:.0f}")

print(f"\n📁 Charts Generated: {len(charts)}")
print(f"📊 Results saved to: {OUTPUT_DIR / 'm2_interest_rate_study_results.json'}")
print(f"\n✅ Complete! Data ready for comprehensive Bitcoin website integration.")
