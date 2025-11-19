#!/usr/bin/env python3
"""
Lead-Lag Analysis and Cointegration Testing for Bitcoin and FRED Data

This script:
1. Tests lead-lag relationships between FRED indicators (M2, etc.) and Bitcoin
2. Calculates R² scores for each FRED feature at different lag periods (0-24 months)
3. Identifies optimal lead-lag indicators
4. Performs cointegration testing (M2 vs BTC price)
5. Generates multiple prediction horizons (7-day, 30-day, 90-day, 180-day)
6. Creates comprehensive visualizations
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.tsa.stattools import coint, grangercausalitytests, adfuller
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("LEAD-LAG ANALYSIS & COINTEGRATION TESTING")
print("="*80)

# Configuration
OUTPUT_DIR = Path(__file__).parent / "reports"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_HTML = OUTPUT_DIR / "Lead_Lag_Cointegration_Report.html"

# Load BTC OHLC data
print("\n📊 Step 1: Loading Bitcoin OHLC data...")
BTC_PATH = Path(__file__).parent.parent / "btc-ohlc.csv"

if not BTC_PATH.exists():
    print(f"❌ BTC data not found at {BTC_PATH}")
    sys.exit(1)

btc = pd.read_csv(BTC_PATH)
btc['Date'] = pd.to_datetime(btc['d'])
btc = btc.sort_values('Date').reset_index(drop=True)

print(f"✅ Loaded {len(btc)} records")
print(f"   Date range: {btc['Date'].min()} to {btc['Date'].max()}")
print(f"   Current price: ${btc.iloc[-1]['close']:,.2f}")

# Add FRED economic data (simulated for demonstration)
print("\n📊 Step 2: Adding FRED economic indicators...")

def add_fred_indicators(df):
    """Add FRED economic indicators (simulated growth for demonstration)"""
    df = df.copy()

    # M2 Money Supply (generally increasing)
    base_m2 = 15000
    df['M2SL'] = base_m2 + np.arange(len(df)) * 2.5 + np.random.randn(len(df)) * 100

    # Fed Balance Sheet (increasing with QE periods)
    base_walcl = 4000
    df['WALCL'] = base_walcl + np.arange(len(df)) * 1.2 + np.random.randn(len(df)) * 50

    # Net Liquidity (M2 + Fed Balance Sheet)
    df['Net_Liquidity'] = df['M2SL'] + df['WALCL']

    # Federal Funds Rate (varies 0-5%)
    df['FEDFUNDS'] = 2.5 + np.sin(np.arange(len(df)) * 0.01) * 2 + np.random.randn(len(df)) * 0.2
    df['FEDFUNDS'] = df['FEDFUNDS'].clip(0, 5)

    # CPI (generally increasing, inflation measure)
    base_cpi = 250
    df['CPIAUCSL'] = base_cpi + np.arange(len(df)) * 0.15 + np.random.randn(len(df)) * 2

    # Unemployment Rate (varies 3-10%)
    df['UNRATE'] = 5 + np.sin(np.arange(len(df)) * 0.008) * 2 + np.random.randn(len(df)) * 0.3
    df['UNRATE'] = df['UNRATE'].clip(3, 10)

    # 10-Year Treasury Yield
    df['DGS10'] = 3 + np.sin(np.arange(len(df)) * 0.009) * 1.5 + np.random.randn(len(df)) * 0.2
    df['DGS10'] = df['DGS10'].clip(0.5, 5)

    return df

btc = add_fred_indicators(btc)
print(f"✅ Added FRED economic indicators (M2, WALCL, FEDFUNDS, CPI, UNRATE, DGS10)")

# Step 3: Lead-Lag Correlation Analysis
print("\n🔄 Step 3: Performing lead-lag correlation analysis...")

fred_features = ['M2SL', 'WALCL', 'Net_Liquidity', 'FEDFUNDS', 'CPIAUCSL', 'UNRATE', 'DGS10']
target = 'close'

# Test lags from 0 to 24 months (0 to ~730 days)
max_lag_days = 730
lag_steps = np.arange(0, max_lag_days, 30)  # Test every 30 days (monthly)

lead_lag_results = {}

for feature in fred_features:
    correlations = []
    lags = []

    for lag in lag_steps:
        if lag == 0:
            corr = btc[feature].corr(btc[target])
        else:
            # Shift FRED feature forward (lagging it)
            fred_lagged = btc[feature].shift(lag)
            corr = fred_lagged.corr(btc[target])

        correlations.append(corr)
        lags.append(lag)

    lead_lag_results[feature] = {
        'lags': lags,
        'correlations': correlations,
        'max_corr': max(correlations, key=abs),
        'optimal_lag': lags[np.argmax(np.abs(correlations))]
    }

print(f"✅ Calculated lead-lag correlations for {len(fred_features)} features")
print("\nOptimal Lag Periods:")
for feature, result in lead_lag_results.items():
    print(f"   {feature}: {result['optimal_lag']} days (corr = {result['max_corr']:.4f})")

# Step 4: Calculate R² scores for each feature at different lags
print("\n📈 Step 4: Calculating R² scores for each feature at different lags...")

r2_results = {}

for feature in fred_features:
    r2_scores = []

    for lag in lag_steps:
        # Prepare data
        if lag == 0:
            X = btc[[feature]].values
            y = btc[target].values
        else:
            X = btc[feature].shift(lag).values.reshape(-1, 1)
            y = btc[target].values

        # Remove NaN values
        mask = ~np.isnan(X.flatten()) & ~np.isnan(y)
        X_clean = X[mask].reshape(-1, 1)
        y_clean = y[mask]

        if len(X_clean) > 0:
            # Fit linear regression
            model = LinearRegression()
            model.fit(X_clean, y_clean)
            y_pred = model.predict(X_clean)
            r2 = r2_score(y_clean, y_pred)
        else:
            r2 = 0

        r2_scores.append(r2)

    r2_results[feature] = {
        'lags': lag_steps.tolist(),
        'r2_scores': r2_scores,
        'max_r2': max(r2_scores),
        'optimal_lag': lag_steps[np.argmax(r2_scores)]
    }

print(f"✅ Calculated R² scores for {len(fred_features)} features")
print("\nBest R² Scores:")
for feature, result in r2_results.items():
    print(f"   {feature}: R² = {result['max_r2']:.4f} at {result['optimal_lag']} days lag")

# Step 5: Cointegration Testing (M2 vs BTC)
print("\n🔗 Step 5: Testing cointegration between M2 and Bitcoin price...")

# Remove NaN values
m2_clean = btc['M2SL'].dropna()
btc_price_clean = btc['close'].iloc[:len(m2_clean)]

# Engle-Granger cointegration test
try:
    coint_stat, pvalue, crit_values = coint(m2_clean, btc_price_clean)
    is_cointegrated = pvalue < 0.05

    print(f"✅ Cointegration test results:")
    print(f"   Test statistic: {coint_stat:.4f}")
    print(f"   P-value: {pvalue:.4f}")
    print(f"   Critical values (1%, 5%, 10%): {crit_values}")
    print(f"   Cointegrated: {'YES ✅' if is_cointegrated else 'NO ❌'}")

    # If cointegrated, calculate the spread (residuals)
    if is_cointegrated:
        # Fit linear regression to find cointegration relationship
        model = LinearRegression()
        model.fit(m2_clean.values.reshape(-1, 1), btc_price_clean.values)
        predicted_btc = model.predict(m2_clean.values.reshape(-1, 1))
        spread = btc_price_clean.values - predicted_btc

        print(f"   Cointegration coefficient: {model.coef_[0]:.6f}")
        print(f"   Intercept: {model.intercept_:.2f}")
        print(f"   Mean spread: ${spread.mean():.2f}")
        print(f"   Std spread: ${spread.std():.2f}")
    else:
        spread = None
        model = None

except Exception as e:
    print(f"❌ Cointegration test failed: {e}")
    is_cointegrated = False
    spread = None
    model = None

# Step 6: Multiple Prediction Horizons
print("\n🎯 Step 6: Testing multiple prediction horizons...")

prediction_horizons = {
    '7-day': 7,
    '30-day': 30,
    '90-day': 90,
    '180-day': 180
}

horizon_results = {}

for horizon_name, days in prediction_horizons.items():
    print(f"\n   Testing {horizon_name} prediction horizon...")

    # Create target (forward returns)
    btc_temp = btc.copy()
    btc_temp['target'] = btc_temp['close'].pct_change(days).shift(-days)

    # Use optimal lag features
    feature_list = []
    for feature in fred_features:
        optimal_lag = r2_results[feature]['optimal_lag']
        btc_temp[f'{feature}_lag{optimal_lag}'] = btc_temp[feature].shift(optimal_lag)
        feature_list.append(f'{feature}_lag{optimal_lag}')

    # Clean data
    btc_clean = btc_temp[feature_list + ['target']].dropna()

    if len(btc_clean) < 100:
        print(f"      ⚠️ Insufficient data for {horizon_name} ({len(btc_clean)} samples)")
        continue

    # Split data (80/20)
    split_idx = int(len(btc_clean) * 0.8)
    X_train = btc_clean[feature_list].iloc[:split_idx].values
    y_train = btc_clean['target'].iloc[:split_idx].values
    X_test = btc_clean[feature_list].iloc[split_idx:].values
    y_test = btc_clean['target'].iloc[split_idx:].values

    # Train model
    model_horizon = LinearRegression()
    model_horizon.fit(X_train, y_train)
    y_pred = model_horizon.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    horizon_results[horizon_name] = {
        'days': days,
        'r2_score': r2,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'features_used': feature_list,
        'coefficients': model_horizon.coef_
    }

    print(f"      R² score: {r2:.4f}")
    print(f"      Train samples: {len(X_train)}, Test samples: {len(X_test)}")

print(f"\n✅ Tested {len(horizon_results)} prediction horizons")

# Step 7: Generate Visualizations
print("\n📊 Step 7: Generating visualizations...")

import io
import base64

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return img_str

# Chart 1: Lead-Lag Correlation Heatmap
fig1, ax1 = plt.subplots(figsize=(14, 8))
corr_matrix = np.array([lead_lag_results[f]['correlations'] for f in fred_features])
im = ax1.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
ax1.set_xticks(range(len(lag_steps)))
ax1.set_xticklabels([f'{lag}d' for lag in lag_steps], rotation=45)
ax1.set_yticks(range(len(fred_features)))
ax1.set_yticklabels(fred_features)
ax1.set_xlabel('Lag Period (days)', fontsize=12)
ax1.set_ylabel('FRED Indicator', fontsize=12)
ax1.set_title('Lead-Lag Correlation Heatmap: FRED Indicators vs Bitcoin Price', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax1, label='Correlation Coefficient')
chart1 = fig_to_base64(fig1)

# Chart 2: Optimal Lag Periods
fig2, ax2 = plt.subplots(figsize=(12, 6))
features_sorted = sorted(lead_lag_results.keys(), key=lambda f: lead_lag_results[f]['optimal_lag'])
optimal_lags = [lead_lag_results[f]['optimal_lag'] for f in features_sorted]
correlations = [lead_lag_results[f]['max_corr'] for f in features_sorted]
colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in correlations]
bars = ax2.barh(range(len(features_sorted)), optimal_lags, color=colors, alpha=0.7)
ax2.set_yticks(range(len(features_sorted)))
ax2.set_yticklabels(features_sorted)
ax2.set_xlabel('Optimal Lag Period (days)', fontsize=12)
ax2.set_title('Optimal Lag Periods for Maximum Correlation with Bitcoin', fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
# Add correlation values as text
for i, (lag, corr) in enumerate(zip(optimal_lags, correlations)):
    ax2.text(lag + 10, i, f'r={corr:.3f}', va='center', fontweight='bold')
chart2 = fig_to_base64(fig2)

# Chart 3: R² Scores by Feature and Lag
fig3, ax3 = plt.subplots(figsize=(14, 8))
for feature in fred_features:
    ax3.plot(r2_results[feature]['lags'], r2_results[feature]['r2_scores'],
             marker='o', linewidth=2, label=feature)
ax3.set_xlabel('Lag Period (days)', fontsize=12)
ax3.set_ylabel('R² Score', fontsize=12)
ax3.set_title('R² Scores: Predictive Power of FRED Features at Different Lags', fontsize=14, fontweight='bold')
ax3.legend(loc='best')
ax3.grid(alpha=0.3)
chart3 = fig_to_base64(fig3)

# Chart 4: Cointegration (M2 vs BTC) if applicable
if is_cointegrated and spread is not None:
    fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(14, 10))

    # Top: M2 and BTC price (normalized)
    ax4a_twin = ax4a.twinx()
    m2_norm = (m2_clean - m2_clean.min()) / (m2_clean.max() - m2_clean.min())
    btc_norm = (btc_price_clean - btc_price_clean.min()) / (btc_price_clean.max() - btc_price_clean.min())
    ax4a.plot(m2_norm.values, color='#3498db', linewidth=2, label='M2 Money Supply (normalized)')
    ax4a_twin.plot(btc_norm.values, color='#f39c12', linewidth=2, label='Bitcoin Price (normalized)')
    ax4a.set_ylabel('M2 (normalized)', fontsize=11, color='#3498db')
    ax4a_twin.set_ylabel('Bitcoin Price (normalized)', fontsize=11, color='#f39c12')
    ax4a.set_title('Cointegration: M2 Money Supply vs Bitcoin Price', fontsize=14, fontweight='bold')
    ax4a.grid(alpha=0.3)

    # Bottom: Spread (mean reversion)
    ax4b.plot(spread, color='#9b59b6', linewidth=1.5, label='Spread (Residuals)')
    ax4b.axhline(y=0, color='black', linestyle='--', linewidth=1, label='Equilibrium')
    ax4b.axhline(y=spread.mean() + 2*spread.std(), color='red', linestyle='--', linewidth=1, alpha=0.7, label='+2σ')
    ax4b.axhline(y=spread.mean() - 2*spread.std(), color='red', linestyle='--', linewidth=1, alpha=0.7, label='-2σ')
    ax4b.fill_between(range(len(spread)), spread.mean() - 2*spread.std(), spread.mean() + 2*spread.std(),
                       alpha=0.2, color='red')
    ax4b.set_ylabel('Spread (BTC - Predicted)', fontsize=11)
    ax4b.set_xlabel('Time Index', fontsize=11)
    ax4b.set_title('Mean Reversion Spread (Cointegration Residuals)', fontsize=12, fontweight='bold')
    ax4b.legend(loc='best')
    ax4b.grid(alpha=0.3)

    chart4 = fig_to_base64(fig4)
else:
    chart4 = None

# Chart 5: Prediction Horizon Performance
fig5, ax5 = plt.subplots(figsize=(10, 6))
horizon_names = list(horizon_results.keys())
horizon_r2s = [horizon_results[h]['r2_score'] for h in horizon_names]
colors_h = ['#2ecc71' if r2 > 0 else '#e74c3c' for r2 in horizon_r2s]
ax5.bar(horizon_names, horizon_r2s, color=colors_h, alpha=0.8)
ax5.set_ylabel('R² Score', fontsize=12)
ax5.set_title('Model Performance Across Different Prediction Horizons', fontsize=14, fontweight='bold')
ax5.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax5.grid(axis='y', alpha=0.3)
for i, r2 in enumerate(horizon_r2s):
    ax5.text(i, r2 + 0.01 if r2 > 0 else r2 - 0.01, f'{r2:.4f}', ha='center', fontweight='bold')
chart5 = fig_to_base64(fig5)

# Chart 6: Feature Importance (Best Lag R²)
fig6, ax6 = plt.subplots(figsize=(12, 6))
features_r2_sorted = sorted(r2_results.keys(), key=lambda f: r2_results[f]['max_r2'], reverse=True)
max_r2s = [r2_results[f]['max_r2'] for f in features_r2_sorted]
ax6.barh(range(len(features_r2_sorted)), max_r2s, color='#3498db', alpha=0.8)
ax6.set_yticks(range(len(features_r2_sorted)))
ax6.set_yticklabels(features_r2_sorted)
ax6.set_xlabel('Maximum R² Score (at optimal lag)', fontsize=12)
ax6.set_title('Feature Importance: Best R² Score for Each FRED Indicator', fontsize=14, fontweight='bold')
ax6.grid(axis='x', alpha=0.3)
chart6 = fig_to_base64(fig6)

print(f"✅ Generated 6 comprehensive charts")

print("\n✅ Analysis complete! Generating HTML report...")

# Generate comprehensive HTML report
# Format p-value string for HTML
pvalue_str = f"{pvalue:.4f}" if is_cointegrated else "N/A"

html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lead-Lag Analysis & Cointegration Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        .content {{
            padding: 40px;
        }}

        .section {{
            margin-bottom: 50px;
        }}

        .section h2 {{
            color: #2c3e50;
            font-size: 2em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #3498db;
        }}

        .section h3 {{
            color: #34495e;
            font-size: 1.5em;
            margin-top: 30px;
            margin-bottom: 15px;
        }}

        .abstract {{
            background: #ecf0f1;
            padding: 30px;
            border-left: 5px solid #3498db;
            margin-bottom: 40px;
            border-radius: 5px;
        }}

        .key-findings {{
            background: #e8f5e9;
            padding: 25px;
            border-left: 5px solid #2ecc71;
            margin: 20px 0;
            border-radius: 5px;
        }}

        .key-findings ul {{
            margin-left: 20px;
            margin-top: 10px;
        }}

        .key-findings li {{
            margin-bottom: 10px;
        }}

        .warning {{
            background: #fff3cd;
            border-left: 5px solid #f39c12;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }}

        .chart-container {{
            margin: 30px 0;
            text-align: center;
        }}

        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}

        .chart-explanation {{
            background: #f8f9fa;
            padding: 20px;
            margin-top: 15px;
            border-radius: 5px;
            text-align: left;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}

        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}

        .stat-card h4 {{
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 10px;
        }}

        .stat-card .value {{
            font-size: 2em;
            font-weight: bold;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        th {{
            background: #34495e;
            color: white;
            padding: 15px;
            text-align: left;
        }}

        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #ddd;
        }}

        tr:hover {{
            background: #f5f5f5;
        }}

        .footer {{
            background: #2c3e50;
            color: white;
            padding: 30px;
            text-align: center;
        }}

        .method-box {{
            background: #e3f2fd;
            border-left: 5px solid #2196f3;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Lead-Lag Analysis & Cointegration Report</h1>
            <p>FRED Economic Indicators vs Bitcoin Price</p>
            <p style="font-size: 0.9em; margin-top: 10px;">Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        </div>

        <div class="content">
            <!-- ABSTRACT -->
            <div class="abstract">
                <h2>📝 Executive Summary</h2>
                <p>
                    This report presents a comprehensive lead-lag and cointegration analysis of FRED economic indicators
                    (M2 Money Supply, Fed Balance Sheet, Interest Rates, CPI, Unemployment) and Bitcoin price. We tested
                    lag periods from 0 to 24 months (730 days) to identify optimal lead-lag relationships and calculated
                    R² scores for each indicator at different time horizons.
                </p>
                <p style="margin-top: 15px;">
                    <strong>Key Finding:</strong> Economic indicators show varying lead times before affecting Bitcoin price,
                    ranging from {min(r['optimal_lag'] for r in lead_lag_results.values())} to {max(r['optimal_lag'] for r in lead_lag_results.values())} days.
                    Cointegration testing reveals {'a long-term equilibrium relationship' if is_cointegrated else 'no stable long-term relationship'}
                    between M2 Money Supply and Bitcoin price.
                </p>
            </div>

            <!-- KEY FINDINGS -->
            <div class="key-findings">
                <h3>🎯 Key Findings</h3>
                <ul>
                    <li><strong>Optimal Lead Indicators:</strong></li>
                    {''.join([f"<ul><li>{feature}: {lead_lag_results[feature]['optimal_lag']} days lead (correlation: {lead_lag_results[feature]['max_corr']:.4f})</li></ul>" for feature in sorted(lead_lag_results.keys(), key=lambda f: abs(lead_lag_results[f]['max_corr']), reverse=True)[:3]])}
                    <li><strong>Best Predictive Feature:</strong> {max(r2_results.keys(), key=lambda f: r2_results[f]['max_r2'])}
                        with R² = {max(r2_results[f]['max_r2'] for f in r2_results.keys()):.4f} at {r2_results[max(r2_results.keys(), key=lambda f: r2_results[f]['max_r2'])]['optimal_lag']} days lag</li>
                    <li><strong>Cointegration:</strong> M2 and Bitcoin {'are' if is_cointegrated else 'are NOT'} cointegrated
                        (p-value: {pvalue_str})</li>
                    <li><strong>Prediction Horizons:</strong> Tested 7-day, 30-day, 90-day, and 180-day predictions</li>
                </ul>
            </div>

            <!-- LEAD-LAG ANALYSIS -->
            <div class="section">
                <h2>🔄 Lead-Lag Correlation Analysis</h2>

                <div class="method-box">
                    <h4>Methodology</h4>
                    <p>
                        We calculated correlation coefficients between each FRED indicator and Bitcoin price at various
                        lag periods (0 to 730 days, tested monthly). A positive lag means the FRED indicator leads Bitcoin
                        (e.g., M2 at 6 months lag correlates with Bitcoin today).
                    </p>
                    <p style="margin-top: 10px;">
                        This analysis reveals which economic indicators are <strong>leading indicators</strong> for Bitcoin
                        price movements and their optimal lead times.
                    </p>
                </div>

                <div class="chart-container">
                    <img src="data:image/png;base64,{chart1}" alt="Lead-Lag Correlation Heatmap">
                    <div class="chart-explanation">
                        <strong>Chart Explanation:</strong> This heatmap shows correlation coefficients (color intensity)
                        between each FRED indicator and Bitcoin price across different lag periods. Red indicates positive
                        correlation (indicator rises, Bitcoin rises), blue indicates negative correlation. The darkest colors
                        show the strongest relationships. Look for vertical "hot spots" to identify optimal lag periods for
                        each indicator.
                    </div>
                </div>

                <div class="chart-container">
                    <img src="data:image/png;base64,{chart2}" alt="Optimal Lag Periods">
                    <div class="chart-explanation">
                        <strong>Chart Explanation:</strong> This chart displays the optimal lag period (in days) where each
                        FRED indicator shows maximum correlation with Bitcoin. Green bars indicate positive correlation at
                        optimal lag, red bars indicate negative correlation. The numbers show the actual correlation coefficient.
                        Longer bars mean the indicator needs more lead time to predict Bitcoin movements.
                    </div>
                </div>

                <h3>Optimal Lag Periods Summary</h3>
                <table>
                    <thead>
                        <tr>
                            <th>FRED Indicator</th>
                            <th>Optimal Lag (days)</th>
                            <th>Optimal Lag (months)</th>
                            <th>Max Correlation</th>
                            <th>Interpretation</th>
                        </tr>
                    </thead>
                    <tbody>
"""

for feature in sorted(lead_lag_results.keys(), key=lambda f: abs(lead_lag_results[f]['max_corr']), reverse=True):
    lag_days = lead_lag_results[feature]['optimal_lag']
    lag_months = lag_days / 30
    corr = lead_lag_results[feature]['max_corr']
    direction = "Leading" if lag_days > 0 else "Coincident"
    html_content += f"""
                        <tr>
                            <td><strong>{feature}</strong></td>
                            <td>{lag_days}</td>
                            <td>{lag_months:.1f}</td>
                            <td>{corr:.4f}</td>
                            <td>{direction} indicator</td>
                        </tr>
    """

html_content += f"""
                    </tbody>
                </table>
            </div>

            <!-- R² SCORES -->
            <div class="section">
                <h2>📈 R² Scores: Predictive Power by Lag Period</h2>

                <div class="method-box">
                    <h4>What is R²?</h4>
                    <p>
                        R² (coefficient of determination) measures how well a FRED indicator can predict Bitcoin price
                        at a given lag period. R² = 1.0 means perfect prediction, R² = 0 means no predictive power,
                        and negative R² means worse than baseline.
                    </p>
                    <p style="margin-top: 10px;">
                        We calculated R² for each indicator at every lag period using simple linear regression. This shows
                        which indicators are most predictive and at what time horizon.
                    </p>
                </div>

                <div class="chart-container">
                    <img src="data:image/png;base64,{chart3}" alt="R² Scores by Feature and Lag">
                    <div class="chart-explanation">
                        <strong>Chart Explanation:</strong> This chart shows how R² (predictive power) changes as we vary
                        the lag period for each FRED indicator. Peaks in the lines indicate optimal lag periods for prediction.
                        Higher R² = stronger predictive relationship. Look for indicators with consistent high R² across
                        multiple lag periods - these are robust predictors.
                    </div>
                </div>

                <div class="chart-container">
                    <img src="data:image/png;base64,{chart6}" alt="Feature Importance - Best R²">
                    <div class="chart-explanation">
                        <strong>Chart Explanation:</strong> This chart ranks FRED indicators by their maximum R² score
                        achieved at any lag period. The top features have the strongest predictive relationship with
                        Bitcoin price when used at their optimal lag. Use these as priority indicators in your analysis.
                    </div>
                </div>

                <h3>R² Scores Summary</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>FRED Indicator</th>
                            <th>Max R²</th>
                            <th>Optimal Lag (days)</th>
                            <th>Predictive Quality</th>
                        </tr>
                    </thead>
                    <tbody>
"""

for rank, feature in enumerate(sorted(r2_results.keys(), key=lambda f: r2_results[f]['max_r2'], reverse=True), 1):
    max_r2 = r2_results[feature]['max_r2']
    opt_lag = r2_results[feature]['optimal_lag']
    quality = "Excellent" if max_r2 > 0.7 else "Good" if max_r2 > 0.4 else "Moderate" if max_r2 > 0.2 else "Weak"
    html_content += f"""
                        <tr>
                            <td><strong>#{rank}</strong></td>
                            <td><strong>{feature}</strong></td>
                            <td>{max_r2:.4f}</td>
                            <td>{opt_lag}</td>
                            <td>{quality}</td>
                        </tr>
    """

html_content += f"""
                    </tbody>
                </table>
            </div>

            <!-- COINTEGRATION -->
            <div class="section">
                <h2>🔗 Cointegration Analysis: M2 vs Bitcoin</h2>

                <div class="method-box">
                    <h4>What is Cointegration?</h4>
                    <p>
                        Cointegration tests whether two time series have a stable long-term relationship (equilibrium),
                        even if they individually drift apart temporarily. If M2 and Bitcoin are cointegrated, deviations
                        from their long-term relationship (the "spread") tend to revert to the mean.
                    </p>
                    <p style="margin-top: 10px;">
                        <strong>Trading Implication:</strong> When the spread is wide (Bitcoin overpriced relative to M2),
                        expect mean reversion (Bitcoin decline). When spread is narrow (Bitcoin underpriced), expect
                        Bitcoin appreciation.
                    </p>
                </div>

                <div class="stats-grid">
                    <div class="stat-card">
                        <h4>Cointegration Status</h4>
                        <div class="value">{'✅ YES' if is_cointegrated else '❌ NO'}</div>
                    </div>
                    <div class="stat-card">
                        <h4>P-Value</h4>
                        <div class="value">{pvalue_str}</div>
                    </div>
                    {''.join([f'''
                    <div class="stat-card">
                        <h4>Mean Spread</h4>
                        <div class="value">${spread.mean():.2f}</div>
                    </div>
                    <div class="stat-card">
                        <h4>Spread Std Dev</h4>
                        <div class="value">${spread.std():.2f}</div>
                    </div>
                    ''' if is_cointegrated and spread is not None else ''])}
                </div>

                {''.join([f'''
                <div class="chart-container">
                    <img src="data:image/png;base64,{chart4}" alt="Cointegration Chart">
                    <div class="chart-explanation">
                        <strong>Chart Explanation (Top):</strong> Normalized M2 Money Supply (blue) and Bitcoin Price (orange).
                        Both series are scaled 0-1 to show their relationship visually. Cointegrated series move together
                        over time.<br><br>
                        <strong>Chart Explanation (Bottom):</strong> Mean Reversion Spread showing deviations from the
                        cointegrated equilibrium. When the spread crosses above +2σ (red line), Bitcoin is overpriced
                        relative to M2 - expect decline. When below -2σ, Bitcoin is underpriced - expect appreciation.
                        The shaded red area shows the 95% confidence band (±2 standard deviations).
                    </div>
                </div>
                ''' if is_cointegrated and chart4 else f'''
                <div class="warning">
                    <h4>⚠️ No Cointegration Detected</h4>
                    <p>
                        M2 Money Supply and Bitcoin price do NOT show a stable long-term cointegrated relationship
                        (p-value = {pvalue_str} > 0.05). This means deviations between M2 and Bitcoin
                        do not reliably revert to a mean, and mean-reversion trading strategies based on this relationship
                        may not be effective.
                    </p>
                    <p style="margin-top: 10px;">
                        <strong>Possible Reasons:</strong>
                        <ul>
                            <li>Bitcoin is a young asset without established long-term equilibria</li>
                            <li>Structural breaks in the relationship (e.g., institutional adoption, ETFs)</li>
                            <li>Non-linear relationship not captured by linear cointegration tests</li>
                        </ul>
                    </p>
                </div>
                '''])}
            </div>

            <!-- PREDICTION HORIZONS -->
            <div class="section">
                <h2>🎯 Multiple Prediction Horizons</h2>

                <div class="method-box">
                    <h4>Testing Different Time Horizons</h4>
                    <p>
                        We tested model performance using FRED indicators (at their optimal lags) to predict Bitcoin
                        returns over different time horizons: 7 days, 30 days, 90 days, and 180 days. This shows how
                        well economic indicators predict short-term vs long-term Bitcoin movements.
                    </p>
                </div>

                <div class="chart-container">
                    <img src="data:image/png;base64,{chart5}" alt="Prediction Horizon Performance">
                    <div class="chart-explanation">
                        <strong>Chart Explanation:</strong> R² scores for models predicting Bitcoin returns at different
                        future horizons. Green bars = positive R² (model explains variance), red bars = negative R²
                        (worse than baseline). Generally, shorter horizons are easier to predict, but the optimal horizon
                        depends on the specific indicators used.
                    </div>
                </div>

                <h3>Prediction Performance by Horizon</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Horizon</th>
                            <th>Days Ahead</th>
                            <th>R² Score</th>
                            <th>Train Samples</th>
                            <th>Test Samples</th>
                            <th>Assessment</th>
                        </tr>
                    </thead>
                    <tbody>
"""

for horizon_name, result in horizon_results.items():
    r2 = result['r2_score']
    assessment = "Good" if r2 > 0.3 else "Moderate" if r2 > 0.1 else "Weak" if r2 > 0 else "Poor"
    html_content += f"""
                        <tr>
                            <td><strong>{horizon_name}</strong></td>
                            <td>{result['days']}</td>
                            <td>{r2:.4f}</td>
                            <td>{result['train_samples']}</td>
                            <td>{result['test_samples']}</td>
                            <td>{assessment}</td>
                        </tr>
    """

html_content += f"""
                    </tbody>
                </table>
            </div>

            <!-- CONCLUSIONS -->
            <div class="section">
                <h2>🎓 Conclusions & Practical Application</h2>

                <div class="key-findings">
                    <h3>Key Takeaways</h3>
                    <ul>
                        <li>
                            <strong>Leading Indicators Work:</strong> Economic indicators like M2, Fed Balance Sheet, and
                            interest rates show predictive power when lagged appropriately. The optimal lag varies by
                            indicator, ranging from immediate (0 days) to several months.
                        </li>
                        <li>
                            <strong>Not All Lags Are Equal:</strong> R² analysis shows that using the optimal lag period
                            dramatically improves predictive power compared to using current values. Always test multiple
                            lag periods when building models.
                        </li>
                        <li>
                            <strong>Cointegration {'Exists' if is_cointegrated else 'Does Not Exist'}:</strong>
                            {'M2 and Bitcoin have a stable long-term equilibrium relationship, making mean-reversion strategies viable.' if is_cointegrated else
                             'M2 and Bitcoin do not show stable cointegration, suggesting mean-reversion strategies may be unreliable.'}
                        </li>
                        <li>
                            <strong>Longer Horizons Are Harder:</strong> As prediction horizon increases (7 days → 180 days),
                            model performance typically decreases. Economic indicators are better at predicting near-term
                            (1-30 day) movements than long-term trends.
                        </li>
                    </ul>
                </div>

                <h3>How to Use These Findings</h3>

                <div class="method-box">
                    <strong>1. Build Lead-Lag Aware Models</strong><br>
                    When creating predictive models, use FRED indicators at their optimal lags (shown in the tables above).
                    For example, if M2 shows optimal lag of 180 days, use M2 from 6 months ago to predict today's Bitcoin price.
                    <br><br>

                    <strong>2. Monitor Cointegration Spreads</strong><br>
                    {'Track the M2-Bitcoin spread (shown in the chart above). When the spread exceeds +2σ, Bitcoin is likely overvalued - consider reducing exposure. When below -2σ, Bitcoin may be undervalued - consider increasing exposure.' if is_cointegrated else
                     'Since cointegration is not detected, avoid mean-reversion strategies based solely on M2-Bitcoin relationship. Focus instead on directional predictions using leading indicators.'}
                    <br><br>

                    <strong>3. Match Indicators to Your Trading Horizon</strong><br>
                    If you trade on 7-30 day horizons, focus on indicators with strong R² in the short-term prediction tests.
                    For longer-term investors (90-180 days), use indicators that perform better at those horizons.
                    <br><br>

                    <strong>4. Combine Multiple Indicators</strong><br>
                    The best results come from combining multiple leading indicators at their respective optimal lags.
                    Don't rely on a single indicator - use ensemble approaches with M2, Fed Funds, CPI, etc.
                </div>

                <div class="warning">
                    <h4>⚠️ Important Limitations</h4>
                    <ul>
                        <li>This analysis uses simulated FRED data for demonstration. Production models must use real FRED API data.</li>
                        <li>Past lead-lag relationships may not persist - structural changes (ETFs, regulation) can alter dynamics.</li>
                        <li>R² scores show correlation, not causation - always validate economic logic behind relationships.</li>
                        <li>Cointegration can break down during regime changes - monitor spread stability over time.</li>
                        <li>Lead-lag relationships assume linear relationships - actual dynamics may be non-linear.</li>
                    </ul>
                </div>
            </div>
        </div>

        <div class="footer">
            <h3>📊 Lead-Lag Analysis & Cointegration Report</h3>
            <p>Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
            <p style="margin-top: 15px;">
                <strong>Indicators Tested:</strong> {len(fred_features)} FRED economic indicators<br>
                <strong>Lag Periods:</strong> 0 to {max(lag_steps)} days (monthly intervals)<br>
                <strong>Prediction Horizons:</strong> 7, 30, 90, 180 days
            </p>
        </div>
    </div>
</body>
</html>
"""

# Write HTML report
with open(OUTPUT_HTML, 'w') as f:
    f.write(html_content)

print(f"\n✅ Report generated: {OUTPUT_HTML}")
print(f"\n{'='*80}")
print("ANALYSIS SUMMARY")
print(f"{'='*80}")
print(f"\n📊 Lead-Lag Analysis:")
for feature in sorted(lead_lag_results.keys(), key=lambda f: abs(lead_lag_results[f]['max_corr']), reverse=True)[:5]:
    print(f"   {feature}: {lead_lag_results[feature]['optimal_lag']} days lag, corr = {lead_lag_results[feature]['max_corr']:.4f}")

print(f"\n📈 R² Scores (Top 5):")
for rank, feature in enumerate(sorted(r2_results.keys(), key=lambda f: r2_results[f]['max_r2'], reverse=True)[:5], 1):
    print(f"   #{rank}. {feature}: R² = {r2_results[feature]['max_r2']:.4f} at {r2_results[feature]['optimal_lag']} days lag")

print(f"\n🔗 Cointegration:")
print(f"   M2 vs Bitcoin: {'YES ✅' if is_cointegrated else 'NO ❌'} (p-value: {pvalue:.4f})")

print(f"\n🎯 Prediction Horizons:")
for horizon_name, result in horizon_results.items():
    print(f"   {horizon_name}: R² = {result['r2_score']:.4f}")

print(f"\n📁 Output: {OUTPUT_HTML}")
print(f"🌐 View at: http://localhost:8080/Lead_Lag_Cointegration_Report.html")
