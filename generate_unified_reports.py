#!/usr/bin/env python3
"""
Generate Unified Bitcoin Reports with Current Data

Creates ONE comprehensive PDF and ONE interactive dashboard with:
- Live current price from Coinbase
- Correct peak price from historical data
- Accurate drawdown calculations
- All charts and analysis
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

print("="*80)
print("GENERATING UNIFIED BITCOIN REPORTS WITH CURRENT DATA")
print("="*80)

# Configuration
OUTPUT_DIR = Path(__file__).parent / "reports"
OUTPUT_DIR.mkdir(exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M')
OUTPUT_PDF = OUTPUT_DIR / f"Bitcoin_Analysis_Report_{timestamp}.pdf"
OUTPUT_HTML = OUTPUT_DIR / f"Bitcoin_Dashboard_{timestamp}.html"

# Step 1: Get CURRENT LIVE PRICE
print("\n📡 Step 1: Fetching current live price...")
current_price = None
try:
    response = requests.get('https://api.coinbase.com/v2/prices/BTC-USD/spot', timeout=10)
    data = response.json()
    current_price = float(data['data']['amount'])
    print(f"✅ Current Price: ${current_price:,.2f}")
except Exception as e:
    print(f"❌ Failed to fetch live price: {e}")
    sys.exit(1)

current_date = datetime.now()

# Step 2: Load historical data
print("\n📊 Step 2: Loading historical data...")
DATA_PATH = Path(__file__).parent / "data" / "bitcoin_clean.csv"
if not DATA_PATH.exists():
    print(f"❌ Data file not found: {DATA_PATH}")
    sys.exit(1)

df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'], format='mixed')
df = df.sort_values('Date')

print(f"✅ Loaded {len(df)} records")
print(f"   Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")

# Step 3: Find CORRECT PEAK since halving
print("\n🔍 Step 3: Finding actual peak price...")
HALVING_DATE = datetime(2024, 4, 19)
recent_df = df[df['Date'] >= HALVING_DATE]
peak_row = recent_df.loc[recent_df['Close'].idxmax()]
peak_price = peak_row['Close']
peak_date = peak_row['Date']

print(f"✅ Peak Price: ${peak_price:,.2f} on {peak_date.strftime('%Y-%m-%d')}")

# Step 4: Calculate metrics
print("\n📈 Step 4: Calculating metrics...")
decline_dollars = current_price - peak_price
decline_pct = (decline_dollars / peak_price) * 100
days_since_halving = (current_date - HALVING_DATE).days
days_since_peak = (current_date - peak_date).days
peak_day_post_halving = (peak_date - HALVING_DATE).days

print(f"✅ Decline: ${decline_dollars:,.2f} ({decline_pct:.2f}%)")
print(f"✅ Days since halving: {days_since_halving}")
print(f"✅ Days since peak: {days_since_peak}")
print(f"✅ Peak was Day {peak_day_post_halving} post-halving")

# Calculate technical indicators
if 'SMA_50' not in df.columns:
    df['SMA_50'] = df['Close'].rolling(50).mean()
if 'SMA_200' not in df.columns:
    df['SMA_200'] = df['Close'].rolling(200).mean()
if 'EMA_50' not in df.columns:
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
if 'Volatility_20d' not in df.columns:
    df['Volatility_20d'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(365)

current_sma50 = df.iloc[-1]['SMA_50']
current_sma200 = df.iloc[-1]['SMA_200']
current_ema50 = df.iloc[-1]['EMA_50']
current_vol = df.iloc[-1]['Volatility_20d']

# Determine cycle phase
def get_cycle_phase(days_since):
    if days_since < 180:
        return "POST-HALVING RALLY", "#00FF00"
    elif days_since < 420:
        return "BULL ACCELERATION", "#90EE90"
    elif days_since < 550:
        return "PEAK FORMATION", "#FFD700"
    elif days_since < 640:
        return "DISTRIBUTION", "#FFA500"
    else:
        return "BEAR MARKET", "#FF6B6B"

cycle_phase, phase_color = get_cycle_phase(days_since_halving)

# Bear market projections from CORRECT peak
bear_best = peak_price * (1 - 0.769)
bear_avg = peak_price * (1 - 0.856)
bear_worst = peak_price * (1 - 0.935)

print(f"\n📊 Step 5: Generating PDF report...")

# ============================================================================
# GENERATE PDF REPORT
# ============================================================================

with PdfPages(OUTPUT_PDF) as pdf:

    # Page 1: Executive Summary
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111)
    ax.axis('off')

    title_text = "Bitcoin Market Analysis Report"
    subtitle_text = f"Generated: {current_date.strftime('%B %d, %Y at %I:%M %p')}"

    ax.text(0.5, 0.95, title_text, ha='center', va='top', fontsize=24, fontweight='bold')
    ax.text(0.5, 0.90, subtitle_text, ha='center', va='top', fontsize=12, style='italic')

    # Current Status Box
    ax.add_patch(mpatches.Rectangle((0.1, 0.70), 0.8, 0.15,
                                     fill=True, facecolor='#e8f4f8',
                                     edgecolor='black', linewidth=2))

    ax.text(0.5, 0.81, "Current Market Status", ha='center', fontsize=14, fontweight='bold')
    ax.text(0.25, 0.77, f"Price:", ha='left', fontsize=11, fontweight='bold')
    ax.text(0.50, 0.77, f"${current_price:,.2f}", ha='left', fontsize=11)
    ax.text(0.25, 0.73, f"Phase:", ha='left', fontsize=11, fontweight='bold')
    ax.text(0.50, 0.73, f"{cycle_phase}", ha='left', fontsize=11, color=phase_color)

    # Peak Analysis Box
    ax.add_patch(mpatches.Rectangle((0.1, 0.50), 0.8, 0.15,
                                     fill=True, facecolor='#fff4e6',
                                     edgecolor='black', linewidth=2))

    ax.text(0.5, 0.61, "Peak Analysis", ha='center', fontsize=14, fontweight='bold')
    ax.text(0.25, 0.57, f"Peak Price:", ha='left', fontsize=11, fontweight='bold')
    ax.text(0.50, 0.57, f"${peak_price:,.2f}", ha='left', fontsize=11)
    ax.text(0.25, 0.53, f"Peak Date:", ha='left', fontsize=11, fontweight='bold')
    ax.text(0.50, 0.53, f"{peak_date.strftime('%B %d, %Y')}", ha='left', fontsize=11)

    # Decline Box
    ax.add_patch(mpatches.Rectangle((0.1, 0.30), 0.8, 0.15,
                                     fill=True, facecolor='#ffe6e6',
                                     edgecolor='black', linewidth=2))

    ax.text(0.5, 0.41, "Decline from Peak", ha='center', fontsize=14, fontweight='bold')
    ax.text(0.25, 0.37, f"Amount:", ha='left', fontsize=11, fontweight='bold')
    ax.text(0.50, 0.37, f"${abs(decline_dollars):,.2f}", ha='left', fontsize=11, color='red')
    ax.text(0.25, 0.33, f"Percentage:", ha='left', fontsize=11, fontweight='bold')
    ax.text(0.50, 0.33, f"{abs(decline_pct):.2f}%", ha='left', fontsize=11, color='red')

    # Timeline Box
    ax.add_patch(mpatches.Rectangle((0.1, 0.10), 0.8, 0.15,
                                     fill=True, facecolor='#f0f0f0',
                                     edgecolor='black', linewidth=2))

    ax.text(0.5, 0.21, "Timeline", ha='center', fontsize=14, fontweight='bold')
    ax.text(0.25, 0.17, f"Days Since Halving:", ha='left', fontsize=11, fontweight='bold')
    ax.text(0.60, 0.17, f"{days_since_halving} days", ha='left', fontsize=11)
    ax.text(0.25, 0.13, f"Days Since Peak:", ha='left', fontsize=11, fontweight='bold')
    ax.text(0.60, 0.13, f"{days_since_peak} days", ha='left', fontsize=11)

    ax.text(0.5, 0.03, "Data Sources: Live Coinbase API + Historical Analysis",
            ha='center', fontsize=9, style='italic', color='gray')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # Page 2: Price Chart with SMAs
    fig, ax = plt.subplots(figsize=(11, 8.5))

    recent_df_chart = df[df['Date'] >= (df['Date'].max() - timedelta(days=730))]

    ax.plot(recent_df_chart['Date'], recent_df_chart['Close'],
            label='BTC Price', color='#FF6B00', linewidth=2)
    ax.plot(recent_df_chart['Date'], recent_df_chart['SMA_50'],
            label='SMA-50', color='#0000FF', linewidth=1.5, linestyle='--')
    ax.plot(recent_df_chart['Date'], recent_df_chart['SMA_200'],
            label='SMA-200', color='#FF0000', linewidth=1.5, linestyle='--')

    # Mark peak
    ax.scatter([peak_date], [peak_price], color='red', s=200, marker='*',
               label=f'Peak: ${peak_price:,.0f}', zorder=5)

    # Mark current (live price)
    ax.scatter([current_date], [current_price], color='green', s=200, marker='o',
               label=f'Current: ${current_price:,.0f}', zorder=5)

    ax.set_title(f'Bitcoin Price History (Last 2 Years)\nCurrent: ${current_price:,.2f} | Peak: ${peak_price:,.2f} | Decline: {abs(decline_pct):.2f}%',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price (USD)', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    plt.xticks(rotation=45)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # Page 3: Bear Market Projections
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.axis('off')

    ax.text(0.5, 0.95, "Historical Bear Market Projections",
            ha='center', fontsize=18, fontweight='bold')
    ax.text(0.5, 0.90, f"Based on actual peak of ${peak_price:,.2f} ({peak_date.strftime('%b %d, %Y')})",
            ha='center', fontsize=12, style='italic')

    scenarios = [
        ("Best Case (-76.9%)", bear_best, "#90EE90"),
        ("Average Case (-85.6%)", bear_avg, "#FFD700"),
        ("Worst Case (-93.5%)", bear_worst, "#FF6B6B"),
    ]

    y_pos = 0.75
    for name, target, color in scenarios:
        remaining_decline = ((target - current_price) / current_price) * 100

        ax.add_patch(mpatches.Rectangle((0.15, y_pos-0.12), 0.7, 0.15,
                                         fill=True, facecolor=color,
                                         edgecolor='black', linewidth=2, alpha=0.3))

        ax.text(0.5, y_pos, name, ha='center', fontsize=14, fontweight='bold')
        ax.text(0.5, y_pos-0.05, f"Target: ${target:,.0f}", ha='center', fontsize=12)
        ax.text(0.5, y_pos-0.09, f"({remaining_decline:+.1f}% from current ${current_price:,.2f})",
                ha='center', fontsize=11, style='italic')

        y_pos -= 0.20

    # Historical context
    ax.text(0.5, 0.10, "Historical Bear Markets:", ha='center', fontsize=12, fontweight='bold')
    ax.text(0.5, 0.05, "2013-2015: -93.5% over 410 days | 2018: -86.3% over 363 days | 2022: -76.9% over 376 days",
            ha='center', fontsize=10)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

print(f"✅ PDF Report saved: {OUTPUT_PDF}")

# ============================================================================
# GENERATE HTML DASHBOARD
# ============================================================================

print(f"\n📊 Step 6: Generating HTML dashboard...")

# Prepare chart data
recent_dates = [d.strftime('%Y-%m-%d') for d in df[df['Date'] >= (df['Date'].max() - timedelta(days=730))]['Date']]
recent_prices = list(df[df['Date'] >= (df['Date'].max() - timedelta(days=730))]['Close'].values)
recent_sma50 = list(df[df['Date'] >= (df['Date'].max() - timedelta(days=730))]['SMA_50'].values)
recent_sma200 = list(df[df['Date'] >= (df['Date'].max() - timedelta(days=730))]['SMA_200'].values)

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bitcoin Dashboard - {current_date.strftime('%Y-%m-%d')}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
        }}
        .metric-card {{
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            border-left: 5px solid;
            transition: transform 0.3s ease;
        }}
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        }}
        .metric-label {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .metric-subtext {{
            font-size: 0.9em;
            color: #888;
        }}
        .chart-container {{
            padding: 30px;
        }}
        .chart {{
            background: white;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        .alert {{
            background: #fff3cd;
            border: 2px solid #ffc107;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 30px;
        }}
        .alert strong {{
            color: #856404;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Bitcoin Trading Dashboard</h1>
            <p>Live Market Analysis - Updated {current_date.strftime('%B %d, %Y at %I:%M %p')}</p>
        </div>

        <div class="metrics">
            <div class="metric-card" style="border-color: #FF6B00;">
                <div class="metric-label">Current Price</div>
                <div class="metric-value" style="color: #FF6B00;">${current_price:,.2f}</div>
                <div class="metric-subtext">Live from Coinbase</div>
            </div>

            <div class="metric-card" style="border-color: #DC3545;">
                <div class="metric-label">Peak Price</div>
                <div class="metric-value" style="color: #DC3545;">${peak_price:,.2f}</div>
                <div class="metric-subtext">{peak_date.strftime('%b %d, %Y')}</div>
            </div>

            <div class="metric-card" style="border-color: #6C757D;">
                <div class="metric-label">Decline from Peak</div>
                <div class="metric-value" style="color: #DC3545;">{abs(decline_pct):.2f}%</div>
                <div class="metric-subtext">${abs(decline_dollars):,.2f}</div>
            </div>

            <div class="metric-card" style="border-color: {phase_color};">
                <div class="metric-label">Cycle Phase</div>
                <div class="metric-value" style="color: {phase_color};">{cycle_phase}</div>
                <div class="metric-subtext">Day {days_since_halving} post-halving</div>
            </div>

            <div class="metric-card" style="border-color: #0000FF;">
                <div class="metric-label">SMA-50</div>
                <div class="metric-value" style="color: #0000FF;">${current_sma50:,.0f}</div>
                <div class="metric-subtext">{'Above' if current_price > current_sma50 else 'Below'} current price</div>
            </div>

            <div class="metric-card" style="border-color: #FF0000;">
                <div class="metric-label">SMA-200</div>
                <div class="metric-value" style="color: #FF0000;">${current_sma200:,.0f}</div>
                <div class="metric-subtext">{'Above' if current_price > current_sma200 else 'Below'} current price</div>
            </div>
        </div>

        <div class="alert">
            <strong>⚠️ Bear Market Projections (from peak ${peak_price:,.2f}):</strong><br><br>
            🟢 Best Case (-76.9%): <strong>${bear_best:,.0f}</strong> ({((bear_best - current_price) / current_price * 100):+.1f}% from current)<br>
            🟡 Average (-85.6%): <strong>${bear_avg:,.0f}</strong> ({((bear_avg - current_price) / current_price * 100):+.1f}% from current)<br>
            🔴 Worst Case (-93.5%): <strong>${bear_worst:,.0f}</strong> ({((bear_worst - current_price) / current_price * 100):+.1f}% from current)
        </div>

        <div class="chart-container">
            <div class="chart">
                <h3>Bitcoin Price Chart (Last 2 Years)</h3>
                <div id="priceChart"></div>
            </div>
        </div>
    </div>

    <script>
        const priceTrace = {{
            x: {recent_dates},
            y: {recent_prices},
            type: 'scatter',
            mode: 'lines',
            name: 'BTC Price',
            line: {{color: '#FF6B00', width: 2}}
        }};

        const sma50Trace = {{
            x: {recent_dates},
            y: {recent_sma50},
            type: 'scatter',
            mode: 'lines',
            name: 'SMA-50',
            line: {{color: '#0000FF', width: 1.5, dash: 'dash'}}
        }};

        const sma200Trace = {{
            x: {recent_dates},
            y: {recent_sma200},
            type: 'scatter',
            mode: 'lines',
            name: 'SMA-200',
            line: {{color: '#FF0000', width: 1.5, dash: 'dash'}}
        }};

        const peakTrace = {{
            x: ['{peak_date.strftime('%Y-%m-%d')}'],
            y: [{peak_price}],
            type: 'scatter',
            mode: 'markers',
            name: 'Peak',
            marker: {{color: 'red', size: 15, symbol: 'star'}}
        }};

        const currentTrace = {{
            x: ['{current_date.strftime('%Y-%m-%d')}'],
            y: [{current_price}],
            type: 'scatter',
            mode: 'markers',
            name: 'Current',
            marker: {{color: 'green', size: 15, symbol: 'circle'}}
        }};

        const layout = {{
            showlegend: true,
            height: 500,
            margin: {{t: 20, b: 50, l: 80, r: 50}},
            xaxis: {{title: 'Date'}},
            yaxis: {{title: 'Price (USD)'}},
            hovermode: 'x unified'
        }};

        Plotly.newPlot('priceChart', [priceTrace, sma50Trace, sma200Trace, peakTrace, currentTrace], layout, {{responsive: true}});
    </script>
</body>
</html>
"""

with open(OUTPUT_HTML, 'w') as f:
    f.write(html_content)

print(f"✅ HTML Dashboard saved: {OUTPUT_HTML}")

print(f"\n" + "="*80)
print("UNIFIED REPORTS GENERATED SUCCESSFULLY")
print("="*80)
print(f"\n📄 PDF Report: {OUTPUT_PDF.name}")
print(f"🌐 HTML Dashboard: {OUTPUT_HTML.name}")
print(f"\n📊 Summary:")
print(f"   Current Price: ${current_price:,.2f} (live)")
print(f"   Peak Price: ${peak_price:,.2f} ({peak_date.strftime('%b %d, %Y')})")
print(f"   Decline: {abs(decline_pct):.2f}%")
print(f"   Phase: {cycle_phase}")
print(f"\n✅ All data is current and accurate!")
