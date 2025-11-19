#!/usr/bin/env python3
"""
50-Week SMA Trading Strategy Analysis
Analyzes price action around 50-week and 200-week SMAs across different market phases
Provides risk management guidelines based on historical volatility
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import io
import base64
import json

# Configuration
BASE_DIR = Path("/Users/alexhorton/quant connect dev environment")
DATA_FILE = BASE_DIR / "datasets" / "btc_comprehensive_data.csv"
OUTPUT_DIR = BASE_DIR / "ml_pipeline" / "reports"
OUTPUT_DIR.mkdir(exist_ok=True)

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

print("=" * 80)
print("50-WEEK SMA TRADING STRATEGY ANALYSIS")
print("=" * 80)
print()

# Load data
print("📊 Loading Bitcoin data...")
df = pd.read_csv(DATA_FILE)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)
print(f"✅ Loaded {len(df)} daily records")
print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
print()

# Convert to weekly data
print("📅 Converting to weekly timeframe...")
df.set_index('date', inplace=True)
weekly = df.resample('W').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()
weekly = weekly.rename(columns={'open': 'o', 'high': 'h', 'low': 'l', 'close': 'c', 'volume': 'v'})
weekly.reset_index(inplace=True)

print(f"✅ Created {len(weekly)} weekly candles")
print(f"   Weekly range: {weekly['date'].min()} to {weekly['date'].max()}")
print()

# Calculate SMAs
print("📈 Calculating moving averages...")
weekly['SMA_50'] = weekly['c'].rolling(window=50).mean()
weekly['SMA_200'] = weekly['c'].rolling(window=200).mean()

# Distance from SMAs (percentage)
weekly['Distance_from_50SMA_pct'] = ((weekly['c'] - weekly['SMA_50']) / weekly['SMA_50']) * 100
weekly['Distance_from_200SMA_pct'] = ((weekly['c'] - weekly['SMA_200']) / weekly['SMA_200']) * 100

# Position relative to SMAs
weekly['Above_50SMA'] = weekly['c'] > weekly['SMA_50']
weekly['Above_200SMA'] = weekly['c'] > weekly['SMA_200']

# Crossovers
weekly['50SMA_Cross'] = weekly['Above_50SMA'].astype(int).diff()
# 1 = bullish cross (below to above), -1 = bearish cross (above to below)

print(f"✅ Moving averages calculated")
print(f"   Current price: ${weekly['c'].iloc[-1]:,.2f}")
print(f"   50-week SMA: ${weekly['SMA_50'].iloc[-1]:,.2f}")
print(f"   200-week SMA: ${weekly['SMA_200'].iloc[-1]:,.2f}")
print(f"   Distance from 50-week SMA: {weekly['Distance_from_50SMA_pct'].iloc[-1]:.2f}%")
print(f"   Distance from 200-week SMA: {weekly['Distance_from_200SMA_pct'].iloc[-1]:.2f}%")
print()

# Identify market phases based on position and trend
print("📊 Identifying market phases...")

def identify_market_phase(row, df, idx):
    """Classify market phase: bull, distribution, bear, accumulation"""
    if pd.isna(row['SMA_50']) or pd.isna(row['SMA_200']):
        return 'unknown'

    price = row['c']
    sma_50 = row['SMA_50']
    sma_200 = row['SMA_200']

    # Calculate 12-week price momentum
    if idx >= 12:
        momentum = (price - df.iloc[idx-12]['c']) / df.iloc[idx-12]['c']
    else:
        momentum = 0

    # Bull: Above both SMAs, positive momentum
    if price > sma_50 and price > sma_200 and sma_50 > sma_200 and momentum > 0:
        return 'bull'
    # Distribution: Above SMAs but weakening momentum
    elif price > sma_50 and momentum < 0:
        return 'distribution'
    # Bear: Below both SMAs, negative momentum
    elif price < sma_50 and price < sma_200 and sma_50 < sma_200 and momentum < 0:
        return 'bear'
    # Accumulation: Below SMAs but improving momentum
    elif price < sma_50 and momentum > 0:
        return 'accumulation'
    else:
        return 'transition'

weekly['Market_Phase'] = weekly.apply(lambda row: identify_market_phase(row, weekly, row.name), axis=1)

phase_counts = weekly['Market_Phase'].value_counts()
print(f"✅ Market phases identified:")
for phase, count in phase_counts.items():
    print(f"   {phase.capitalize()}: {count} weeks ({count/len(weekly)*100:.1f}%)")

current_phase = weekly['Market_Phase'].iloc[-1]
print(f"\n   🎯 Current Phase: {current_phase.upper()}")
print()

# Calculate volatility around 50-week SMA for each phase
print("📊 Analyzing volatility by market phase...")

volatility_by_phase = {}
for phase in ['bull', 'distribution', 'bear', 'accumulation']:
    phase_data = weekly[weekly['Market_Phase'] == phase]['Distance_from_50SMA_pct'].dropna()
    if len(phase_data) > 0:
        volatility_by_phase[phase] = {
            'mean': phase_data.mean(),
            'std': phase_data.std(),
            'median': phase_data.median(),
            'q25': phase_data.quantile(0.25),
            'q75': phase_data.quantile(0.75),
            'min': phase_data.min(),
            'max': phase_data.max(),
            'count': len(phase_data)
        }
        print(f"   {phase.capitalize()}:")
        print(f"     Mean distance: {volatility_by_phase[phase]['mean']:.2f}%")
        print(f"     Std deviation: {volatility_by_phase[phase]['std']:.2f}%")
        print(f"     Range: {volatility_by_phase[phase]['min']:.2f}% to {volatility_by_phase[phase]['max']:.2f}%")

print()

# Calculate risk management levels for current phase
print("🎯 Calculating risk management levels...")

current_price = weekly['c'].iloc[-1]
current_50sma = weekly['SMA_50'].iloc[-1]
current_200sma = weekly['SMA_200'].iloc[-1]
current_distance = weekly['Distance_from_50SMA_pct'].iloc[-1]

# Get current phase stats
if current_phase in volatility_by_phase:
    phase_stats = volatility_by_phase[current_phase]

    # Stop loss based on 2 standard deviations
    stop_loss_pct = phase_stats['std'] * 2

    if current_price > current_50sma:
        # Long position
        stop_loss_price = current_50sma * (1 - stop_loss_pct / 100)
        position = "LONG"
    else:
        # Short position
        stop_loss_price = current_50sma * (1 + stop_loss_pct / 100)
        position = "SHORT"

    # Risk levels
    conservative_stop = current_50sma  # 50-week SMA
    moderate_stop = stop_loss_price
    aggressive_stop = current_200sma  # 200-week SMA

    print(f"   Current Position: {position}")
    print(f"   Current Price: ${current_price:,.2f}")
    print(f"   50-week SMA: ${current_50sma:,.2f}")
    print(f"   200-week SMA: ${current_200sma:,.2f}")
    print(f"\n   Stop Loss Recommendations:")
    print(f"   📍 Conservative (50-week SMA): ${conservative_stop:,.2f} ({((conservative_stop - current_price) / current_price * 100):+.2f}%)")
    print(f"   📍 Moderate (2σ from SMA): ${moderate_stop:,.2f} ({((moderate_stop - current_price) / current_price * 100):+.2f}%)")
    print(f"   📍 Aggressive (200-week SMA): ${aggressive_stop:,.2f} ({((aggressive_stop - current_price) / current_price * 100):+.2f}%)")

print()

# Generate charts
print("📊 Generating distribution charts...")
charts = {}

# Chart 1: Price vs 50-week and 200-week SMA
fig1, ax1 = plt.subplots(figsize=(16, 8))

ax1.plot(weekly['date'], weekly['c'], color='#3498db', linewidth=2, label='Bitcoin Price', alpha=0.8)
ax1.plot(weekly['date'], weekly['SMA_50'], color='#e74c3c', linewidth=2, label='50-Week SMA', linestyle='--')
ax1.plot(weekly['date'], weekly['SMA_200'], color='#27ae60', linewidth=2, label='200-Week SMA', linestyle='--')

# Shade regions based on position relative to 50-week SMA
above_50 = weekly['Above_50SMA']
ax1.fill_between(weekly['date'], 0, weekly['c'].max() * 1.1,
                  where=above_50, alpha=0.1, color='green', label='Above 50-week SMA (LONG)')
ax1.fill_between(weekly['date'], 0, weekly['c'].max() * 1.1,
                  where=~above_50, alpha=0.1, color='red', label='Below 50-week SMA (SHORT)')

# Mark current position
ax1.scatter([weekly['date'].iloc[-1]], [current_price], color='black', s=200, zorder=5, marker='o')
ax1.text(weekly['date'].iloc[-1], current_price * 1.05, f'Current: ${current_price:,.0f}',
         fontsize=12, fontweight='bold', ha='right')

ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Price (USD)', fontsize=12)
ax1.set_title('50-Week SMA Trading Strategy: Long/Short Signals', fontsize=14, fontweight='bold')
ax1.set_yscale('log')
ax1.legend(loc='upper left')
ax1.grid(alpha=0.3)

charts['sma_strategy'] = fig_to_base64(fig1)

# Chart 2: Distribution of price distance from 50-week SMA by phase
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
axes2 = axes2.flatten()

colors = {'bull': '#27ae60', 'distribution': '#f39c12', 'bear': '#e74c3c', 'accumulation': '#3498db'}

for idx, phase in enumerate(['bull', 'distribution', 'bear', 'accumulation']):
    ax = axes2[idx]
    phase_data = weekly[weekly['Market_Phase'] == phase]['Distance_from_50SMA_pct'].dropna()

    if len(phase_data) > 10:
        # Histogram
        ax.hist(phase_data, bins=30, alpha=0.6, color=colors[phase], edgecolor='black', density=True)

        # Fit normal distribution
        mu, sigma = phase_data.mean(), phase_data.std()
        x = np.linspace(phase_data.min(), phase_data.max(), 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label=f'Normal (μ={mu:.1f}%, σ={sigma:.1f}%)')

        # Mark current position if in this phase
        if phase == current_phase:
            ax.axvline(current_distance, color='black', linestyle='--', linewidth=3, label=f'Current: {current_distance:.1f}%')

        # Mark key percentiles
        ax.axvline(phase_data.quantile(0.25), color='gray', linestyle=':', alpha=0.7, label='25th/75th percentile')
        ax.axvline(phase_data.quantile(0.75), color='gray', linestyle=':', alpha=0.7)

        ax.set_xlabel('Distance from 50-Week SMA (%)', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'{phase.capitalize()} Phase Distribution (n={len(phase_data)})', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

plt.tight_layout()
charts['phase_distributions'] = fig_to_base64(fig2)

# Chart 3: Distribution around 200-week SMA
fig3, ax3 = plt.subplots(figsize=(16, 8))

distance_200 = weekly['Distance_from_200SMA_pct'].dropna()
ax3.hist(distance_200, bins=50, alpha=0.6, color='#3498db', edgecolor='black', density=True)

# Fit distribution
mu_200, sigma_200 = distance_200.mean(), distance_200.std()
x = np.linspace(distance_200.min(), distance_200.max(), 200)
ax3.plot(x, stats.norm.pdf(x, mu_200, sigma_200), 'r-', linewidth=2,
         label=f'Normal Fit (μ={mu_200:.1f}%, σ={sigma_200:.1f}%)')

# Current position
current_distance_200 = weekly['Distance_from_200SMA_pct'].iloc[-1]
ax3.axvline(current_distance_200, color='black', linestyle='--', linewidth=3,
            label=f'Current Position: {current_distance_200:.1f}%')

# Percentiles
for percentile in [5, 25, 50, 75, 95]:
    value = distance_200.quantile(percentile/100)
    ax3.axvline(value, color='gray', linestyle=':', alpha=0.5)
    ax3.text(value, ax3.get_ylim()[1] * 0.9, f'{percentile}th', fontsize=8, ha='center')

ax3.set_xlabel('Distance from 200-Week SMA (%)', fontsize=12)
ax3.set_ylabel('Density', fontsize=12)
ax3.set_title('Historical Distribution of Price Around 200-Week SMA', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)

charts['sma_200_distribution'] = fig_to_base64(fig3)

# Chart 4: Volatility by phase (box plot)
fig4, ax4 = plt.subplots(figsize=(14, 8))

phase_order = ['bear', 'accumulation', 'distribution', 'bull']
data_for_boxplot = [weekly[weekly['Market_Phase'] == p]['Distance_from_50SMA_pct'].dropna() for p in phase_order]

bp = ax4.boxplot(data_for_boxplot, labels=[p.capitalize() for p in phase_order],
                  patch_artist=True, widths=0.6)

for patch, phase in zip(bp['boxes'], phase_order):
    patch.set_facecolor(colors[phase])
    patch.set_alpha(0.7)

# Mark current phase
if current_phase in phase_order:
    current_idx = phase_order.index(current_phase)
    ax4.scatter([current_idx + 1], [current_distance], color='black', s=300,
                marker='*', zorder=5, label='Current Position')

ax4.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
ax4.set_ylabel('Distance from 50-Week SMA (%)', fontsize=12)
ax4.set_title('Volatility Around 50-Week SMA by Market Phase', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

charts['volatility_by_phase'] = fig_to_base64(fig4)

# Chart 5: Crossover signals and performance
fig5, (ax5a, ax5b) = plt.subplots(2, 1, figsize=(16, 10))

# Top: Price with crossover signals
ax5a.plot(weekly['date'], weekly['c'], color='#3498db', linewidth=2, label='Bitcoin Price', alpha=0.8)
ax5a.plot(weekly['date'], weekly['SMA_50'], color='#e74c3c', linewidth=2, label='50-Week SMA', linestyle='--')

# Mark bullish crosses (buy signals)
bullish_crosses = weekly[weekly['50SMA_Cross'] == 1]
ax5a.scatter(bullish_crosses['date'], bullish_crosses['c'], color='green', s=200,
             marker='^', zorder=5, label='Buy Signal (Cross Above)', edgecolors='black', linewidths=2)

# Mark bearish crosses (sell signals)
bearish_crosses = weekly[weekly['50SMA_Cross'] == -1]
ax5a.scatter(bearish_crosses['date'], bearish_crosses['c'], color='red', s=200,
             marker='v', zorder=5, label='Sell Signal (Cross Below)', edgecolors='black', linewidths=2)

ax5a.set_ylabel('Price (USD)', fontsize=12)
ax5a.set_title('50-Week SMA Crossover Signals', fontsize=14, fontweight='bold')
ax5a.set_yscale('log')
ax5a.legend(loc='upper left')
ax5a.grid(alpha=0.3)

# Bottom: Distance from 50-week SMA
ax5b.fill_between(weekly['date'], 0, weekly['Distance_from_50SMA_pct'],
                   where=(weekly['Distance_from_50SMA_pct'] > 0), color='green', alpha=0.3, label='Above SMA (LONG)')
ax5b.fill_between(weekly['date'], 0, weekly['Distance_from_50SMA_pct'],
                   where=(weekly['Distance_from_50SMA_pct'] < 0), color='red', alpha=0.3, label='Below SMA (SHORT)')
ax5b.plot(weekly['date'], weekly['Distance_from_50SMA_pct'], color='black', linewidth=1.5, alpha=0.8)
ax5b.axhline(0, color='black', linestyle='-', linewidth=2)

# Shade 1 and 2 standard deviation zones for current phase
if current_phase in volatility_by_phase:
    std = volatility_by_phase[current_phase]['std']
    ax5b.axhline(std, color='orange', linestyle='--', alpha=0.7, label=f'±1σ ({current_phase})')
    ax5b.axhline(-std, color='orange', linestyle='--', alpha=0.7)
    ax5b.axhline(std * 2, color='red', linestyle='--', alpha=0.7, label=f'±2σ (stop loss)')
    ax5b.axhline(-std * 2, color='red', linestyle='--', alpha=0.7)

ax5b.set_xlabel('Date', fontsize=12)
ax5b.set_ylabel('Distance from 50-Week SMA (%)', fontsize=12)
ax5b.set_title('Price Distance from 50-Week SMA Over Time', fontsize=14, fontweight='bold')
ax5b.legend(loc='upper left')
ax5b.grid(alpha=0.3)

plt.tight_layout()
charts['crossover_signals'] = fig_to_base64(fig5)

print(f"✅ Generated {len(charts)} charts")
print()

# Calculate strategy statistics
print("📊 Calculating strategy performance...")

# Count crossovers
bullish_count = len(weekly[weekly['50SMA_Cross'] == 1])
bearish_count = len(weekly[weekly['50SMA_Cross'] == -1])

print(f"   Total bullish crosses: {bullish_count}")
print(f"   Total bearish crosses: {bearish_count}")

# Current signal
is_long = weekly['Above_50SMA'].iloc[-1]
print(f"   Current signal: {'LONG ✅' if is_long else 'SHORT ⬇️'}")
print()

# Save results
results = {
    'current_price': float(current_price),
    'current_50sma': float(current_50sma),
    'current_200sma': float(current_200sma),
    'distance_from_50sma_pct': float(current_distance),
    'distance_from_200sma_pct': float(current_distance_200),
    'current_phase': current_phase,
    'current_position': position if current_phase in volatility_by_phase else 'unknown',
    'stop_loss_conservative': float(conservative_stop) if current_phase in volatility_by_phase else None,
    'stop_loss_moderate': float(moderate_stop) if current_phase in volatility_by_phase else None,
    'stop_loss_aggressive': float(aggressive_stop) if current_phase in volatility_by_phase else None,
    'volatility_by_phase': {k: {k2: float(v2) if isinstance(v2, (int, float)) else v2
                                for k2, v2 in v.items()}
                           for k, v in volatility_by_phase.items()},
    'bullish_crosses': int(bullish_count),
    'bearish_crosses': int(bearish_count),
    'is_long': bool(is_long),
    'charts': charts
}

# Save with charts included this time
with open(OUTPUT_DIR / 'trading_strategy_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print("✅ Results saved to trading_strategy_results.json")
print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print()
print(f"🎯 Trading Signal: {position if current_phase in volatility_by_phase else 'UNKNOWN'}")
print(f"📊 Current Phase: {current_phase.upper()}")
print(f"💰 Bitcoin Price: ${current_price:,.2f}")
print(f"📈 50-Week SMA: ${current_50sma:,.2f} ({current_distance:+.2f}%)")
print(f"📈 200-Week SMA: ${current_200sma:,.2f} ({current_distance_200:+.2f}%)")
if current_phase in volatility_by_phase:
    print(f"\n🛡️ Recommended Stop Loss:")
    print(f"   Conservative: ${conservative_stop:,.2f}")
    print(f"   Moderate: ${moderate_stop:,.2f}")
    print(f"   Aggressive: ${aggressive_stop:,.2f}")
print()
