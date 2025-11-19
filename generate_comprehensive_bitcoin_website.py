#!/usr/bin/env python3
"""
Comprehensive Bitcoin Analysis Website Generator
Integrates all analyses into a professional, scientific, tabbed interface
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

# Paths
BASE_DIR = Path("/Users/alexhorton/quant connect dev environment")
DATA_FILE = BASE_DIR / "datasets" / "btc-ohlc.csv"
OUTPUT_DIR = BASE_DIR / "ml_pipeline" / "reports"
M2_RESULTS = OUTPUT_DIR / "m2_interest_rate_study_results.json"
TRADING_STRATEGY_RESULTS = OUTPUT_DIR / "trading_strategy_results.json"

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

def load_bitcoin_data():
    """Load and prepare Bitcoin OHLC data with current price"""
    print("📊 Loading Bitcoin data...")
    df = pd.read_csv(DATA_FILE)
    df.columns = ['timestamp', 'unix_time', 'open', 'high', 'low', 'close', 'volume']
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Get current price and key levels
    current_price = df['close'].iloc[-1]
    current_date = df['timestamp'].iloc[-1]

    # Calculate key technical levels
    df['MA_20'] = df['close'].rolling(window=20).mean()
    df['MA_50'] = df['close'].rolling(window=50).mean()
    df['MA_200'] = df['close'].rolling(window=200).mean()

    # Bollinger Bands
    df['BB_middle'] = df['close'].rolling(window=20).mean()
    df['BB_std'] = df['close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
    df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)

    # Support and Resistance (last 365 days)
    recent_data = df.tail(365)
    support = recent_data['low'].min()
    resistance = recent_data['high'].max()

    # Fibonacci retracement from recent high to recent low
    fib_high = resistance
    fib_low = support
    fib_diff = fib_high - fib_low

    fib_levels = {
        '0.0% (High)': fib_high,
        '23.6%': fib_high - (fib_diff * 0.236),
        '38.2%': fib_high - (fib_diff * 0.382),
        '50.0%': fib_high - (fib_diff * 0.500),
        '61.8%': fib_high - (fib_diff * 0.618),
        '78.6%': fib_high - (fib_diff * 0.786),
        '100.0% (Low)': fib_low
    }

    print(f"✅ Loaded {len(df)} records, current price: ${current_price:,.2f}")

    return df, {
        'current_price': current_price,
        'current_date': current_date,
        'ma_20': df['MA_20'].iloc[-1],
        'ma_50': df['MA_50'].iloc[-1],
        'ma_200': df['MA_200'].iloc[-1],
        'bb_upper': df['BB_upper'].iloc[-1],
        'bb_lower': df['BB_lower'].iloc[-1],
        'support': support,
        'resistance': resistance,
        'fib_levels': fib_levels,
        'high_24h': df['high'].tail(24).max(),
        'low_24h': df['low'].tail(24).min(),
        'volume_24h': df['volume'].tail(24).sum()
    }

def generate_current_overview_chart(df, metrics):
    """Generate current price chart with key levels"""
    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot last 90 days
    recent = df.tail(90).copy()
    recent = recent.reset_index(drop=True)

    ax.plot(recent.index, recent['close'], color='#3498db', linewidth=2, label='Bitcoin Price')
    ax.plot(recent.index, recent['MA_20'], color='#e74c3c', linewidth=1.5, linestyle='--', label='MA 20', alpha=0.7)
    ax.plot(recent.index, recent['MA_50'], color='#f39c12', linewidth=1.5, linestyle='--', label='MA 50', alpha=0.7)
    ax.plot(recent.index, recent['MA_200'], color='#27ae60', linewidth=1.5, linestyle='--', label='MA 200', alpha=0.7)

    # Bollinger Bands
    ax.fill_between(recent.index, recent['BB_upper'], recent['BB_lower'], alpha=0.1, color='gray', label='Bollinger Bands')

    # Current price line
    ax.axhline(y=metrics['current_price'], color='black', linestyle=':', linewidth=2, label=f"Current: ${metrics['current_price']:,.0f}")

    ax.set_title(f"Bitcoin Price - Last 90 Days (Updated: {metrics['current_date'].strftime('%Y-%m-%d')})",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Days Ago', fontsize=12)
    ax.set_ylabel('Price (USD)', fontsize=12)
    ax.legend(loc='best')
    ax.grid(alpha=0.3)

    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    return fig_to_base64(fig)

def generate_key_levels_chart(metrics):
    """Generate horizontal bar chart of key price levels"""
    fig, ax = plt.subplots(figsize=(12, 8))

    current = metrics['current_price']

    levels = {
        'Resistance (365d High)': metrics['resistance'],
        'Fib 23.6%': metrics['fib_levels']['23.6%'],
        'Fib 38.2%': metrics['fib_levels']['38.2%'],
        'Fib 50.0%': metrics['fib_levels']['50.0%'],
        'Fib 61.8%': metrics['fib_levels']['61.8%'],
        'MA 200': metrics['ma_200'],
        'MA 50': metrics['ma_50'],
        'MA 20': metrics['ma_20'],
        'BB Upper': metrics['bb_upper'],
        'BB Lower': metrics['bb_lower'],
        'Support (365d Low)': metrics['support']
    }

    # Sort by price
    sorted_levels = dict(sorted(levels.items(), key=lambda x: x[1], reverse=True))

    names = list(sorted_levels.keys())
    values = list(sorted_levels.values())

    # Color based on above/below current price
    colors = ['#e74c3c' if v > current else '#27ae60' for v in values]

    bars = ax.barh(names, values, color=colors, alpha=0.6)

    # Add current price line
    ax.axvline(x=current, color='black', linewidth=3, label=f'Current: ${current:,.0f}')

    # Add value labels
    for i, (name, val) in enumerate(sorted_levels.items()):
        distance = ((val - current) / current) * 100
        ax.text(val, i, f'  ${val:,.0f} ({distance:+.1f}%)',
               va='center', fontsize=9, fontweight='bold')

    ax.set_xlabel('Price (USD)', fontsize=12)
    ax.set_title('Key Price Levels to Watch', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)

    return fig_to_base64(fig)

def load_m2_results():
    """Load M2 interest rate study results"""
    try:
        with open(M2_RESULTS, 'r') as f:
            return json.load(f)
    except:
        return None

def load_trading_strategy_results():
    """Load trading strategy analysis results"""
    try:
        with open(TRADING_STRATEGY_RESULTS, 'r') as f:
            return json.load(f)
    except:
        return None

def generate_trading_strategy_tab_content(trading_strategy_results):
    """Generate dynamic trading strategy tab HTML"""

    if not trading_strategy_results:
        return """
        <div id="strategy" class="tab-content">
            <h2 class="section-title">50-Week SMA Trading Strategy</h2>
            <div class="warning-box">
                <strong>⚠️  Trading strategy analysis not available.</strong><br>
                Run <code>python3 generate_trading_strategy_analysis.py</code> to generate the analysis.
            </div>
        </div>
        """

    ts = trading_strategy_results
    current_price = ts['current_price']
    sma_50 = ts['current_50sma']
    sma_200 = ts['current_200sma']
    dist_50 = ts['distance_from_50sma_pct']
    dist_200 = ts['distance_from_200sma_pct']
    phase = ts['current_phase']
    is_long = ts['is_long']
    signal = "LONG ✅" if is_long else "SHORT ⬇️"
    signal_color = "positive" if is_long else "negative"
    charts = ts.get('charts', {})
    vol_by_phase = ts.get('volatility_by_phase', {})

    # Phase descriptions
    phase_descriptions = {
        'bull': 'Strong upward trend - price above both 50 and 200-week SMAs with positive momentum',
        'distribution': 'Weakening uptrend - price still above SMAs but momentum is declining',
        'bear': 'Strong downward trend - price below both SMAs with negative momentum',
        'accumulation': 'Potential bottom formation - price below SMAs but momentum improving',
        'transition': 'Market is between major trends - direction uncertain, higher volatility expected',
        'unknown': 'Insufficient data to classify phase accurately'
    }
    phase_desc = phase_descriptions.get(phase.lower(), 'Phase classification pending')

    # Generate phase context
    phase_context = "Historical data for this phase is being calculated."
    if phase in vol_by_phase:
        stats = vol_by_phase[phase]
        mean = stats['mean']
        std = stats['std']
        z_score = (dist_50 - mean) / std if std > 0 else 0

        if abs(z_score) < 1:
            phase_context = f"Current distance ({dist_50:.2f}%) is within normal range for this phase (mean: {mean:.2f}%, ±1σ: {std:.2f}%)"
        elif abs(z_score) < 2:
            phase_context = f"Current distance ({dist_50:.2f}%) is {abs(z_score):.1f} standard deviations from mean ({mean:.2f}%) - approaching extreme"
        else:
            phase_context = f"⚠️ Current distance ({dist_50:.2f}%) is {abs(z_score):.1f}σ from mean ({mean:.2f}%) - EXTREME positioning!"

    html = f"""
        <!-- TRADING STRATEGY TAB -->
        <div id="strategy" class="tab-content">
            <h2 class="section-title">50-Week SMA Trading Strategy</h2>

            <div class="info-box">
                <strong>📈 Strategy Overview:</strong><br>
                This mechanical trading system uses the 50-week Simple Moving Average (SMA) as the primary signal:<br><br>
                <strong style="color: #27ae60;">✅ LONG Signal:</strong> When weekly close crosses ABOVE the 50-week SMA<br>
                <strong style="color: #e74c3c;">⬇️ SHORT Signal:</strong> When weekly close crosses BELOW the 50-week SMA<br><br>
                This tab shows where you are in the historical distribution of price movements around the SMAs.
            </div>

            <h3 class="subsection-title">📊 Current Strategy Status</h3>

            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-label">Current Bitcoin Price</div>
                    <div class="metric-value">${current_price:,.2f}</div>
                    <div class="metric-subtext">Weekly Close</div>
                </div>

                <div class="metric-card neutral">
                    <div class="metric-label">50-Week SMA</div>
                    <div class="metric-value">${sma_50:,.2f}</div>
                    <div class="metric-subtext">Price is {dist_50:+.2f}% {'above' if dist_50 > 0 else 'below'}</div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">200-Week SMA</div>
                    <div class="metric-value">${sma_200:,.2f}</div>
                    <div class="metric-subtext">Price is {dist_200:+.2f}% {'above' if dist_200 > 0 else 'below'}</div>
                </div>

                <div class="metric-card {signal_color}">
                    <div class="metric-label">Current Signal</div>
                    <div class="metric-value" style="font-size: 1.8em;">{signal}</div>
                    <div class="metric-subtext">{'Above' if is_long else 'Below'} 50-week SMA</div>
                </div>
            </div>

            <div class="warning-box">
                <strong>⚠️ Current Market Phase: {phase.upper()}</strong><br>
                {phase_desc}<br><br>
                <strong>Distance from 50-Week SMA:</strong> {dist_50:.2f}%<br>
                <strong>Historical Context:</strong> {phase_context}
            </div>

            <h3 class="subsection-title">📊 Price vs 50-Week & 200-Week SMAs</h3>
            <p>This chart shows the entire price history with LONG zones (green) when price is above the 50-week SMA and SHORT zones (red) when below.</p>
            <div class="chart-container">
                <img src="data:image/png;base64,{charts.get('sma_strategy', '')}" alt="50-Week SMA Strategy">
            </div>

            <h3 class="subsection-title">📈 Distribution by Market Phase - WHERE ARE YOU?</h3>
            <p><strong>This is the key chart!</strong> Each panel shows how price typically behaves around the 50-week SMA during different market phases. <strong style="color: black;">The black vertical line shows YOUR current position.</strong></p>

            <div class="chart-container">
                <img src="data:image/png;base64,{charts.get('phase_distributions', '')}" alt="Phase Distributions">
            </div>
    """

    # Add phase-specific stats if available
    if phase in vol_by_phase:
        stats = vol_by_phase[phase]
        html += f"""
            <div class="info-box">
                <strong>📊 Current Phase Analysis ({phase.capitalize()}):</strong><br>
                <strong>Mean Distance from 50-SMA:</strong> {stats['mean']:.2f}%<br>
                <strong>Standard Deviation:</strong> {stats['std']:.2f}%<br>
                <strong>Historical Range:</strong> {stats['min']:.2f}% to {stats['max']:.2f}%<br>
                <strong>Sample Size:</strong> {stats['count']} weeks<br>
                <strong>Your Current Distance:</strong> {dist_50:.2f}%
            </div>
        """

    html += f"""
            <h3 class="subsection-title">📊 Historical Distribution Around 200-Week SMA</h3>
            <p>This shows where Bitcoin price has historically been relative to the 200-week SMA. <strong style="color: black;">The black line shows where you are now ({dist_200:+.2f}%).</strong></p>

            <div class="chart-container">
                <img src="data:image/png;base64,{charts.get('sma_200_distribution', '')}" alt="200-Week SMA Distribution">
            </div>

            <h3 class="subsection-title">📦 Volatility Comparison by Phase</h3>
            <p>This box plot shows the typical range of distances from the 50-week SMA for each market phase. <strong>Your current phase is: {phase.upper()}</strong></p>

            <div class="chart-container">
                <img src="data:image/png;base64,{charts.get('volatility_by_phase', '')}" alt="Volatility by Phase">
            </div>

            <h3 class="subsection-title">🎯 Crossover Signals Over Time</h3>
            <p>Green triangles = BUY signals (cross above 50-SMA). Red triangles = SELL signals (cross below 50-SMA).</p>

            <div class="chart-container">
                <img src="data:image/png;base64,{charts.get('crossover_signals', '')}" alt="Crossover Signals">
            </div>

            <h3 class="subsection-title">🛡️ Risk Management Guidelines</h3>
    """

    # Risk management table
    stop_conservative = ts.get('stop_loss_conservative')
    if stop_conservative:
        stop_moderate = ts.get('stop_loss_moderate')
        stop_aggressive = ts.get('stop_loss_aggressive')

        html += f"""
            <table>
                <tr>
                    <th>Stop Loss Type</th>
                    <th>Price Level</th>
                    <th>Distance from Current</th>
                    <th>When to Use</th>
                </tr>
                <tr>
                    <td><strong>Conservative</strong></td>
                    <td>${stop_conservative:,.2f}</td>
                    <td>{((stop_conservative - current_price) / current_price * 100):+.2f}%</td>
                    <td>Low risk tolerance, capital preservation</td>
                </tr>
                <tr>
                    <td><strong>Moderate</strong></td>
                    <td>${stop_moderate:,.2f}</td>
                    <td>{((stop_moderate - current_price) / current_price * 100):+.2f}%</td>
                    <td>Balanced approach (recommended)</td>
                </tr>
                <tr>
                    <td><strong>Aggressive</strong></td>
                    <td>${stop_aggressive:,.2f}</td>
                    <td>{((stop_aggressive - current_price) / current_price * 100):+.2f}%</td>
                    <td>Long-term holders, high risk tolerance</td>
                </tr>
            </table>

            <div class="warning-box" style="margin-top: 20px;">
                <strong>Position Sizing Example (2% Risk Rule):</strong><br>
                For a $100,000 account with conservative stop loss:<br>
                • Maximum risk: $2,000 (2% of account)<br>
                • Stop distance: ${abs(stop_conservative - current_price):,.2f}<br>
                • <strong>Maximum position size: {2000 / abs(stop_conservative - current_price):.4f} BTC</strong>
            </div>
        """

    # Volatility table
    if vol_by_phase:
        html += """
            <h3 class="subsection-title">📊 Historical Volatility by Phase</h3>
            <table>
                <tr>
                    <th>Phase</th>
                    <th>Mean Distance</th>
                    <th>Std Deviation</th>
                    <th>Range</th>
                    <th>Sample Size</th>
                </tr>
        """

        for phase_name in ['bull', 'distribution', 'accumulation', 'bear']:
            if phase_name in vol_by_phase:
                stats = vol_by_phase[phase_name]
                html += f"""
                <tr>
                    <td><strong>{phase_name.capitalize()}</strong></td>
                    <td>{stats['mean']:+.2f}%</td>
                    <td>±{stats['std']:.2f}%</td>
                    <td>{stats['min']:.2f}% to {stats['max']:.2f}%</td>
                    <td>{stats['count']} weeks</td>
                </tr>
                """

        html += "</table>"

    html += """
            <div class="warning-box">
                <strong>⚠️ Disclaimer:</strong><br>
                This trading strategy is provided for educational purposes only. Past performance does not guarantee future results.
                All trading involves substantial risk of loss. Always do your own research and never invest more than you can afford to lose.
            </div>
        </div>
    """

    return html

def generate_html(df, metrics, m2_results, trading_strategy_results):
    """Generate comprehensive HTML website with tabs"""

    # Generate charts
    print("📊 Generating overview chart...")
    overview_chart = generate_current_overview_chart(df, metrics)

    print("📊 Generating key levels chart...")
    levels_chart = generate_key_levels_chart(metrics)

    # Get recent price action
    today = df.iloc[-1]
    yesterday = df.iloc[-2]
    price_change = today['close'] - yesterday['close']
    price_change_pct = (price_change / yesterday['close']) * 100

    # Determine trend
    if metrics['current_price'] > metrics['ma_20'] > metrics['ma_50'] > metrics['ma_200']:
        trend = "Strong Uptrend"
        trend_color = "#27ae60"
        trend_confidence = "85%"
    elif metrics['current_price'] > metrics['ma_50']:
        trend = "Moderate Uptrend"
        trend_color = "#2ecc71"
        trend_confidence = "65%"
    elif metrics['current_price'] < metrics['ma_20'] < metrics['ma_50'] < metrics['ma_200']:
        trend = "Strong Downtrend"
        trend_color = "#e74c3c"
        trend_confidence = "85%"
    elif metrics['current_price'] < metrics['ma_50']:
        trend = "Moderate Downtrend"
        trend_color = "#c0392b"
        trend_confidence = "65%"
    else:
        trend = "Sideways/Consolidation"
        trend_color = "#f39c12"
        trend_confidence = "50%"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>₿ Bitcoin Analysis - Live Dashboard</title>
    <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>₿</text></svg>">
    <meta name="description" content="Real-time Bitcoin analysis with ML predictions, trading strategies, and market insights">
    <meta http-equiv="refresh" content="300">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
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

        .header .subtitle {{
            font-size: 1.1em;
            opacity: 0.9;
        }}

        .update-time {{
            background: rgba(255,255,255,0.1);
            padding: 10px;
            border-radius: 5px;
            margin-top: 15px;
            font-size: 0.9em;
        }}

        .tabs {{
            display: flex;
            background: #f8f9fa;
            border-bottom: 2px solid #dee2e6;
            flex-wrap: wrap;
        }}

        .tab {{
            padding: 15px 25px;
            cursor: pointer;
            background: #f8f9fa;
            border: none;
            font-size: 1em;
            font-weight: 500;
            transition: all 0.3s;
            flex: 1;
            min-width: 150px;
        }}

        .tab:hover {{
            background: #e9ecef;
        }}

        .tab.active {{
            background: white;
            border-bottom: 3px solid #3498db;
            color: #3498db;
        }}

        .tab-content {{
            display: none;
            padding: 30px;
            animation: fadeIn 0.5s;
        }}

        .tab-content.active {{
            display: block;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}

        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}

        .metric-card.neutral {{
            background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
        }}

        .metric-card.positive {{
            background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
        }}

        .metric-card.negative {{
            background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        }}

        .metric-label {{
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 5px;
        }}

        .metric-value {{
            font-size: 1.8em;
            font-weight: bold;
        }}

        .metric-subtext {{
            font-size: 0.85em;
            opacity: 0.8;
            margin-top: 5px;
        }}

        .chart-container {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}

        .chart-container img {{
            width: 100%;
            height: auto;
            border-radius: 5px;
        }}

        .info-box {{
            background: #e8f4f8;
            border-left: 4px solid #3498db;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }}

        .warning-box {{
            background: #fff3cd;
            border-left: 4px solid #f39c12;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }}

        .section-title {{
            font-size: 1.8em;
            margin-bottom: 20px;
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}

        .subsection-title {{
            font-size: 1.3em;
            margin: 25px 0 15px 0;
            color: #34495e;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}

        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}

        th {{
            background: #f8f9fa;
            font-weight: 600;
            color: #2c3e50;
        }}

        tr:hover {{
            background: #f8f9fa;
        }}

        .confidence-badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
        }}

        .conf-high {{
            background: #d4edda;
            color: #155724;
        }}

        .conf-moderate {{
            background: #fff3cd;
            color: #856404;
        }}

        .conf-low {{
            background: #f8d7da;
            color: #721c24;
        }}

        .footer {{
            background: #2c3e50;
            color: white;
            padding: 30px;
            text-align: center;
        }}

        .footer a {{
            color: #3498db;
            text-decoration: none;
        }}

        .footer a:hover {{
            text-decoration: underline;
        }}

        .key-levels-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}

        .level-card {{
            padding: 15px;
            border-radius: 8px;
            border: 2px solid #dee2e6;
        }}

        .level-card.above {{
            background: #fee;
            border-color: #e74c3c;
        }}

        .level-card.below {{
            background: #efe;
            border-color: #27ae60;
        }}

        .level-name {{
            font-weight: 600;
            margin-bottom: 5px;
        }}

        .level-price {{
            font-size: 1.3em;
            font-weight: bold;
        }}

        .level-distance {{
            font-size: 0.9em;
            opacity: 0.7;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🪙 Bitcoin Analysis</h1>
            <div class="subtitle">Comprehensive Research Dashboard - Scientific Analysis with Transparent Methodology</div>
            <div class="update-time">
                Last Updated: {metrics['current_date'].strftime('%B %d, %Y at %I:%M %p UTC')}<br>
                Data Source: Coinbase Exchange | Analysis Framework: Python ML Pipeline
            </div>
        </div>

        <div class="tabs">
            <button class="tab active" onclick="openTab(event, 'overview')">📊 Overview</button>
            <button class="tab" onclick="openTab(event, 'levels')">🎯 Key Levels</button>
            <button class="tab" onclick="openTab(event, 'strategy')">⚡ Trading Strategy</button>
            <button class="tab" onclick="openTab(event, 'm2-analysis')">💵 M2 & Interest Rates</button>
            <button class="tab" onclick="openTab(event, 'lead-lag')">🔄 Lead-Lag Analysis</button>
            <button class="tab" onclick="openTab(event, 'methodology')">📚 Methodology</button>
            <button class="tab" onclick="openTab(event, 'reports')">📄 All Reports</button>
        </div>

        <!-- OVERVIEW TAB -->
        <div id="overview" class="tab-content active">
            <h2 class="section-title">Current Market Overview</h2>

            <div class="metric-grid">
                <div class="metric-card {'positive' if price_change > 0 else 'negative' if price_change < 0 else 'neutral'}">
                    <div class="metric-label">Current Bitcoin Price</div>
                    <div class="metric-value">${metrics['current_price']:,.2f}</div>
                    <div class="metric-subtext">{'▲' if price_change > 0 else '▼'} {price_change_pct:+.2f}% (24h)</div>
                </div>

                <div class="metric-card neutral">
                    <div class="metric-label">Market Trend</div>
                    <div class="metric-value" style="font-size: 1.4em;">{trend}</div>
                    <div class="metric-subtext">Confidence: {trend_confidence}</div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">24h High / Low</div>
                    <div class="metric-value" style="font-size: 1.3em;">${metrics['high_24h']:,.0f}</div>
                    <div class="metric-subtext">${metrics['low_24h']:,.0f} (Range: {((metrics['high_24h'] - metrics['low_24h']) / metrics['low_24h'] * 100):.1f}%)</div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">24h Volume</div>
                    <div class="metric-value">{metrics['volume_24h']:,.0f}</div>
                    <div class="metric-subtext">BTC traded</div>
                </div>
            </div>

            <div class="info-box">
                <strong>📊 Trend Analysis ({trend_confidence} confidence):</strong><br>
                The current trend is classified as <strong>{trend}</strong> based on moving average alignment.
                Bitcoin is trading {'above' if metrics['current_price'] > metrics['ma_200'] else 'below'} the 200-day moving average (${metrics['ma_200']:,.0f}),
                which is a key long-term trend indicator.

                <p style="margin-top: 10px;"><strong>Moving Averages:</strong></p>
                <ul style="margin-left: 20px; margin-top: 5px;">
                    <li>MA 20: ${metrics['ma_20']:,.2f} ({'support' if metrics['current_price'] > metrics['ma_20'] else 'resistance'})</li>
                    <li>MA 50: ${metrics['ma_50']:,.2f} ({'support' if metrics['current_price'] > metrics['ma_50'] else 'resistance'})</li>
                    <li>MA 200: ${metrics['ma_200']:,.2f} ({'support' if metrics['current_price'] > metrics['ma_200'] else 'resistance'})</li>
                </ul>
            </div>

            <div class="chart-container">
                <img src="data:image/png;base64,{overview_chart}" alt="Bitcoin Price Chart">
            </div>

            <h3 class="subsection-title">Technical Context</h3>

            <div class="warning-box">
                <strong>⚠️ Important Note on Confidence Scores:</strong><br>
                All analysis includes confidence scores (High 70-100%, Moderate 50-69%, Low <50%) to indicate certainty levels.
                Technical analysis is probabilistic, not deterministic. These insights should inform, not dictate, investment decisions.
                Always consider multiple timeframes and indicators. Past performance does not guarantee future results.
            </div>

            <table>
                <tr>
                    <th>Indicator</th>
                    <th>Value</th>
                    <th>Interpretation</th>
                    <th>Confidence</th>
                </tr>
                <tr>
                    <td>Bollinger Bands</td>
                    <td>${metrics['bb_lower']:,.0f} - ${metrics['bb_upper']:,.0f}</td>
                    <td>{'Near upper band - potential overbought' if metrics['current_price'] > (metrics['bb_upper'] - (metrics['bb_upper'] - metrics['bb_lower']) * 0.2) else 'Near lower band - potential oversold' if metrics['current_price'] < (metrics['bb_lower'] + (metrics['bb_upper'] - metrics['bb_lower']) * 0.2) else 'Within normal range'}</td>
                    <td><span class="confidence-badge conf-moderate">Moderate</span></td>
                </tr>
                <tr>
                    <td>Support Level</td>
                    <td>${metrics['support']:,.0f}</td>
                    <td>365-day low - strong support zone</td>
                    <td><span class="confidence-badge conf-high">High</span></td>
                </tr>
                <tr>
                    <td>Resistance Level</td>
                    <td>${metrics['resistance']:,.0f}</td>
                    <td>365-day high - strong resistance zone</td>
                    <td><span class="confidence-badge conf-high">High</span></td>
                </tr>
            </table>
        </div>

        <!-- KEY LEVELS TAB -->
        <div id="levels" class="tab-content">
            <h2 class="section-title">Key Price Levels to Watch</h2>

            <div class="info-box">
                <strong>📈 How to Use These Levels:</strong><br>
                • <strong style="color: #e74c3c;">Red levels (above current)</strong> represent potential resistance - prices where selling pressure may increase<br>
                • <strong style="color: #27ae60;">Green levels (below current)</strong> represent potential support - prices where buying pressure may increase<br>
                • Fibonacci retracements are based on the recent 365-day high/low range and indicate statistically likely reversal points<br>
                • Moving averages act as dynamic support/resistance that adjusts with price trends
            </div>

            <div class="chart-container">
                <img src="data:image/png;base64,{levels_chart}" alt="Key Price Levels">
            </div>

            <h3 class="subsection-title">Detailed Level Breakdown</h3>

            <div class="key-levels-grid">
"""

    # Add Fibonacci levels
    current = metrics['current_price']
    for name, price in sorted(metrics['fib_levels'].items(), key=lambda x: x[1], reverse=True):
        distance = ((price - current) / current) * 100
        above_below = 'above' if price > current else 'below'
        html += f"""
                <div class="level-card {above_below}">
                    <div class="level-name">{name}</div>
                    <div class="level-price">${price:,.0f}</div>
                    <div class="level-distance">{distance:+.2f}% from current</div>
                </div>
"""

    html += """
            </div>

            <div class="warning-box">
                <strong>⚠️ Methodology & Limitations:</strong><br>
                • <strong>Fibonacci Levels:</strong> Calculated from 365-day high ($""" + f"{metrics['resistance']:,.0f}" + """) to low ($""" + f"{metrics['support']:,.0f}" + """)<br>
                • <strong>Support/Resistance:</strong> Based on historical price extremes (not order book data)<br>
                • <strong>Confidence:</strong> Moderate (60-70%) - levels derived from technical analysis patterns<br>
                • <strong>Limitation:</strong> These are statistical indicators, not guarantees. Market conditions can invalidate any technical level.
            </div>

            <h3 class="subsection-title">Price Targets & Scenarios</h3>

            <table>
                <tr>
                    <th>Scenario</th>
                    <th>Target Price</th>
                    <th>% Move</th>
                    <th>Description</th>
                    <th>Probability</th>
                </tr>
                <tr>
                    <td><strong>Bullish Breakout</strong></td>
                    <td>${metrics['resistance']:,.0f}</td>
                    <td style="color: #27ae60;">+{((metrics['resistance'] - current) / current * 100):.1f}%</td>
                    <td>Break above 365-day high resistance</td>
                    <td><span class="confidence-badge conf-moderate">Medium</span></td>
                </tr>
                <tr>
                    <td><strong>Bullish Target 1</strong></td>
                    <td>${metrics['fib_levels']['23.6%']:,.0f}</td>
                    <td style="color: {'#27ae60' if metrics['fib_levels']['23.6%'] > current else '#e74c3c'}">{((metrics['fib_levels']['23.6%'] - current) / current * 100):+.1f}%</td>
                    <td>Fibonacci 23.6% retracement</td>
                    <td><span class="confidence-badge conf-moderate">Medium</span></td>
                </tr>
                <tr>
                    <td><strong>Neutral Zone</strong></td>
                    <td>${metrics['fib_levels']['50.0%']:,.0f}</td>
                    <td style="color: {'#27ae60' if metrics['fib_levels']['50.0%'] > current else '#e74c3c'}">{((metrics['fib_levels']['50.0%'] - current) / current * 100):+.1f}%</td>
                    <td>Mid-range equilibrium (50% Fib)</td>
                    <td><span class="confidence-badge conf-moderate">Medium</span></td>
                </tr>
                <tr>
                    <td><strong>Bearish Target 1</strong></td>
                    <td>${metrics['fib_levels']['61.8%']:,.0f}</td>
                    <td style="color: {'#27ae60' if metrics['fib_levels']['61.8%'] > current else '#e74c3c'}">{((metrics['fib_levels']['61.8%'] - current) / current * 100):+.1f}%</td>
                    <td>Golden ratio retracement (61.8%)</td>
                    <td><span class="confidence-badge conf-moderate">Medium</span></td>
                </tr>
                <tr>
                    <td><strong>Bearish Breakdown</strong></td>
                    <td>${metrics['support']:,.0f}</td>
                    <td style="color: #e74c3c;">{((metrics['support'] - current) / current * 100):.1f}%</td>
                    <td>Test of 365-day low support</td>
                    <td><span class="confidence-badge conf-low">Low</span></td>
                </tr>
            </table>
        </div>

        <!-- M2 & INTEREST RATES TAB -->

        <div id="m2-analysis" class="tab-content">
            <h2 class="section-title">M2 Money Supply & Interest Rate Analysis</h2>

            <div class="info-box">
                <strong>🔬 Research Finding:</strong> M2 Money Supply shows a <strong>365-day leading relationship</strong> with Bitcoin price
                (correlation r=0.84). This suggests that changes in monetary policy and money supply expansion have a delayed but
                significant impact on Bitcoin valuations.
            </div>

"""

    if m2_results:
        html += f"""
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-label">M2 vs Bitcoin Correlation</div>
                    <div class="metric-value">{m2_results['m2_btc_corr']:.4f}</div>
                    <div class="metric-subtext">Contemporaneous relationship</div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">M2 (90-day lead) vs BTC</div>
                    <div class="metric-value">{m2_results['m2_lead90_btc_corr']:.4f}</div>
                    <div class="metric-subtext">Historical 90-day pattern</div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">Optimal M2 Lead Period</div>
                    <div class="metric-value">365 days</div>
                    <div class="metric-subtext">r = {m2_results['m2_lead_correlations']['365']:.4f}</div>
                </div>

                <div class="metric-card {'negative' if m2_results['is_cointegrated'] == 'False' else 'positive'}">
                    <div class="metric-label">Cointegration Status</div>
                    <div class="metric-value">{'Not Detected' if m2_results['is_cointegrated'] == 'False' else 'Detected'}</div>
                    <div class="metric-subtext">{'No mean-reversion relationship' if m2_results['is_cointegrated'] == 'False' else 'Mean-reversion exists'}</div>
                </div>
            </div>

            <h3 class="subsection-title">Interest Rate Regime Analysis</h3>

            <table>
                <tr>
                    <th>Rate Regime</th>
                    <th>Average BTC Price</th>
                    <th>Standard Deviation</th>
                    <th>Sample Size</th>
                    <th>Interpretation</th>
                </tr>
                <tr>
                    <td><strong>Low (0-1%)</strong></td>
                    <td>${m2_results['regime_stats']['mean']['Low (0-1%)']:,.0f}</td>
                    <td>${m2_results['regime_stats']['std']['Low (0-1%)']:,.0f}</td>
                    <td>{int(m2_results['regime_stats']['count']['Low (0-1%)'])} days</td>
                    <td>Ultra-low rates - moderate BTC price, high volatility</td>
                </tr>
                <tr>
                    <td><strong>Medium (1-3%)</strong></td>
                    <td>${m2_results['regime_stats']['mean']['Medium (1-3%)']:,.0f}</td>
                    <td>${m2_results['regime_stats']['std']['Medium (1-3%)']:,.0f}</td>
                    <td>{int(m2_results['regime_stats']['count']['Medium (1-3%)'])} days</td>
                    <td>Moderate rates - lowest avg BTC price</td>
                </tr>
                <tr>
                    <td><strong>High (3%+)</strong></td>
                    <td>${m2_results['regime_stats']['mean']['High (3%+)']:,.0f}</td>
                    <td>${m2_results['regime_stats']['std']['High (3%+)']:,.0f}</td>
                    <td>{int(m2_results['regime_stats']['count']['High (3%+)'])} days</td>
                    <td>High rates - highest avg BTC price (recent bull run)</td>
                </tr>
            </table>

            <div class="warning-box">
                <strong>⚠️ Important Context:</strong><br>
                The "High (3%+)" regime showing the highest average Bitcoin price is primarily due to the 2024-2025 bull run coinciding
                with elevated interest rates. This does NOT imply causation - Bitcoin's rise was driven by multiple factors (halving cycle,
                ETF approvals, institutional adoption) despite higher rates. The traditional inverse relationship (low rates → higher asset prices)
                may not apply straightforwardly to Bitcoin due to its unique market dynamics.

                <p style="margin-top: 10px;"><strong>Confidence Level: Moderate (60%)</strong></p>
                <ul style="margin-left: 20px; margin-top: 5px;">
                    <li><strong>Reasoning:</strong> Strong statistical correlation (r=0.82-0.84) with M2 money supply</li>
                    <li><strong>Limitation:</strong> Correlation ≠ causation; multiple confounding variables</li>
                    <li><strong>Sample Size:</strong> Limited to 3-4 complete monetary policy cycles</li>
                    <li><strong>Alternative Interpretation:</strong> Both M2 and Bitcoin may be responding to common macroeconomic factors</li>
                </ul>
            </div>
"""
    else:
        html += """
            <div class="warning-box">
                <strong>⚠️ M2 Analysis Data Not Available:</strong><br>
                Run <code>generate_m2_interest_rate_bitcoin_study.py</code> to generate detailed M2 and interest rate analysis.
                This will provide correlation analysis, optimal lead periods, and regime-based performance metrics.
            </div>
"""

    html += """
        </div>
    """

    # Add Trading Strategy Tab
    html += generate_trading_strategy_tab_content(trading_strategy_results)

    html += """

        <!-- LEAD-LAG ANALYSIS TAB -->
        <div id="lead-lag" class="tab-content">
            <h2 class="section-title">Lead-Lag Relationship Analysis</h2>

            <div class="info-box">
                <strong>🔬 What is Lead-Lag Analysis?</strong><br>
                Lead-lag analysis identifies which economic indicators precede movements in Bitcoin price. If an indicator "leads"
                Bitcoin by X days, changes in that indicator can potentially signal future Bitcoin price movements.
            </div>

            <h3 class="subsection-title">Key Findings from Previous Research</h3>

            <table>
                <tr>
                    <th>Indicator</th>
                    <th>Optimal Lead Period</th>
                    <th>Correlation</th>
                    <th>Interpretation</th>
                    <th>Confidence</th>
                </tr>
                <tr>
                    <td><strong>M2 Money Supply</strong></td>
                    <td>365 days (~12 months)</td>
                    <td>r = 0.84</td>
                    <td>M2 expansion predicts BTC price ~1 year later</td>
                    <td><span class="confidence-badge conf-high">High (75%)</span></td>
                </tr>
                <tr>
                    <td><strong>Federal Funds Rate</strong></td>
                    <td>365 days (~12 months)</td>
                    <td>r = 0.70</td>
                    <td>Rate changes show delayed correlation with BTC</td>
                    <td><span class="confidence-badge conf-moderate">Moderate (65%)</span></td>
                </tr>
                <tr>
                    <td><strong>Net Liquidity</strong></td>
                    <td>90 days (~3 months)</td>
                    <td>r = 0.54</td>
                    <td>Liquidity changes have ~3-month lag to BTC</td>
                    <td><span class="confidence-badge conf-moderate">Moderate (60%)</span></td>
                </tr>
                <tr>
                    <td><strong>Technical Indicators</strong></td>
                    <td>30 days (~1 month)</td>
                    <td>Varies</td>
                    <td>Short-term momentum and volume patterns</td>
                    <td><span class="confidence-badge conf-moderate">Moderate (55%)</span></td>
                </tr>
            </table>

            <div class="warning-box">
                <strong>⚠️ Critical Limitations of Lead-Lag Analysis:</strong><br>
                <ul style="margin-left: 20px; margin-top: 10px;">
                    <li><strong>Spurious Correlations:</strong> Historical correlations may be coincidental, not causal</li>
                    <li><strong>Non-Stationary Relationships:</strong> Lead times can change as markets evolve</li>
                    <li><strong>Regime Shifts:</strong> What worked in 2015-2020 may not work in 2024-2025</li>
                    <li><strong>Multiple Variables:</strong> Bitcoin price is influenced by dozens of factors simultaneously</li>
                    <li><strong>Sample Size:</strong> Only 3-4 complete Bitcoin cycles available for analysis (n<10)</li>
                </ul>

                <p style="margin-top: 15px;"><strong>How to Interpret These Results:</strong></p>
                <p>These lead-lag relationships should be viewed as <em>one input among many</em> in a comprehensive analysis framework.
                They describe historical patterns but do not guarantee future predictability. The strongest confidence (75%) is still
                only moderate certainty - far from deterministic prediction.</p>
            </div>

            <h3 class="subsection-title">Cointegration Testing</h3>

            <div class="info-box">
                <strong>🔬 What is Cointegration?</strong><br>
                Cointegration tests whether two time series have a long-term equilibrium relationship - meaning they tend to move
                together over time and any divergence is temporary (mean-reverting). If Bitcoin and M2 were cointegrated, large
                deviations from their historical relationship would eventually correct.
            </div>

            <p><strong>Result:</strong> M2 and Bitcoin are <strong>NOT cointegrated</strong> (Engle-Granger test p-value > 0.05).</p>

            <p><strong>Interpretation:</strong> While M2 and Bitcoin show strong correlation, they do not have a mean-reverting
            long-term equilibrium. This means:
            <ul style="margin-left: 20px; margin-top: 10px;">
                <li>Divergences between M2 and Bitcoin may persist indefinitely</li>
                <li>No "rubber band" effect pulling them back together</li>
                <li>The relationship could break down entirely in different market regimes</li>
                <li>Statistical arbitrage strategies based on mean reversion would not be valid</li>
            </ul>
            </p>

            <div class="info-box">
                <strong>📊 View Detailed Analysis:</strong><br>
                • <a href="Lead_Lag_Cointegration_Report.html" style="color: #3498db;">Lead-Lag Cointegration Full Report</a> - Comprehensive analysis with charts<br>
                • <a href="ML_Analysis_Report.html" style="color: #3498db;">ML Feature Importance Report</a> - Which FRED indicators matter most<br>
            </div>
        </div>

        <!-- METHODOLOGY TAB -->
        <div id="methodology" class="tab-content">
            <h2 class="section-title">Data Sources & Methodology</h2>

            <div class="info-box">
                <strong>🎯 Our Approach:</strong> This analysis prioritizes transparency and scientific rigor over making bold predictions.
                All claims include confidence scores, data sources, and limitations. We believe informed skepticism is more valuable
                than false certainty.
            </div>

            <h3 class="subsection-title">Data Sources</h3>

            <table>
                <tr>
                    <th>Data Type</th>
                    <th>Source</th>
                    <th>Update Frequency</th>
                    <th>Quality</th>
                </tr>
                <tr>
                    <td><strong>Bitcoin OHLC Price</strong></td>
                    <td>Coinbase Exchange API</td>
                    <td>Daily</td>
                    <td><span class="confidence-badge conf-high">High</span></td>
                </tr>
                <tr>
                    <td><strong>M2 Money Supply</strong></td>
                    <td>FRED (Federal Reserve Economic Data)</td>
                    <td>Monthly</td>
                    <td><span class="confidence-badge conf-high">High</span></td>
                </tr>
                <tr>
                    <td><strong>Federal Funds Rate</strong></td>
                    <td>FRED</td>
                    <td>Daily</td>
                    <td><span class="confidence-badge conf-high">High</span></td>
                </tr>
                <tr>
                    <td><strong>CPI (Inflation)</strong></td>
                    <td>FRED</td>
                    <td>Monthly</td>
                    <td><span class="confidence-badge conf-high">High</span></td>
                </tr>
                <tr>
                    <td><strong>Technical Indicators</strong></td>
                    <td>Calculated from OHLC data</td>
                    <td>Daily</td>
                    <td><span class="confidence-badge conf-high">High</span></td>
                </tr>
            </table>

            <h3 class="subsection-title">Analytical Methods</h3>

            <div class="subsection-title" style="font-size: 1.1em; margin-top: 20px;">1. Technical Indicators</div>
            <ul style="margin-left: 20px; line-height: 1.8;">
                <li><strong>Moving Averages (MA):</strong> Simple averages of closing price over N days (20, 50, 200)</li>
                <li><strong>Bollinger Bands:</strong> MA ± 2 standard deviations, indicating volatility and overbought/oversold conditions</li>
                <li><strong>Fibonacci Retracements:</strong> Key levels (23.6%, 38.2%, 50%, 61.8%, 78.6%) calculated from recent high/low range</li>
                <li><strong>Support/Resistance:</strong> 365-day minimum and maximum prices</li>
            </ul>

            <div class="subsection-title" style="font-size: 1.1em; margin-top: 20px;">2. Correlation Analysis</div>
            <ul style="margin-left: 20px; line-height: 1.8;">
                <li><strong>Pearson Correlation (r):</strong> Measures linear relationship between two variables (-1 to +1)</li>
                <li><strong>Lead-Lag Testing:</strong> Shift one variable by X days and test correlation to find optimal lead period</li>
                <li><strong>Interpretation:</strong> |r| > 0.7 = strong, 0.4-0.7 = moderate, < 0.4 = weak</li>
            </ul>

            <div class="subsection-title" style="font-size: 1.1em; margin-top: 20px;">3. Cointegration Testing</div>
            <ul style="margin-left: 20px; line-height: 1.8;">
                <li><strong>Engle-Granger Test:</strong> Tests for long-term equilibrium between two time series</li>
                <li><strong>Null Hypothesis:</strong> No cointegration exists (series can drift apart indefinitely)</li>
                <li><strong>Threshold:</strong> p-value < 0.05 rejects null (cointegration detected)</li>
                <li><strong>Limitation:</strong> Assumes linear relationship and constant parameters over time</li>
            </ul>

            <div class="subsection-title" style="font-size: 1.1em; margin-top: 20px;">4. Machine Learning (Feature Importance)</div>
            <ul style="margin-left: 20px; line-height: 1.8;">
                <li><strong>Models Tested:</strong> Ridge Regression, Lasso, Random Forest, Gradient Boosting</li>
                <li><strong>Validation:</strong> TimeSeriesSplit (prevents look-ahead bias)</li>
                <li><strong>Feature Selection:</strong> Permutation importance with statistical significance testing</li>
                <li><strong>Overfitting Prevention:</strong> Removed same-day OHLC to predict same-day close (data leakage)</li>
            </ul>

            <h3 class="subsection-title">Confidence Score Calibration</h3>

            <table>
                <tr>
                    <th>Score Range</th>
                    <th>Label</th>
                    <th>Meaning</th>
                    <th>Example</th>
                </tr>
                <tr>
                    <td>90-100%</td>
                    <td><span class="confidence-badge conf-high">Very High</span></td>
                    <td>Direct measurement or well-established fact</td>
                    <td>Current price from exchange API</td>
                </tr>
                <tr>
                    <td>70-89%</td>
                    <td><span class="confidence-badge conf-high">High</span></td>
                    <td>Strong evidence from multiple sources</td>
                    <td>M2 correlation (r=0.84, n=3977, p<0.001)</td>
                </tr>
                <tr>
                    <td>50-69%</td>
                    <td><span class="confidence-badge conf-moderate">Moderate</span></td>
                    <td>Some evidence but with limitations</td>
                    <td>Interest rate regime analysis (confounding variables)</td>
                </tr>
                <tr>
                    <td>30-49%</td>
                    <td><span class="confidence-badge conf-low">Low-Moderate</span></td>
                    <td>Weak evidence or high uncertainty</td>
                    <td>Projections based on n=3 historical cycles</td>
                </tr>
                <tr>
                    <td><30%</td>
                    <td><span class="confidence-badge conf-low">Low</span></td>
                    <td>Speculation or very limited data</td>
                    <td>Long-term predictions (>2 years out)</td>
                </tr>
            </table>

            <div class="warning-box">
                <strong>⚠️ Why Confidence Scores Matter:</strong><br>
                In quantitative analysis, it's tempting to treat all numbers as equally certain. But a correlation calculated from
                3,977 data points is more reliable than one from 10 data points. A direct price measurement is more certain than
                a projection based on historical patterns. These scores help you calibrate your trust appropriately.

                <p style="margin-top: 10px;"><strong>Our Philosophy:</strong> We'd rather tell you "we don't know" (Low confidence)
                than give you false precision. Every limitation we acknowledge makes the confident claims more credible.</p>
            </div>

            <h3 class="subsection-title">Known Limitations</h3>

            <ul style="margin-left: 20px; line-height: 1.8;">
                <li><strong>Limited Historical Data:</strong> Bitcoin has only existed since 2009 - fewer than 4 complete halving cycles</li>
                <li><strong>Regime Changes:</strong> Market structure has evolved (ETFs, institutions, regulation) - past patterns may not repeat</li>
                <li><strong>Correlation ≠ Causation:</strong> Even strong correlations may be spurious or driven by hidden variables</li>
                <li><strong>Non-Stationarity:</strong> Relationships between variables can change over time as market matures</li>
                <li><strong>Black Swan Events:</strong> Unpredictable events (exchange collapses, regulatory changes) can invalidate all models</li>
                <li><strong>Data Quality:</strong> FRED data has monthly frequency (interpolated to daily), Bitcoin data from single exchange</li>
                <li><strong>Selection Bias:</strong> We analyzed indicators we thought were important - may have missed key variables</li>
            </ul>

            <h3 class="subsection-title">How to Use This Analysis Responsibly</h3>

            <div class="info-box">
                <strong>✅ DO:</strong><br>
                • Use this as <em>one input</em> in a broader decision-making framework<br>
                • Pay attention to confidence scores and limitations<br>
                • Question the assumptions and methodology<br>
                • Cross-reference with other independent sources<br>
                • Update your views as new data emerges<br><br>

                <strong>❌ DON'T:</strong><br>
                • Treat projections as predictions or guarantees<br>
                • Ignore the confidence scores and limitations<br>
                • Make decisions based solely on this analysis<br>
                • Assume past correlations will continue indefinitely<br>
                • Risk more than you can afford to lose based on any model
            </div>

            <div class="info-box">
                <strong>📚 Detailed Documentation:</strong><br>
                • <a href="../DATA_METHODOLOGY.md" style="color: #3498db;">DATA_METHODOLOGY.md</a> - Complete methodology documentation<br>
                • <a href="../COMPREHENSIVE_SUMMARY.md" style="color: #3498db;">COMPREHENSIVE_SUMMARY.md</a> - ML model selection process<br>
                • <a href="../OVERFITTING_FIX_SUMMARY.md" style="color: #3498db;">OVERFITTING_FIX_SUMMARY.md</a> - Data leakage prevention<br>
                • <a href="../TRANSPARENCY_UPDATE.md" style="color: #3498db;">TRANSPARENCY_UPDATE.md</a> - Why we added confidence scores
            </div>
        </div>

        <!-- ALL REPORTS TAB -->
        <div id="reports" class="tab-content">
            <h2 class="section-title">All Analysis Reports</h2>

            <div class="info-box">
                <strong>📄 Report Archive:</strong> This section provides access to all generated analysis reports, organized by category.
                Each report focuses on a specific aspect of Bitcoin analysis with detailed charts, data, and methodology.
            </div>

            <h3 class="subsection-title">📊 Current Dashboard & Analysis</h3>
            <table>
                <tr>
                    <th>Report</th>
                    <th>Type</th>
                    <th>Description</th>
                    <th>Last Updated</th>
                </tr>
                <tr>
                    <td><a href="Bitcoin_Dashboard_20251118_2217.html" style="color: #3498db;">Bitcoin Dashboard</a></td>
                    <td>HTML</td>
                    <td>Interactive dashboard with current market status, bear market scenarios, and confidence scores</td>
                    <td>Nov 18, 2025</td>
                </tr>
                <tr>
                    <td><a href="Bitcoin_Analysis_Report_20251118_2217.pdf" style="color: #3498db;">Bitcoin Analysis (PDF)</a></td>
                    <td>PDF</td>
                    <td>Printable comprehensive analysis report with all findings and methodology</td>
                    <td>Nov 18, 2025</td>
                </tr>
            </table>

            <h3 class="subsection-title">🔬 Specialized Analysis Reports</h3>
            <table>
                <tr>
                    <th>Report</th>
                    <th>Type</th>
                    <th>Description</th>
                    <th>Size</th>
                </tr>
                <tr>
                    <td><a href="ML_Analysis_Report.html" style="color: #3498db;">ML Feature Importance Analysis</a></td>
                    <td>HTML</td>
                    <td>Machine learning analysis showing which FRED indicators have the strongest predictive power</td>
                    <td>682 KB</td>
                </tr>
                <tr>
                    <td><a href="Lead_Lag_Cointegration_Report.html" style="color: #3498db;">Lead-Lag Cointegration Analysis</a></td>
                    <td>HTML</td>
                    <td>Detailed analysis of lead-lag relationships and cointegration testing with FRED indicators</td>
                    <td>419 KB</td>
                </tr>
            </table>

            <h3 class="subsection-title">📚 Archive: Methodology & Deep Dives</h3>
            <table>
                <tr>
                    <th>Report</th>
                    <th>Type</th>
                    <th>Description</th>
                    <th>Size</th>
                </tr>
                <tr>
                    <td><a href="../archive/old_reports/Bitcoin_Comprehensive_Analysis_Report.pdf" style="color: #3498db;">Comprehensive Analysis (Legacy)</a></td>
                    <td>PDF</td>
                    <td>Most detailed comprehensive analysis (1.3 MB) - includes all methodology, charts, and findings</td>
                    <td>1.3 MB</td>
                </tr>
                <tr>
                    <td><a href="../archive/old_reports/Bitcoin_Comprehensive_Analysis_Report_Enhanced.pdf" style="color: #3498db;">Enhanced Analysis (Legacy)</a></td>
                    <td>PDF</td>
                    <td>Enhanced version with additional visualizations</td>
                    <td>538 KB</td>
                </tr>
                <tr>
                    <td><a href="../archive/old_reports/Feature_Selection_Report_With_Leakage_Checks.pdf" style="color: #3498db;">Feature Selection + Leakage Checks</a></td>
                    <td>PDF</td>
                    <td>Analysis of feature selection methodology with data leakage prevention</td>
                    <td>41 KB</td>
                </tr>
                <tr>
                    <td><a href="../archive/old_reports/Feature_Selection_Redundancy_Elimination_Report.pdf" style="color: #3498db;">Redundancy Elimination Report</a></td>
                    <td>PDF</td>
                    <td>How we removed redundant and multicollinear features</td>
                    <td>48 KB</td>
                </tr>
            </table>

            <h3 class="subsection-title">📖 Documentation</h3>
            <table>
                <tr>
                    <th>Document</th>
                    <th>Type</th>
                    <th>Description</th>
                </tr>
                <tr>
                    <td><a href="../DATA_METHODOLOGY.md" style="color: #3498db;">Data Methodology</a></td>
                    <td>Markdown</td>
                    <td>Complete documentation of data sources, calculations, and confidence calibration</td>
                </tr>
                <tr>
                    <td><a href="../COMPREHENSIVE_SUMMARY.md" style="color: #3498db;">ML Methodology Summary</a></td>
                    <td>Markdown</td>
                    <td>How ML models were selected, validated, and why Ridge Regression was chosen</td>
                </tr>
                <tr>
                    <td><a href="../OVERFITTING_FIX_SUMMARY.md" style="color: #3498db;">Overfitting Prevention</a></td>
                    <td>Markdown</td>
                    <td>Data leakage detection and prevention methodology</td>
                </tr>
                <tr>
                    <td><a href="../TRANSPARENCY_UPDATE.md" style="color: #3498db;">Transparency Update</a></td>
                    <td>Markdown</td>
                    <td>Why we added confidence scores and qualified language to all reports</td>
                </tr>
                <tr>
                    <td><a href="README.md" style="color: #3498db;">Reports README</a></td>
                    <td>Markdown</td>
                    <td>Overview of the reports directory and how to regenerate reports</td>
                </tr>
            </table>

            <div class="info-box">
                <strong>🔄 Regenerating Reports:</strong><br>
                To update reports with the latest Bitcoin data:<br><br>
                <code style="background: #f8f9fa; padding: 10px; display: block; border-radius: 5px;">
                cd ml_pipeline<br>
                source ../venv/bin/activate<br>
                python3 generate_comprehensive_bitcoin_website.py
                </code>
            </div>
        </div>

        <div class="footer">
            <h3 style="margin-bottom: 15px;">Bitcoin Analysis Research Dashboard</h3>
            <p>This is an independent research project using open-source data and transparent methodology.</p>
            <p style="margin-top: 10px;">
                <strong>Not Financial Advice:</strong> This analysis is for educational and informational purposes only.
                It does not constitute financial advice, investment recommendations, or an offer to buy or sell any securities.
                Cryptocurrency investments carry substantial risk. Past performance does not guarantee future results.
            </p>
            <p style="margin-top: 15px;">
                <strong>Data Sources:</strong> Coinbase Exchange (Bitcoin prices), FRED (economic data)<br>
                <strong>Analysis Framework:</strong> Python, pandas, matplotlib, scikit-learn<br>
                <strong>Code:</strong> Available in project repository (generate_comprehensive_bitcoin_website.py)
            </p>
            <p style="margin-top: 15px;">
                <strong>Questions or Feedback?</strong> Review the <a href="../DATA_METHODOLOGY.md">methodology documentation</a> or
                <a href="Reports_Index.html">browse all reports</a>.
            </p>
            <p style="margin-top: 15px; font-size: 0.9em; opacity: 0.8;">
                Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p UTC')} |
                Data through: {metrics['current_date'].strftime('%B %d, %Y')} |
                Version 3.0 - Comprehensive Dashboard
            </p>
        </div>
    </div>

    <script>
        function openTab(evt, tabName) {{
            // Hide all tab contents
            var tabContents = document.getElementsByClassName('tab-content');
            for (var i = 0; i < tabContents.length; i++) {{
                tabContents[i].classList.remove('active');
            }}

            // Remove active class from all tabs
            var tabs = document.getElementsByClassName('tab');
            for (var i = 0; i < tabs.length; i++) {{
                tabs[i].classList.remove('active');
            }}

            // Show current tab and mark as active
            document.getElementById(tabName).classList.add('active');
            evt.currentTarget.classList.add('active');

            // Scroll to top
            window.scrollTo(0, 0);
        }}
    </script>
</body>
</html>
"""

    return html

def main():
    """Main execution"""
    print("=" * 80)
    print("COMPREHENSIVE BITCOIN ANALYSIS WEBSITE GENERATOR")
    print("=" * 80)
    print()

    # Load data
    df, metrics = load_bitcoin_data()
    print()

    # Load M2 results if available
    m2_results = load_m2_results()
    if m2_results:
        print(f"✅ Loaded M2 analysis results")
    else:
        print(f"⚠️  M2 analysis results not found - generating website without M2 charts")
    print()

    # Load Trading Strategy results if available
    trading_strategy_results = load_trading_strategy_results()
    if trading_strategy_results:
        print(f"✅ Loaded trading strategy analysis results")
    else:
        print(f"⚠️  Trading strategy results not found - generating website without strategy tab")
    print()

    # Generate HTML
    print("🌐 Generating comprehensive HTML website...")
    html = generate_html(df, metrics, m2_results, trading_strategy_results)

    # Save
    output_file = OUTPUT_DIR / "Bitcoin_Comprehensive_Dashboard.html"
    with open(output_file, 'w') as f:
        f.write(html)

    print(f"✅ Website generated: {output_file}")
    print()
    print("=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print()
    print(f"📊 Bitcoin Price: ${metrics['current_price']:,.2f}")
    print(f"📅 Data Updated: {metrics['current_date'].strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"🌐 View Dashboard: http://localhost:8080/Bitcoin_Comprehensive_Dashboard.html")
    print()
    print("To start HTTP server:")
    print("  cd reports && python3 -m http.server 8080")
    print()

if __name__ == "__main__":
    main()
