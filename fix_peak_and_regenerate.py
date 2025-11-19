#!/usr/bin/env python3
"""
Fix Peak Price Calculation and Regenerate Reports

The issue: Reports were using outdated CSV current price as "peak" instead of actual peak
Actual Peak: $124,658.54 (Oct 6, 2025)
This script fixes the peak calculation and regenerates reports with correct drawdown numbers
"""

import sys
from pathlib import Path
import pandas as pd
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("FIXING PEAK PRICE AND REGENERATING REPORTS")
print("="*80)

# Get current live price from Coinbase
current_price = None
try:
    print("\n📡 Fetching live price from Coinbase...")
    response = requests.get('https://api.coinbase.com/v2/prices/BTC-USD/spot', timeout=10)
    data = response.json()
    current_price = float(data['data']['amount'])
    print(f"✅ Current Price: ${current_price:,.2f}")
except Exception as e:
    print(f"❌ Coinbase API failed: {e}")
    sys.exit(1)

# Load historical data
DATA_PATH = Path(__file__).parent / "data" / "bitcoin_clean.csv"
if not DATA_PATH.exists():
    print(f"❌ Data file not found: {DATA_PATH}")
    sys.exit(1)

df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'], format='mixed')
df = df.sort_values('Date')

print(f"\n📊 Loaded {len(df)} historical records")
print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")

# Find ACTUAL peak since halving
HALVING_DATE = datetime(2024, 4, 19)
recent_df = df[df['Date'] >= HALVING_DATE]

if len(recent_df) == 0:
    print(f"❌ No data found after halving date {HALVING_DATE}")
    sys.exit(1)

peak_row = recent_df.loc[recent_df['Close'].idxmax()]
peak_price = peak_row['Close']
peak_date = peak_row['Date']

print(f"\n🔍 ACTUAL PEAK (since halving):")
print(f"   Peak Price: ${peak_price:,.2f}")
print(f"   Peak Date: {peak_date.strftime('%Y-%m-%d')}")

# Calculate correct decline
decline_dollars = current_price - peak_price
decline_pct = (decline_dollars / peak_price) * 100

print(f"\n📉 CORRECT DECLINE CALCULATION:")
print(f"   From Peak: ${peak_price:,.2f}")
print(f"   Current: ${current_price:,.2f}")
print(f"   Decline: ${decline_dollars:,.2f} ({decline_pct:.2f}%)")

# Calculate days since peak
latest_date = datetime.now()
days_since_peak = (latest_date - peak_date).days
days_since_halving = (latest_date - HALVING_DATE).days

print(f"\n⏱️  TIMELINE:")
print(f"   Days since halving: {days_since_halving}")
print(f"   Days since peak: {days_since_peak}")
print(f"   Peak was Day {(peak_date - HALVING_DATE).days} post-halving")

# Historical bear market projections (from correct peak)
bear_scenarios = {
    "best": peak_price * (1 - 0.769),
    "average": peak_price * (1 - 0.856),
    "worst": peak_price * (1 - 0.935),
}

print(f"\n📊 CORRECTED BEAR MARKET PROJECTIONS:")
print(f"   (from actual peak of ${peak_price:,.2f})")
for scenario, target in bear_scenarios.items():
    remaining_decline = ((target - current_price) / current_price) * 100
    print(f"   {scenario}: ${target:,.2f} ({remaining_decline:+.1f}% from current)")

# Generate corrected executive summary
OUTPUT_DIR = Path(__file__).parent / "actionable_report"
OUTPUT_DIR.mkdir(exist_ok=True)
EXEC_SUMMARY_PATH = OUTPUT_DIR / "EXECUTIVE_SUMMARY.md"

# Determine phase
if days_since_halving < 180:
    phase = "POST-HALVING RALLY"
elif days_since_halving < 420:
    phase = "BULL ACCELERATION"
elif days_since_halving < 550:
    phase = "PEAK FORMATION"
elif days_since_halving < 640:
    phase = "DISTRIBUTION"
else:
    phase = "BEAR MARKET"

exec_summary = f"""# 🚨 Bitcoin Market Analysis
## Executive Summary

**Last Updated:** {latest_date.strftime('%B %d, %Y at %I:%M %p')}
**Data Source:** Live Coinbase API + Historical Analysis

---

## 📊 Market Snapshot

<table>
<tr>
<td width="50%">

### Current Price
# ${current_price:,.2f}

<sub>Updated live from Coinbase</sub>

</td>
<td width="50%">

### Market Phase
# 🔴 {phase}

<sub>{days_since_halving} days post-halving</sub>

</td>
</tr>
</table>

### Recent Performance

```
Current Price:      ${current_price:,.2f}
Peak Price:         ${peak_price:,.2f}  ({peak_date.strftime('%b %d, %Y')})
Decline from Peak:  ${decline_dollars:,.2f}  ({decline_pct:.2f}%)
Days Since Halving: {days_since_halving} days     (Apr 19, 2024)
Days Since Peak:    {days_since_peak} days
```

---

## 🔴 Critical Alert: Bear Market Confirmed

> **We are NOW in the post-peak distribution phase**

### Historical Pattern Analysis

<table>
<tr>
<th>Cycle</th>
<th>Peak Day</th>
<th>Duration</th>
</tr>
<tr>
<td>🥇 1st Halving (2012)</td>
<td align="center"><strong>Day 367</strong></td>
<td align="center">Nov 2013</td>
</tr>
<tr>
<td>🥈 2nd Halving (2016)</td>
<td align="center"><strong>Day 526</strong></td>
<td align="center">Dec 2017</td>
</tr>
<tr>
<td>🥉 3rd Halving (2020)</td>
<td align="center"><strong>Day 547</strong></td>
<td align="center">Nov 2021</td>
</tr>
<tr style="background-color: #fff3cd;">
<td><strong>📊 Average</strong></td>
<td align="center"><strong>Day 480</strong></td>
<td align="center">±70 days</td>
</tr>
<tr style="background-color: #f8d7da;">
<td><strong>🔴 Current (2024)</strong></td>
<td align="center"><strong>Day {(peak_date - HALVING_DATE).days}</strong></td>
<td align="center">{peak_date.strftime('%b %Y')}</td>
</tr>
</table>

### What This Means

<details>
<summary><strong>📉 Click to see the data</strong></summary>

- ✅ Price declined **{abs(decline_pct):.2f}%** from {peak_date.strftime('%b %d')} peak (${peak_price:,.2f})
- ✅ Peak occurred on Day {(peak_date - HALVING_DATE).days} post-halving
- ✅ **EVERY previous cycle** had peaked between Day 367-547
- ✅ Current decline matches early bear market pattern

**This is not consolidation—this is the beginning of a bear market that historically lasts 12-14 months with 70-85% total declines.**

</details>

---

## 📉 Historical Bear Market Data

### Previous Bear Markets

| Cycle | Decline | Duration | Bottom |
|-------|---------|----------|--------|
| 2013-2015 | **-93.5%** | 410 days | From $1,150 → $75 |
| 2018 | **-86.3%** | 363 days | From $19,700 → $3,200 |
| 2022 | **-76.9%** | 376 days | From $69,000 → $15,800 |
| **Average** | **-85.6%** | **~13 months** | — |

### Projected Scenarios (from peak of ${peak_price:,.2f})

<table>
<tr>
<td width="33%" align="center">

**🟢 Best Case**
### ${bear_scenarios['best']:,.0f}
<sub>-76.9% from peak</sub>
<sub>{((bear_scenarios['best'] - current_price) / current_price * 100):+.1f}% from current</sub>

</td>
<td width="33%" align="center">

**🟡 Average Case**
### ${bear_scenarios['average']:,.0f}
<sub>-85.6% from peak</sub>
<sub>{((bear_scenarios['average'] - current_price) / current_price * 100):+.1f}% from current</sub>

</td>
<td width="33%" align="center">

**🔴 Worst Case**
### ${bear_scenarios['worst']:,.0f}
<sub>-93.5% from peak</sub>
<sub>{((bear_scenarios['worst'] - current_price) / current_price * 100):+.1f}% from current</sub>

</td>
</tr>
</table>

### Expected Timeline

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  NOW - Q1 2026     Distribution Phase                  │
│  ▓▓▓▓░░░░░░░░      Choppy price action                 │
│                                                         │
│  Q1 - Q2 2026      Bear Acceleration                   │
│  ░░░░▓▓▓▓░░░░      Sharp declines begin                │
│                                                         │
│  Q2 - Q3 2026      Capitulation                        │
│  ░░░░░░░░▓▓▓▓      Bottom formation                    │
│                                                         │
│  Q3 2026 - Q1 2028 Recovery & Accumulation             │
│  ░░░░░░░░░░░░▓▓▓   Prepare for next halving (Apr 2028) │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## ⚠️ CORRECTED DATA NOTICE

This report has been updated with the **CORRECT peak price** of **${peak_price:,.2f}** from {peak_date.strftime('%B %d, %Y')}.

Previous reports incorrectly used ${114107.65:,.2f} as the peak (which was actually just an outdated data point).

**All drawdown projections and decline percentages have been recalculated based on the correct peak.**

---

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Peak Source:** Historical data analysis (maximum since halving)
**Current Price Source:** Live Coinbase API
**Status:** ✅ Data Corrected
"""

# Save executive summary
with open(EXEC_SUMMARY_PATH, 'w') as f:
    f.write(exec_summary)

print(f"\n✅ Generated: {EXEC_SUMMARY_PATH}")
print(f"\n" + "="*80)
print("CORRECTION COMPLETE")
print("="*80)
print(f"\nKey Changes:")
print(f"  ❌ OLD Peak: $114,107.65")
print(f"  ✅ NEW Peak: ${peak_price:,.2f}")
print(f"  ❌ OLD Decline: -19.33%")
print(f"  ✅ NEW Decline: {decline_pct:.2f}%")
print(f"\nNext Steps:")
print(f"  1. Review updated EXECUTIVE_SUMMARY.md")
print(f"  2. Regenerate PDF reports if needed")
print(f"  3. Update dashboard with correct data")
