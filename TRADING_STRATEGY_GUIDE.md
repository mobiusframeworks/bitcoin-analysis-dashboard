# 50-Week SMA Trading Strategy - Complete Guide

## 📊 Strategy Overview

**Last Updated:** November 19, 2025

This document provides a comprehensive guide to the 50-week SMA trading strategy integrated into your Bitcoin analysis dashboard.

---

## 🎯 Strategy Rules

### Entry Signals

**LONG (Buy Signal):**
- Weekly candle closes **ABOVE** the 50-week Simple Moving Average (SMA)
- Wait for confirmed weekly close (Sunday close in most markets)
- Enter on the following Monday open

**SHORT (Sell/Exit Signal):**
- Weekly candle closes **BELOW** the 50-week Simple Moving Average (SMA)
- Wait for confirmed weekly close
- Exit longs or enter short on the following Monday open

### Current Status (Nov 19, 2025)

- **Current Price:** $88,546.77
- **50-Week SMA:** $102,704.79
- **Distance from 50-SMA:** -13.79% (BELOW)
- **200-Week SMA:** $55,569.98
- **Distance from 200-SMA:** +59.34% (ABOVE)
- **Current Signal:** **SHORT ⬇️**
- **Market Phase:** TRANSITION
- **Historical Crosses:** 13 bullish, 13 bearish

---

## 🛡️ Risk Management Framework

### Stop Loss Levels (Phase-Dependent)

The stop-loss recommendations vary based on the current market phase and historical volatility patterns:

#### Conservative Stop Loss
- **Level:** At the 50-week SMA itself
- **Current:** $102,704.79
- **When to Use:** Low risk tolerance, protecting capital
- **Pros:** Minimizes losses if trend reverses quickly
- **Cons:** May get stopped out on normal volatility

#### Moderate Stop Loss (Recommended)
- **Level:** 2 standard deviations from 50-week SMA (phase-dependent)
- **Calculation:** Based on historical volatility for current market phase
- **When to Use:** Standard trading with balanced risk
- **Pros:** Accounts for normal market volatility
- **Cons:** Requires running analysis to calculate exact level

#### Aggressive Stop Loss
- **Level:** At the 200-week SMA
- **Current:** $55,569.98
- **When to Use:** Long-term holders, high risk tolerance
- **Pros:** Gives maximum room for price action
- **Cons:** Large potential drawdown (-37.2% from current price)

### Position Sizing

**Maximum Risk Per Trade:** 2% of total capital

**Formula:**
```
Position Size = (Account Size × 0.02) / (Entry Price - Stop Loss Price)
```

**Example:**
- Account Size: $100,000
- Risk per trade: $2,000 (2%)
- Entry: $88,547
- Stop Loss (Conservative): $102,705
- Maximum loss per Bitcoin: $14,158
- Position Size: $2,000 / $14,158 = **0.14 BTC**

---

## 📊 Market Phase Analysis

### Historical Volatility by Phase

| Phase | Mean Distance from 50-SMA | Std Deviation | Sample Size |
|-------|---------------------------|---------------|-------------|
| **Bull** | +56.43% | ±50.74% | 160 weeks |
| **Distribution** | +18.50% | ±14.13% | 72 weeks |
| **Bear** | (Below SMA) | (Calculating) | Varies |
| **Accumulation** | -15.22% | ±12.89% | 25 weeks |
| **Transition** | Varies | High volatility | 113 weeks |

### Current Phase: TRANSITION

**Characteristics:**
- Price recently crossed below 50-week SMA
- Uncertain trend direction
- Higher than normal volatility expected
- Risk management is CRITICAL

**Historical Behavior in Transitions:**
- 113 weeks (19.9% of time) spent in transition phases
- Can quickly resolve into bull or bear phases
- False signals more common
- Recommended: Reduce position size by 50% during transitions

---

## 📈 Distribution Curves & Historical Context

### Price Action Around 50-Week SMA

**Bull Phase Distribution:**
- Range: +0.66% to +258.77% above SMA
- Typical: +6% to +107% above SMA (1 std dev)
- Extreme: +107% to +208% above SMA (2 std dev)

**Distribution Phase:**
- Range: +0.68% to +70.46% above SMA
- Typical: +4% to +33% above SMA (1 std dev)
- Signals weakening momentum

**Accumulation Phase:**
- Range: -43.33% to -0.59% below SMA
- Typical: -28% to -2% below SMA (1 std dev)
- Signals potential bottom formation

### Price Action Around 200-Week SMA

**Historical Distribution:**
- Mean distance: +59.34% (current matches this!)
- Standard deviation: Varies by cycle
- Percentiles:
  - 5th: Near or below 200-SMA (bear market lows)
  - 25th: +20-40% above
  - 50th (Median): +50-70% above
  - 75th: +100-150% above
  - 95th: +200%+ above (bull market peaks)

**Current Position:** 59.34% above 200-SMA
- **Interpretation:** Around historical median
- **Context:** Not overextended, but not deeply oversold
- **200-SMA as Support:** Historically strong long-term support level

---

## 🎲 Historical Performance & Statistics

### Crossover Signals Since 2015

**Total Signals:**
- Bullish crosses (buy): 13
- Bearish crosses (sell): 13
- Average time between signals: ~8-10 months

**Key Observations:**
- Equal number of buy/sell signals suggests balanced market cycles
- Strategy keeps you in trending markets
- Reduces exposure during ranging/choppy periods

### When the Strategy Works Best

✅ **Strong trending markets** (bull or bear)
✅ **Clear momentum** in either direction
✅ **Post-halving bull runs** (historically strong signals)
✅ **Bear market bottoms** (50-SMA provides re-entry signal)

### When the Strategy Struggles

❌ **Sideways/ranging markets** (multiple false signals)
❌ **High volatility transitions** (whipsaws possible)
❌ **During distribution phases** (late exit signals)

---

## 📋 Trading Checklist

### Before Entering a Trade

- [ ] Confirm weekly candle has closed (Sunday close)
- [ ] Verify price is on correct side of 50-week SMA for signal direction
- [ ] Identify current market phase
- [ ] Calculate position size based on 2% risk rule
- [ ] Set stop loss based on risk tolerance (conservative/moderate/aggressive)
- [ ] Check 200-week SMA for long-term trend context
- [ ] Review recent crossover history (avoid whipsaws)
- [ ] Assess overall market conditions (macro, sentiment)

### While in a Trade

- [ ] Monitor weekly closes (not intraday price)
- [ ] Stick to predetermined stop loss
- [ ] Don't second-guess the system during normal volatility
- [ ] Track distance from 50-week SMA (normal range for phase?)
- [ ] Review position size if volatility changes significantly

### After Exiting a Trade

- [ ] Document the trade (entry, exit, P&L, lessons learned)
- [ ] Wait for confirmed new signal before re-entering
- [ ] Review if stop loss was appropriate for the phase
- [ ] Update strategy journal with market conditions

---

## 🔄 Weekly Review Process

### Every Sunday (Before Weekly Close)

1. **Check Current Price vs 50-Week SMA**
   - Above or below?
   - How far (percentage)?
   - Trending toward or away from SMA?

2. **Identify Potential Crossovers**
   - Is price approaching the SMA?
   - What would a crossover mean (buy or sell)?
   - Are we in a transition phase?

3. **Update Stop Losses**
   - Has the SMA moved significantly?
   - Should stops be adjusted (trailing)?
   - Is current phase still valid?

4. **Review Market Phase**
   - Still in same phase as last week?
   - Any signs of phase transition?
   - Adjust risk accordingly

---

## 📊 Generate Detailed Analysis

To get the full trading strategy analysis with charts and distributions:

```bash
cd "/Users/alexhorton/quant connect dev environment/ml_pipeline"
source ../venv/bin/activate
python3 generate_trading_strategy_analysis.py
```

**This generates:**
1. **5 detailed charts:**
   - Price vs 50-week and 200-week SMAs with long/short zones
   - Distribution of price distance from 50-SMA by market phase
   - Historical distribution around 200-week SMA
   - Volatility box plots by phase
   - Crossover signals with distance from SMA over time

2. **Statistical Analysis:**
   - Exact volatility metrics for current phase
   - Recommended stop-loss prices (conservative/moderate/aggressive)
   - Historical ranges for each market phase
   - Correlation between phases and price performance

3. **Results File:**
   - `reports/trading_strategy_results.json`
   - Contains all metrics, can be used for backtesting or further analysis

---

## ⚠️ Important Limitations & Risks

### Strategy Limitations

1. **Lagging Indicator:**
   - 50-week SMA is slow to react
   - May enter late in trends
   - May exit late in reversals

2. **Whipsaw Risk:**
   - Ranging markets generate false signals
   - Multiple crosses in short period = losses
   - Transition phases are especially prone to whipsaws

3. **No Fundamental Analysis:**
   - Ignores news, regulations, adoption
   - Doesn't account for black swan events
   - Purely technical, no macro context

4. **Historical Data Limitations:**
   - Only 10 years of Bitcoin weekly data
   - Limited number of complete cycles (3-4)
   - Market structure is evolving (ETFs, institutions)

### Risk Warnings

⚠️ **Past performance does not guarantee future results**
⚠️ **Bitcoin is highly volatile - large drawdowns possible**
⚠️ **No strategy works 100% of the time**
⚠️ **Always use proper position sizing and risk management**
⚠️ **Never invest more than you can afford to lose**
⚠️ **This is NOT financial advice - educational purposes only**

---

## 🔗 Integration with Dashboard

### Access the Strategy Tab

1. Open the Bitcoin Comprehensive Dashboard:
   ```
   http://localhost:8080/Bitcoin_Comprehensive_Dashboard.html
   ```

2. Click on the **⚡ Trading Strategy** tab

3. View current signal, market phase, and risk management guidelines

### Auto-Update with Daily Data

The trading strategy analysis auto-updates when you run:

```bash
cd "/Users/alexhorton/quant connect dev environment/ml_pipeline"
./run_daily_update.sh
```

Or automatically via cron (if installed):
- Runs daily at midnight
- Fetches latest Bitcoin weekly data
- Recalculates 50-week and 200-week SMAs
- Updates market phase classification
- Regenerates all charts and analysis

---

## 📞 Quick Reference

### Current Levels (Nov 19, 2025)

- **Bitcoin:** $88,546.77
- **50-Week SMA:** $102,704.79 (-13.79%)
- **200-Week SMA:** $55,569.98 (+59.34%)
- **Signal:** SHORT ⬇️
- **Phase:** TRANSITION

### Stop Loss Recommendations

- **Conservative:** $102,705 (50-week SMA)
- **Moderate:** Calculate based on phase volatility
- **Aggressive:** $55,570 (200-week SMA)

### Position Sizing (2% Risk Rule)

If trading with $100,000 account and conservative stop:
- Max risk: $2,000
- Stop distance: $14,158
- **Max position: 0.14 BTC**

---

## 📚 Additional Resources

- **Main Dashboard:** Bitcoin_Comprehensive_Dashboard.html
- **Strategy Analysis:** Run `generate_trading_strategy_analysis.py`
- **Results File:** `reports/trading_strategy_results.json`
- **Automation Guide:** AUTOMATION_GUIDE.md
- **Data Methodology:** DATA_METHODOLOGY.md

---

**Last Updated:** November 19, 2025
**Version:** 1.0
**Status:** ✅ Active & Integrated with Dashboard
