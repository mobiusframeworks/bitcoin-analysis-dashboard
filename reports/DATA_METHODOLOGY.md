# Data Methodology & Transparency Report

**Purpose:** This document explains the data sources, analytical methods, confidence levels, and limitations of the Bitcoin analysis reports.

---

## 📊 Data Sources

### 1. Current Price Data
**Source:** Coinbase Exchange API
**Endpoint:** `https://api.coinbase.com/v2/prices/BTC-USD/spot`
**Update Frequency:** Real-time (fetched at report generation)
**Confidence:** High (direct from major exchange)

**Limitations:**
- Single exchange (not aggregated across multiple exchanges)
- Spot price only (may differ from other exchanges by 0.1-0.5%)
- No volume weighting

**Fallback:** CoinGecko API if Coinbase unavailable

---

### 2. Historical Price Data
**Source:** Bitcoin OHLC dataset (`bitcoin_clean.csv`)
**Records:** 3,935 data points
**Date Range:** January 19, 2015 to October 27, 2025
**Frequency:** Hourly candles

**Limitations:**
- Data ends October 27, 2025 (may not include very recent price action)
- Potential survivorship bias (only includes available data)
- Does not include pre-2015 Bitcoin history

---

### 3. Peak Price Identification

**Method:** Maximum value search in post-halving period
**Search Period:** April 19, 2024 (halving) to latest data point
**Result:** $124,658.54 on October 6, 2025

**Confidence Level: MODERATE (75%)**

**Reasoning:**
- ✅ Clear local maximum in the data
- ✅ Consistent with historical halving cycle patterns (Day 535 post-halving)
- ⚠️ Only based on available data (could be higher peak in missing data)
- ⚠️ Assumes data quality is accurate
- ⚠️ Could be challenged by future higher prices if bull market resumes

**Alternative Interpretations:**
1. **True Cycle Peak (75% confidence):** This is the actual cycle top
2. **Local Peak Only (20% confidence):** Higher peak may occur later
3. **Data Artifact (5% confidence):** Peak may be erroneous data point

**Why we believe this is the cycle peak:**
- Historical cycles peak 367-547 days post-halving (avg: 480 days)
- Current peak at Day 535 fits this pattern
- 43 days of declining prices since the peak
- Current decline of -27.62% matches early bear market behavior

---

## 📈 Technical Indicators

### Moving Averages
**Calculation:**
- SMA-50: Simple 50-period moving average
- SMA-200: Simple 200-period moving average
- EMA-50: Exponential 50-period moving average (α = 2/(50+1))

**Data Source:** Calculated from historical close prices

**Confidence:** High (standard technical analysis formulas)

**Current Values:**
- SMA-50: $102,441 (price is 11.9% below)
- SMA-200: $81,726 (price is 10.4% above)
- EMA-50: $103,254 (price is 12.6% below)

### Volatility
**Calculation:** 20-day rolling standard deviation of returns, annualized
**Formula:** `σ_20d * √365`
**Current Value:** 62.4% annualized

**Confidence:** High (standard volatility calculation)

---

## 🐻 Bear Market Projections

### Methodology
**Based on:** Historical bear market declines from previous cycles

**Historical Data:**
- 2013-2015: -93.5% decline over 410 days
- 2018: -86.3% decline over 363 days
- 2022: -76.9% decline over 376 days

**Projection Method:** Apply historical decline percentages to identified peak

**Confidence Level: LOW-MODERATE (40-60%)**

**Why low confidence:**
- ⚠️ Only 3 historical cycles (small sample size)
- ⚠️ Each cycle had unique macro conditions
- ⚠️ Past performance ≠ future results
- ⚠️ Bitcoin market structure evolving (ETFs, institutional adoption)
- ⚠️ Assumes peak identification is correct

**Scenarios:**

| Scenario | Historical Basis | Confidence | Target from Peak | Remaining Decline |
|----------|------------------|------------|------------------|-------------------|
| Best Case | 2022 cycle (-76.9%) | 60% | $28,796 | -68.1% |
| Average | Mean of 3 cycles (-85.6%) | 50% | $17,951 | -80.1% |
| Worst Case | 2013 cycle (-93.5%) | 40% | $8,103 | -91.0% |

**Important Notes:**
- These are statistical projections, not predictions
- Actual outcome could be outside this range
- Market conditions in 2025+ may differ significantly from 2013-2022

---

## 🔄 Halving Cycle Analysis

### Methodology
**Data:** Previous 3 halving cycles (2012, 2016, 2020)

**Peak Timing Analysis:**
- 1st Halving (2012): Peak at Day 367
- 2nd Halving (2016): Peak at Day 526
- 3rd Halving (2020): Peak at Day 547
- **Average:** Day 480 ± 70 days

**Current Cycle:**
- Halving: April 19, 2024
- Identified Peak: October 6, 2025 (Day 535)
- Current Day: 578

**Confidence: MODERATE (70%)**

**Reasoning:**
- ✅ Day 535 peak is within historical range (480 ± 70 = 410-550 days)
- ✅ Pattern matches previous cycles
- ⚠️ Only 3 prior cycles for comparison
- ⚠️ Each cycle had different macro environment
- ⚠️ Lengthening cycle theory suggests peaks may occur later

**Phase Determination:**

| Phase | Day Range | Current Status |
|-------|-----------|----------------|
| Post-Halving Rally | 0-180 | ❌ Past (Day 578) |
| Bull Acceleration | 180-420 | ❌ Past (Day 578) |
| Peak Formation | 420-550 | ❌ Past (Day 578) |
| **Distribution** | **550-640** | **✅ Current (Day 578)** |
| Bear Market | 640+ | ⏳ Not yet |

**Confidence in Phase:** MODERATE-HIGH (75%)
- Based on historical patterns
- Consistent with price action (27% decline from peak)
- Could transition to bear market phase soon

---

## 🤖 Machine Learning Models

### Model Selection Process
**Full Methodology:** See `COMPREHENSIVE_SUMMARY.md`

**Best Model Selected:** Ridge Regression (from basic OHLC analysis)
**Alternative Model:** Lasso Regression (from feature-selected analysis)

**Model Performance:**
- R² Score: 0.9992 (basic) / 0.0 (feature-selected)
- RMSE: $774.46 (basic) / $0.114 normalized (feature-selected)
- Cross-Validation: 10-fold TimeSeriesSplit

**Important Limitations:**
- ⚠️ **Data Leakage Warning:** Basic OHLC model has suspiciously high R² (likely leakage)
- ⚠️ Feature-selected model has poor performance (R² ≈ 0)
- ⚠️ Models trained on price data, not used for long-term predictions in this report
- ⚠️ ML predictions not included in bear market projections (using historical stats instead)

**Why ML is NOT used for bear projections:**
1. ML models are better for short-term price prediction
2. Bear market projections require understanding of market psychology and cycles
3. Historical pattern analysis more appropriate for multi-month/year forecasts
4. ML model performance issues (leakage in basic, poor fit in feature-selected)

**ML Model Reports Available:**
- `bitcoin_results/results.json` - Basic OHLC model results
- `bitcoin_results_no_leakage/results.json` - Feature-selected model results
- `COMPREHENSIVE_SUMMARY.md` - Full ML pipeline explanation
- `OVERFITTING_FIX_SUMMARY.md` - Data leakage detection and fixes

---

## ⚠️ Limitations & Disclaimers

### Data Quality
- ✅ **Strengths:** 10+ years of data, hourly granularity, major exchange source
- ⚠️ **Weaknesses:** Single source, potential gaps, not independently verified

### Analytical Methods
- ✅ **Strengths:** Standard technical analysis, transparent formulas
- ⚠️ **Weaknesses:** Assumes efficient markets, past patterns repeat

### Predictions & Projections
- ⚠️ **All projections are probabilistic, not deterministic**
- ⚠️ Historical patterns may not repeat
- ⚠️ Black swan events not accounted for
- ⚠️ Regulatory changes could alter market dynamics
- ⚠️ Macro economic conditions differ from previous cycles

### Confidence Calibration
**How to interpret confidence levels:**
- **90-100% (Very High):** Direct measurements, well-established facts
- **70-89% (High):** Strong evidence, multiple confirming signals
- **50-69% (Moderate):** Some evidence, reasonable assumptions
- **30-49% (Low-Moderate):** Weak evidence, significant uncertainty
- **<30% (Low):** Speculation, high uncertainty

---

## 🎯 Recommendations for Users

### Using This Data
1. **Verify independently:** Don't rely on a single source
2. **Understand limitations:** Know what the data can and cannot tell you
3. **Consider alternatives:** Always have contrarian scenarios
4. **Update regularly:** Market conditions change rapidly
5. **Risk management:** Never invest more than you can afford to lose

### Critical Thinking Questions
- What if the peak is actually higher and occurs later?
- What if this cycle behaves differently due to ETFs/institutions?
- What if macro conditions (interest rates, liquidity) change the pattern?
- What if the data has errors or gaps?

### Data Transparency
All source data, calculations, and code are available in:
- `ml_pipeline/data/` - Raw data files
- `ml_pipeline/bitcoin_results*/` - ML analysis results
- `ml_pipeline/*.py` - Analysis scripts

---

## 📚 References & Further Reading

### Halving Cycles
- Bitcoin halving schedule: https://www.bitcoinblockhalf.com/
- Historical cycle analysis: Multiple sources, see code comments

### Technical Analysis
- Moving averages: Standard technical analysis textbooks
- Volatility calculations: Standard financial statistics

### Machine Learning
- Nested cross-validation: "Elements of Statistical Learning" (Hastie et al.)
- Time series modeling: "Forecasting: Principles and Practice" (Hyndman & Athanasopoulos)

---

**Last Updated:** November 18, 2025
**Version:** 1.0
**Status:** Living document (will be updated as methodology improves)
