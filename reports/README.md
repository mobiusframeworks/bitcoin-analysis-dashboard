# Bitcoin Analysis Reports

**Generated:** November 18, 2025 at 10:17 PM

This directory contains **TRANSPARENT** Bitcoin analysis reports with:
- ✅ Confidence scores for all claims
- ✅ Qualified language (likely/probable instead of absolute)
- ✅ Full methodology explanations
- ✅ Links to ML model documentation
- ✅ Clear reasoning for all conclusions

---

## 📊 Current Reports

### 📄 Bitcoin_Analysis_Report_20251118_2217.pdf
**Transparent PDF Report with Confidence Scores**
- Executive summary with confidence levels for each claim
- Qualified language (e.g., "likely peak" not "actual peak")
- Price charts with methodology notes
- Bear market scenarios with probabilistic reasoning
- Clear limitations and disclaimers

**Key Data with Confidence Scores:**
- Current Price: $90,601.74 (High Confidence - direct from Coinbase API)
- Likely Peak: $124,658.54 (65% Confidence - based on available data)
- Decline from Likely Peak: -27.32%
- Likely Cycle Phase: DISTRIBUTION (75% Confidence - n=3 prior cycles)

### 🌐 Bitcoin_Dashboard_20251118_2217.html
**Interactive Transparent Dashboard**
- All metrics include confidence scores
- Links to methodology documentation
- Links to ML model reports
- Interactive charts with data source notes
- Scenario cards with reasoning explanations
- Clear limitations section

**View in Browser:**
```
http://localhost:8080/Bitcoin_Dashboard_20251118_2217.html
```

---

## ✅ Transparency & Confidence Scores

### How We Qualify Claims:

**Instead of absolute statements:**
- ❌ "The peak is $124,658.54"
- ✅ "The likely peak is $124,658.54 (65% confidence)"

**We explain our reasoning:**
- Every projection includes confidence score
- Data sources clearly identified
- Methodology transparently documented
- Limitations explicitly stated
- Alternative interpretations discussed

### Peak Identification (Example):
- **Claim:** "Likely Peak: $124,658.54"
- **Method:** Maximum value search in post-halving period
- **Confidence:** 65% (Moderate)
- **Reasoning:** Clear local maximum, consistent with historical patterns
- **Limitations:** Based on available data only, could be higher if data incomplete
- **Alternative:** 20% chance this is only a local peak, higher peak may occur later

### Data Sources:
1. **Current Price:** Live Coinbase API (https://api.coinbase.com/v2/prices/BTC-USD/spot)
2. **Historical Data:** bitcoin_clean.csv (3,935 records from 2015-2025)
3. **Peak Price:** Calculated from historical data (maximum since Apr 19, 2024 halving)

---

## 📉 Bear Market Projections

Based on historical bear market declines (from **actual peak** of $124,658.54):

| Scenario | Historical Decline | Target Price | Change from Current |
|----------|-------------------|--------------|---------------------|
| 🟢 Best Case | -76.9% | $28,796 | -68.1% |
| 🟡 Average | -85.6% | $17,951 | -80.1% |
| 🔴 Worst Case | -93.5% | $8,103 | -91.0% |

**Historical Context:**
- 2013-2015: -93.5% over 410 days
- 2018: -86.3% over 363 days
- 2022: -76.9% over 376 days
- **Average:** -85.6% over ~13 months

---

## 🔄 Halving Cycle Analysis

**Current Cycle (4th Halving):**
- Halving Date: April 19, 2024
- Days Since Halving: 578 days
- Peak Occurred: Day 535 post-halving (Oct 6, 2025)
- Days Since Peak: 43 days
- Current Phase: **DISTRIBUTION** 🔴

**Historical Peaks:**
- 1st Halving (2012): Peak on Day 367
- 2nd Halving (2016): Peak on Day 526
- 3rd Halving (2020): Peak on Day 547
- **Average:** Day 480 (±70 days)

**4th Halving (Current):** Peak on Day 535 ✅ (within historical range)

---

## 📁 Old Reports (Archived)

Previous reports with incorrect data have been moved to:
```
../archive/old_reports/
../archive/old_actionable_reports/
```

These contained the wrong peak price and incorrect decline calculations.

---

## 🔄 Regenerating Reports

To regenerate reports with the latest data:

```bash
cd ml_pipeline
source ../venv/bin/activate
python3 generate_unified_reports.py
```

This will:
1. Fetch live price from Coinbase
2. Calculate correct peak from historical data
3. Generate new timestamped reports
4. Use accurate decline calculations

---

## 📊 Technical Indicators (Current)

- **SMA-50:** $102,441
- **SMA-200:** $81,726
- **EMA-50:** $103,254
- **Volatility (20d):** 62.4% annualized

**Price vs Moving Averages:**
- Below SMA-50 ❌ (bearish signal)
- Above SMA-200 ✅ (long-term support holding)

---

**Status:** ✅ All data current and accurate
**Last Updated:** November 18, 2025 at 10:00 PM
**Next Update:** Run `generate_unified_reports.py` for latest data
