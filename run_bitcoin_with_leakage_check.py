#!/usr/bin/env python3
"""
Run Bitcoin Prediction with Comprehensive Data Leakage Check

This version:
1. Checks for OHLC leakage (same-day open/high/low to predict close)
2. Checks for look-ahead bias
3. Checks for target leakage
4. Removes leaky features
5. Applies aggressive feature selection
6. Runs model with clean features
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from model_selection_pipeline import run_model_selection
from overfitting_detector import comprehensive_overfitting_check
from comprehensive_data_leakage_detector import (
    comprehensive_leakage_check,
    remove_leaky_features
)
from aggressive_feature_selection import aggressive_feature_selection
from feature_selector import identify_feature_categories
import pandas as pd
import numpy as np
import json
from datetime import datetime

print("="*80)
print("BITCOIN PREDICTION - WITH COMPREHENSIVE DATA LEAKAGE CHECK")
print("="*80)
print()

# Configuration
DATA_PATH = "datasets/Bitcoin/bitcoin data csvs/btc-ohlc.csv"
TARGET_LAG_DAYS = 20
TASK_TYPE = "REGRESSION"
TIME_COL = "d"
IS_TIMESERIES = True
OUTPUT_DIR = "ml_pipeline/bitcoin_results_no_leakage"

# Load data
df = pd.read_csv(DATA_PATH)
df[TIME_COL] = pd.to_datetime(df[TIME_COL], format='mixed', errors='coerce')
df = df.sort_values(TIME_COL).reset_index(drop=True)

print(f"Data shape: {df.shape}")
print(f"Date range: {df[TIME_COL].min()} to {df[TIME_COL].max()}")
print()

# Create time series features - ALL MUST BE LAGGED (past data only)
print("Creating time series features (ALL LAGGED - no same-day data)...")
print()

lag = 1

# CRITICAL: Only use LAGGED OHLC (prev_open, prev_high, prev_low, prev_close)
# DO NOT use same-day open/high/low to predict same-day close
df['prev_close'] = df['close'].shift(lag)  # Previous day's close
df['prev_open'] = df['open'].shift(lag)    # Previous day's open
df['prev_high'] = df['high'].shift(lag)    # Previous day's high
df['prev_low'] = df['low'].shift(lag)      # Previous day's low
df['prev_volume'] = df['volume'].shift(lag) # Previous day's volume

# Returns (past returns only)
df['return_1d'] = df['close'].pct_change(1).shift(lag)
df['return_5d'] = df['close'].pct_change(5).shift(lag)
df['return_20d'] = df['close'].pct_change(20).shift(lag)

# Volatility (using past data only)
df['volatility_5d'] = df['close'].shift(1).rolling(window=5).std()
df['volatility_20d'] = df['close'].shift(1).rolling(window=20).std()

# Moving averages (using past data only)
df['sma_20'] = df['close'].shift(1).rolling(window=20).mean()
df['sma_50'] = df['close'].shift(1).rolling(window=50).mean()
df['sma_200'] = df['close'].shift(1).rolling(window=200).mean()

# Price ratios (using lagged prices)
df['price_sma20_ratio'] = df['prev_close'] / df['sma_20']
df['price_sma50_ratio'] = df['prev_close'] / df['sma_50']
df['price_sma200_ratio'] = df['prev_close'] / df['sma_200']

# Volume features
df['volume_sma_20'] = df['volume'].shift(1).rolling(window=20).mean()
df['volume_ratio'] = df['prev_volume'] / df['volume_sma_20']

# Additional lagged features
for lag_days in [2, 3, 5, 10]:
    df[f'close_lag_{lag_days}'] = df['close'].shift(lag_days)

# TARGET: 20-day forward returns
target = df['close'].pct_change(TARGET_LAG_DAYS).shift(-TARGET_LAG_DAYS)

# Initial feature set
initial_features = [
    'prev_close', 'prev_open', 'prev_high', 'prev_low', 'prev_volume',
    'return_1d', 'return_5d', 'return_20d',
    'volatility_5d', 'volatility_20d',
    'sma_20', 'sma_50', 'sma_200',
    'price_sma20_ratio', 'price_sma50_ratio', 'price_sma200_ratio',
    'volume_ratio',
    'close_lag_2', 'close_lag_3', 'close_lag_5', 'close_lag_10',
]

# Try to load FRED data
fred_paths = [
    "analysis/bitcoin/results/fred_analysis/fred_bitcoin_combined.csv",
    "trading algo/bitcoin/results/fred_analysis/fred_bitcoin_combined.csv",
]

for path in fred_paths:
    if Path(path).exists():
        try:
            fred_df = pd.read_csv(path)
            fred_df['Date'] = pd.to_datetime(fred_df['Date'], format='mixed', errors='coerce')
            fred_df = fred_df.set_index('Date')
            df_with_fred = df.set_index(TIME_COL)
            for col in ['Net_Liquidity', 'M2SL', 'M2_Growth_Rate', 'WALCL', 
                       'FEDFUNDS', 'CPIAUCSL', 'CPI_YoY', 'UNRATE', 'GFDEBTN', 'DGS10']:
                if col in fred_df.columns:
                    merged = pd.merge_asof(
                        df_with_fred.reset_index(),
                        fred_df[[col]].reset_index(),
                        left_on=TIME_COL,
                        right_on='Date',
                        direction='backward',
                        tolerance=pd.Timedelta('30d')
                    )
                    df[col] = merged[col].values
                    df[col] = df[col].ffill()
                    initial_features.append(col)
            print(f"  ✓ Loaded FRED data")
            break
        except:
            continue

# Prepare data
df_features = df[initial_features + [TIME_COL]].copy()
valid_mask = ~target.isna()
df_features = df_features[valid_mask].copy()
target = target[valid_mask].copy()
df_features = df_features.dropna()
target = target.loc[df_features.index]

X = df_features[initial_features].values
y = target.values

print(f"\nInitial features: {len(initial_features)}")
print()

# COMPREHENSIVE DATA LEAKAGE CHECK
print("="*80)
print("COMPREHENSIVE DATA LEAKAGE AND LOOK-AHEAD BIAS CHECK")
print("="*80)
print()

leakage_report = comprehensive_leakage_check(
    X, y, initial_features, 
    target_col='target_return',  # We're predicting returns, not close
    time_col=TIME_COL
)

print("OHLC Leakage Check:")
if leakage_report['ohlc_leakage']['issues']:
    print("  ❌ CRITICAL ISSUES FOUND:")
    for issue in leakage_report['ohlc_leakage']['issues']:
        print(f"    - {issue['feature']}: {issue['issue']}")
else:
    print("  ✅ PASS - No OHLC leakage detected")
if leakage_report['ohlc_leakage']['warnings']:
    for warning in leakage_report['ohlc_leakage']['warnings']:
        print(f"    ⚠ {warning['feature']}: {warning['issue']}")

print("\nLook-Ahead Bias Check:")
if leakage_report['lookahead_bias']['issues']:
    print("  ❌ CRITICAL ISSUES FOUND:")
    for issue in leakage_report['lookahead_bias']['issues']:
        print(f"    - {issue['feature']}: {issue['issue']}")
else:
    print("  ✅ PASS - No look-ahead bias detected")

print("\nTarget Leakage Check:")
if leakage_report['target_leakage']['issues']:
    print("  ❌ ISSUES FOUND:")
    for issue in leakage_report['target_leakage']['issues']:
        print(f"    - {issue['feature']}: {issue['issue']}")
else:
    print("  ✅ PASS - No target leakage detected")

print("\nPerfect Correlations Check:")
if leakage_report['perfect_correlations']['issues']:
    print(f"  ⚠ Found {len(leakage_report['perfect_correlations']['issues'])} perfect correlations")
    for issue in leakage_report['perfect_correlations']['issues'][:5]:
        print(f"    - {issue['feature1']} vs {issue['feature2']}: {issue['correlation']:.4f}")
else:
    print("  ✅ PASS - No perfect correlations")

print(f"\nOverall Status: {leakage_report['overall_status']}")

if leakage_report['recommendations']:
    print("\nRecommendations:")
    for rec in leakage_report['recommendations']:
        print(f"  - {rec}")

# Remove leaky features
clean_features, removed_features = remove_leaky_features(initial_features, leakage_report)

print(f"\nRemoved {len(removed_features)} leaky features:")
for feat in removed_features:
    print(f"  - {feat}")

print(f"\nClean features remaining: {len(clean_features)}")
print(f"  {', '.join(clean_features)}")

# Save leakage report
leakage_path = Path(OUTPUT_DIR) / "data_leakage_report.json"
leakage_path.parent.mkdir(parents=True, exist_ok=True)
with open(leakage_path, 'w') as f:
    json.dump(leakage_report, f, indent=2, default=str)
print(f"\nLeakage report saved to: {leakage_path}")

# If critical issues found, stop
if leakage_report['overall_status'].startswith('FAIL'):
    print("\n" + "="*80)
    print("CRITICAL: Data leakage detected! Fixing features...")
    print("="*80)
    print()
    
    # Use clean features only
    initial_features = clean_features
    X = df_features[initial_features].values

# Apply aggressive feature selection on clean features
print("="*80)
print("APPLYING AGGRESSIVE FEATURE SELECTION (ON CLEAN FEATURES)")
print("="*80)
print()

selected_features, selection_report = aggressive_feature_selection(
    X, y,
    feature_names=initial_features,
    correlation_threshold=0.85,
    vif_threshold=5.0,
    use_rfe=True,
    n_features_final=8
)

print(f"Final selected features: {len(selected_features)}")
print(f"  {', '.join(selected_features)}")

final_categories = identify_feature_categories(selected_features)
print("\nFinal feature categories:")
for cat, features in final_categories.items():
    if features:
        print(f"  {cat}: {len(features)} - {', '.join(features)}")

# Save prepared data
prepared_data_path = Path(OUTPUT_DIR) / "bitcoin_prepared_no_leakage.csv"
df_features_selected = df_features[selected_features + [TIME_COL]].copy()
df_features_selected['target_return'] = target
df_features_selected['close'] = df.loc[df_features.index, 'close'].values
df_features_selected.to_csv(prepared_data_path, index=False)
print(f"\nSaved prepared data to: {prepared_data_path}")

# Save feature selection report
selection_report_path = Path(OUTPUT_DIR) / "feature_selection_report.json"
with open(selection_report_path, 'w') as f:
    json.dump(selection_report, f, indent=2, default=str)

# Run model selection
print("\n" + "="*80)
print("RUNNING MODEL SELECTION")
print("="*80)
print()

results, best_model, X_test, y_test, X_train_val, y_train_val = run_model_selection(
    data_path=str(prepared_data_path),
    target_col='target_return',
    task_type=TASK_TYPE,
    time_col=TIME_COL,
    is_timeseries=IS_TIMESERIES,
    output_dir=OUTPUT_DIR
)

# Comprehensive overfitting check
print("\n" + "="*80)
print("COMPREHENSIVE OVERFITTING CHECK")
print("="*80)
print()

train_val_score = results['overfitting_analysis']['after_tuning']['train_val_score']
test_score = results['overfitting_analysis']['after_tuning']['test_score']

overfitting_report = comprehensive_overfitting_check(
    X_train_val, y_train_val,
    train_val_score, test_score, test_score,
    time_col=TIME_COL, target_col='target_return',
    threshold=0.15
)

print("Data Leakage Check:")
if overfitting_report['data_leakage']['has_leakage']:
    print("  ❌ FAIL - Data leakage detected!")
else:
    print("  ✅ PASS - No data leakage detected")

print("\nOverfitting Check:")
if overfitting_report['overfitting']['is_overfitting']:
    print(f"  ❌ FAIL - Overfitting detected (Severity: {overfitting_report['overfitting']['severity']})")
else:
    print("  ✅ PASS - No significant overfitting")
print(f"  Train Score (R²): {train_val_score:.4f}")
print(f"  Test Score (R²): {test_score:.4f}")
if overfitting_report['overfitting']['train_val_gap']:
    print(f"  Train-Test Gap: {overfitting_report['overfitting']['train_val_gap']:.2%}")

print(f"\nOverall Status: {overfitting_report['overall_status']}")

# Save overfitting report
overfitting_path = Path(OUTPUT_DIR) / "overfitting_report.json"
with open(overfitting_path, 'w') as f:
    json.dump(overfitting_report, f, indent=2, default=str)

# Save combined results
combined_results = {
    'leakage_report': leakage_report,
    'feature_selection_report': selection_report,
    'model_results': results,
    'overfitting_report': overfitting_report,
    'removed_leaky_features': removed_features,
    'final_features': selected_features
}

combined_path = Path(OUTPUT_DIR) / "combined_analysis_results.json"
with open(combined_path, 'w') as f:
    json.dump(combined_results, f, indent=2, default=str)

print(f"\nCombined results saved to: {combined_path}")
print()



