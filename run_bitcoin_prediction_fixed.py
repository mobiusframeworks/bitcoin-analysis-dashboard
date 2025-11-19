#!/usr/bin/env python3
"""
Run ML Model Selection Pipeline on Bitcoin Data - FIXED VERSION

This version:
1. Uses only past data to predict future prices (proper time series)
2. Checks for data leakage and overfitting
3. Excludes overfitted models from results
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from model_selection_pipeline import run_model_selection
from overfitting_detector import comprehensive_overfitting_check
import pandas as pd
import numpy as np
import json
from datetime import datetime

print("="*80)
print("BITCOIN PRICE PREDICTION - FIXED VERSION (NO DATA LEAKAGE)")
print("="*80)
print()

# Configuration
DATA_PATH = "datasets/Bitcoin/bitcoin data csvs/btc-ohlc.csv"
TARGET_COL = "close"
TASK_TYPE = "REGRESSION"
TIME_COL = "d"
IS_TIMESERIES = True
OUTPUT_DIR = "ml_pipeline/bitcoin_results_fixed"

# Check if data file exists
if not Path(DATA_PATH).exists():
    print(f"ERROR: Data file not found: {DATA_PATH}")
    sys.exit(1)

print(f"Loading data from: {DATA_PATH}")
print()

# Load and prepare data properly
df = pd.read_csv(DATA_PATH)
df[TIME_COL] = pd.to_datetime(df[TIME_COL], format='mixed', errors='coerce')
df = df.sort_values(TIME_COL).reset_index(drop=True)

print(f"Data shape: {df.shape}")
print(f"Date range: {df[TIME_COL].min()} to {df[TIME_COL].max()}")
print()

# Create proper time series features (using LAGGED data only)
# We'll use previous day's OHLC to predict current day's close
# This is a proper prediction task (not data leakage)

print("Creating time series features (lagged)...")
print()

# Create lagged features (use previous day's data)
lag = 1  # Use previous day
df['prev_open'] = df['open'].shift(lag)
df['prev_high'] = df['high'].shift(lag)
df['prev_low'] = df['low'].shift(lag)
df['prev_close'] = df['close'].shift(lag)
df['prev_volume'] = df['volume'].shift(lag)

# Create additional lagged features
for lag_days in [2, 3, 5, 10]:
    df[f'close_lag_{lag_days}'] = df['close'].shift(lag_days)
    df[f'volume_lag_{lag_days}'] = df['volume'].shift(lag_days)

# Create returns (percentage changes)
df['return_1d'] = df['close'].pct_change(1)
df['return_1d_lag'] = df['return_1d'].shift(lag)

# Create moving averages (using past data only)
for window in [5, 10, 20]:
    df[f'sma_{window}'] = df['close'].shift(1).rolling(window=window).mean()
    df[f'volume_sma_{window}'] = df['volume'].shift(1).rolling(window=window).mean()

# Volatility (using past data)
df['volatility_5d'] = df['close'].shift(1).rolling(window=5).std()
df['volatility_20d'] = df['close'].shift(1).rolling(window=20).std()

# Target: Current day's close (we predict this using past data)
target = df['close'].copy()

# Features: Only lagged/past data
feature_cols = [
    'prev_open', 'prev_high', 'prev_low', 'prev_close', 'prev_volume',
    'close_lag_2', 'close_lag_3', 'close_lag_5', 'close_lag_10',
    'volume_lag_2', 'volume_lag_3', 'volume_lag_5', 'volume_lag_10',
    'return_1d_lag',
    'sma_5', 'sma_10', 'sma_20',
    'volume_sma_5', 'volume_sma_10', 'volume_sma_20',
    'volatility_5d', 'volatility_20d'
]

# Remove original OHLC columns to avoid data leakage
df_features = df[feature_cols + [TIME_COL]].copy()

# Remove rows with NaN (from lagging)
df_features = df_features.dropna()
target = target.loc[df_features.index]

print(f"Features used: {len(feature_cols)}")
print(f"  - All features are LAGGED (from past days)")
print(f"  - No same-day data leakage")
print(f"  - Proper time series prediction")
print()

# Save prepared data
prepared_data_path = Path(OUTPUT_DIR) / "bitcoin_prepared_fixed.csv"
prepared_data_path.parent.mkdir(parents=True, exist_ok=True)
df_features_with_target = df_features.copy()
df_features_with_target['close'] = target
df_features_with_target.to_csv(prepared_data_path, index=False)
print(f"Saved prepared data to: {prepared_data_path}")
print()

# Run model selection
print("="*80)
print("RUNNING MODEL SELECTION")
print("="*80)
print()

results, best_model, X_test, y_test, X_train_val, y_train_val = run_model_selection(
    data_path=str(prepared_data_path),
    target_col='close',
    task_type=TASK_TYPE,
    time_col=TIME_COL,
    is_timeseries=IS_TIMESERIES,
    output_dir=OUTPUT_DIR
)

# Comprehensive overfitting check
print("\n" + "="*80)
print("COMPREHENSIVE OVERFITTING AND DATA LEAKAGE CHECK")
print("="*80)
print()

# Get scores from results
train_val_score = results['overfitting_analysis']['after_tuning']['train_val_score']
test_score = results['overfitting_analysis']['after_tuning']['test_score']

# Perform comprehensive check
overfitting_report = comprehensive_overfitting_check(
    X_train_val, y_train_val,
    train_val_score, test_score, test_score,
    time_col=TIME_COL, target_col='close',
    threshold=0.05
)

# Print report
print("Data Leakage Check:")
if overfitting_report['data_leakage']['has_leakage']:
    print("  ❌ FAIL - Data leakage detected!")
    for issue in overfitting_report['data_leakage']['issues']:
        print(f"    - {issue}")
else:
    print("  ✅ PASS - No data leakage detected")
if overfitting_report['data_leakage']['warnings']:
    for warning in overfitting_report['data_leakage']['warnings']:
        print(f"    ⚠ {warning}")

print("\nOverfitting Check:")
if overfitting_report['overfitting']['is_overfitting']:
    print(f"  ❌ FAIL - Overfitting detected (Severity: {overfitting_report['overfitting']['severity']})")
    for issue in overfitting_report['overfitting']['issues']:
        print(f"    - {issue}")
else:
    print("  ✅ PASS - No significant overfitting")
print(f"  Train-Val Gap: {overfitting_report['overfitting']['train_val_gap']:.2%}")
if overfitting_report['overfitting']['train_test_gap']:
    print(f"  Train-Test Gap: {overfitting_report['overfitting']['train_test_gap']:.2%}")

print("\nFeature Quality Check:")
if overfitting_report['feature_quality']['duplicate_features']:
    print(f"  ⚠ Found {len(overfitting_report['feature_quality']['duplicate_features'])} duplicate feature pairs")
if overfitting_report['feature_quality']['highly_correlated_pairs']:
    print(f"  ⚠ Found {len(overfitting_report['feature_quality']['highly_correlated_pairs'])} highly correlated pairs")

print(f"\nOverall Status: {overfitting_report['overall_status']}")

if overfitting_report['recommendations']:
    print("\nRecommendations:")
    for rec in overfitting_report['recommendations']:
        print(f"  - {rec}")

# Save overfitting report
overfitting_path = Path(OUTPUT_DIR) / "overfitting_report.json"
with open(overfitting_path, 'w') as f:
    json.dump(overfitting_report, f, indent=2, default=str)
print(f"\nOverfitting report saved to: {overfitting_path}")

# Only proceed with prediction if model passes checks
if overfitting_report['overall_status'].startswith('FAIL'):
    print("\n" + "="*80)
    print("WARNING: Model failed overfitting/data leakage checks!")
    print("Prediction will not be made with this model.")
    print("="*80)
    sys.exit(1)

# Make prediction on most recent data
print("\n" + "="*80)
print("MAKING PREDICTION ON MOST RECENT DATA")
print("="*80)

# Get most recent row with all features
most_recent = df_features.iloc[-1:].copy()
current_price = target.iloc[-1]
current_date = df_features[TIME_COL].iloc[-1]

# Make prediction
predicted_price = best_model.predict(most_recent.drop(columns=[TIME_COL]))[0]
predicted_change = predicted_price - current_price
predicted_change_pct = (predicted_change / current_price) * 100

# Calculate confidence intervals
test_pred = best_model.predict(X_test)
test_errors = y_test - test_pred
error_std = test_errors.std()

ci_68_low = predicted_price - error_std
ci_68_high = predicted_price + error_std
ci_95_low = predicted_price - 2 * error_std
ci_95_high = predicted_price + 2 * error_std

print(f"\nCurrent Date: {current_date}")
print(f"Current Price: USD {current_price:,.2f}")
print(f"\nPredicted Price: USD {predicted_price:,.2f}")
print(f"Predicted Change: USD {predicted_change:,.2f} ({predicted_change_pct:+.2f}%)")
print(f"Direction: {'UP' if predicted_change > 0 else 'DOWN'}")
print(f"\nConfidence Intervals:")
print(f"  68% CI: USD {ci_68_low:,.2f} - USD {ci_68_high:,.2f}")
print(f"  95% CI: USD {ci_95_low:,.2f} - USD {ci_95_high:,.2f}")

# Save prediction
prediction_result = {
    'current_date': str(current_date),
    'current_price': float(current_price),
    'predicted_price': float(predicted_price),
    'predicted_change': float(predicted_change),
    'predicted_change_pct': float(predicted_change_pct),
    'direction': 'UP' if predicted_change > 0 else 'DOWN',
    'confidence_intervals': {
        '68_low': float(ci_68_low),
        '68_high': float(ci_68_high),
        '95_low': float(ci_95_low),
        '95_high': float(ci_95_high)
    },
    'error_std': float(error_std),
    'model': results['best_model'],
    'overfitting_status': overfitting_report['overall_status'],
    'prediction_date': datetime.now().isoformat()
}

prediction_path = Path(OUTPUT_DIR) / 'latest_prediction.json'
with open(prediction_path, 'w') as f:
    json.dump(prediction_result, f, indent=2)

print(f"\nPrediction saved to: {prediction_path}")
print()



