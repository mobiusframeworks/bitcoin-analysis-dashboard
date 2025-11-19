#!/usr/bin/env python3
"""
Run ML Model Selection Pipeline on Bitcoin Data - CLEAN VERSION

This version:
1. Uses only past data (proper time series)
2. Removes redundant/highly correlated features
3. Uses feature selection to reduce overfitting
4. Comprehensive overfitting checks
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
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA

print("="*80)
print("BITCOIN PRICE PREDICTION - CLEAN VERSION")
print("="*80)
print()

# Configuration
DATA_PATH = "datasets/Bitcoin/bitcoin data csvs/btc-ohlc.csv"
TARGET_COL = "close"
TASK_TYPE = "REGRESSION"
TIME_COL = "d"
IS_TIMESERIES = True
OUTPUT_DIR = "ml_pipeline/bitcoin_results_clean"

# Load data
df = pd.read_csv(DATA_PATH)
df[TIME_COL] = pd.to_datetime(df[TIME_COL], format='mixed', errors='coerce')
df = df.sort_values(TIME_COL).reset_index(drop=True)

print(f"Data shape: {df.shape}")
print(f"Date range: {df[TIME_COL].min()} to {df[TIME_COL].max()}")
print()

# Create CLEAN time series features (minimal, non-redundant)
print("Creating clean time series features...")
print()

# Use only essential lagged features
lag = 1
df['prev_close'] = df['close'].shift(lag)  # Most important: previous close
df['prev_volume'] = df['volume'].shift(lag)

# Returns (more informative than raw prices)
df['return_1d'] = df['close'].pct_change(1).shift(lag)
df['return_5d'] = df['close'].pct_change(5).shift(lag)
df['return_20d'] = df['close'].pct_change(20).shift(lag)

# Volatility (using past data only)
df['volatility_5d'] = df['close'].shift(1).rolling(window=5).std()
df['volatility_20d'] = df['close'].shift(1).rolling(window=20).std()

# Moving averages (using past data)
df['sma_20'] = df['close'].shift(1).rolling(window=20).mean()
df['sma_50'] = df['close'].shift(1).rolling(window=50).mean()

# Price ratios (more stable than raw prices)
df['price_sma20_ratio'] = df['prev_close'] / df['sma_20']
df['price_sma50_ratio'] = df['prev_close'] / df['sma_50']

# Volume features
df['volume_sma_20'] = df['volume'].shift(1).rolling(window=20).mean()
df['volume_ratio'] = df['prev_volume'] / df['volume_sma_20']

# Target: Current day's close
target = df['close'].copy()

# Select only non-redundant features
feature_cols = [
    'prev_close',           # Previous close (most important)
    'return_1d',            # 1-day return
    'return_5d',            # 5-day return  
    'return_20d',           # 20-day return
    'volatility_5d',        # Short-term volatility
    'volatility_20d',       # Long-term volatility
    'price_sma20_ratio',    # Price relative to SMA20
    'price_sma50_ratio',    # Price relative to SMA50
    'volume_ratio',         # Volume relative to average
]

df_features = df[feature_cols + [TIME_COL]].copy()

# Remove rows with NaN
df_features = df_features.dropna()
target = target.loc[df_features.index]

print(f"Features used: {len(feature_cols)}")
print("  - All features are LAGGED (from past days)")
print("  - Minimal redundancy (no duplicate OHLC)")
print("  - Focus on returns, ratios, and volatility")
print(f"  - Final data shape: {df_features.shape}")
print()

# Check for high correlations
print("Checking feature correlations...")
corr_matrix = df_features[feature_cols].corr()
high_corr_pairs = []
for i, col1 in enumerate(corr_matrix.columns):
    for col2 in corr_matrix.columns[i+1:]:
        corr = abs(corr_matrix.loc[col1, col2])
        if corr > 0.95:
            high_corr_pairs.append((col1, col2, corr))

if high_corr_pairs:
    print(f"  ⚠ Found {len(high_corr_pairs)} highly correlated pairs (>0.95)")
    for col1, col2, corr in high_corr_pairs[:5]:  # Show first 5
        print(f"    - {col1} vs {col2}: {corr:.4f}")
else:
    print("  ✅ No highly correlated pairs found")
print()

# Save prepared data
prepared_data_path = Path(OUTPUT_DIR) / "bitcoin_prepared_clean.csv"
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

train_val_score = results['overfitting_analysis']['after_tuning']['train_val_score']
test_score = results['overfitting_analysis']['after_tuning']['test_score']

# Perform comprehensive check
overfitting_report = comprehensive_overfitting_check(
    X_train_val, y_train_val,
    train_val_score, test_score, test_score,
    time_col=TIME_COL, target_col='close',
    threshold=0.10  # More lenient threshold for time series
)

# Print report
print("Data Leakage Check:")
if overfitting_report['data_leakage']['has_leakage']:
    print("  ❌ FAIL - Data leakage detected!")
    for issue in overfitting_report['data_leakage']['issues'][:5]:  # Show first 5
        print(f"    - {issue}")
    if len(overfitting_report['data_leakage']['issues']) > 5:
        print(f"    ... and {len(overfitting_report['data_leakage']['issues']) - 5} more")
else:
    print("  ✅ PASS - No data leakage detected")
if overfitting_report['data_leakage']['warnings']:
    for warning in overfitting_report['data_leakage']['warnings'][:3]:
        print(f"    ⚠ {warning}")

print("\nOverfitting Check:")
if overfitting_report['overfitting']['is_overfitting']:
    print(f"  ❌ FAIL - Overfitting detected (Severity: {overfitting_report['overfitting']['severity']})")
    for issue in overfitting_report['overfitting']['issues']:
        print(f"    - {issue}")
else:
    print("  ✅ PASS - No significant overfitting")
print(f"  Train Score: {train_val_score:.4f}")
print(f"  Test Score: {test_score:.4f}")
if overfitting_report['overfitting']['train_val_gap']:
    print(f"  Train-Test Gap: {overfitting_report['overfitting']['train_val_gap']:.2%}")

print("\nFeature Quality Check:")
if overfitting_report['feature_quality']['duplicate_features']:
    print(f"  ⚠ Found {len(overfitting_report['feature_quality']['duplicate_features'])} duplicate features")
if overfitting_report['feature_quality']['highly_correlated_pairs']:
    print(f"  ⚠ Found {len(overfitting_report['feature_quality']['highly_correlated_pairs'])} highly correlated pairs")
else:
    print("  ✅ No duplicate or highly correlated features")

print(f"\nOverall Status: {overfitting_report['overall_status']}")

if overfitting_report['recommendations']:
    print("\nRecommendations:")
    for rec in overfitting_report['recommendations'][:5]:
        print(f"  - {rec}")

# Save overfitting report
overfitting_path = Path(OUTPUT_DIR) / "overfitting_report.json"
with open(overfitting_path, 'w') as f:
    json.dump(overfitting_report, f, indent=2, default=str)
print(f"\nOverfitting report saved to: {overfitting_path}")

# Only proceed if model passes checks
if overfitting_report['overall_status'].startswith('FAIL'):
    print("\n" + "="*80)
    print("WARNING: Model failed overfitting/data leakage checks!")
    print("This model should NOT be used for predictions.")
    print("="*80)
    
    # Update results to mark as failed
    results['overfitting_status'] = 'FAILED'
    results['overfitting_report'] = overfitting_report
    with open(Path(OUTPUT_DIR) / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    sys.exit(1)

# Make prediction
print("\n" + "="*80)
print("MAKING PREDICTION ON MOST RECENT DATA")
print("="*80)

most_recent = df_features.iloc[-1:].copy()
current_price = target.iloc[-1]
current_date = df_features[TIME_COL].iloc[-1]

predicted_price = best_model.predict(most_recent.drop(columns=[TIME_COL]))[0]
predicted_change = predicted_price - current_price
predicted_change_pct = (predicted_change / current_price) * 100

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
    'test_r2': float(test_score),
    'prediction_date': datetime.now().isoformat()
}

prediction_path = Path(OUTPUT_DIR) / 'latest_prediction.json'
with open(prediction_path, 'w') as f:
    json.dump(prediction_result, f, indent=2)

print(f"\nPrediction saved to: {prediction_path}")
print()



