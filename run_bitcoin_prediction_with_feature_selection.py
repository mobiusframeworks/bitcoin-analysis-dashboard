#!/usr/bin/env python3
"""
Run ML Model Selection Pipeline with Feature Selection

This version:
1. Uses proper time series features (lagged)
2. Applies comprehensive feature selection
3. Separates FRED vs Technical features
4. Removes redundant/correlated features
5. Comprehensive overfitting checks
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from model_selection_pipeline import run_model_selection
from overfitting_detector import comprehensive_overfitting_check
from feature_selector import (
    comprehensive_feature_selection, 
    identify_feature_categories,
    apply_feature_selection
)
import pandas as pd
import numpy as np
import json
from datetime import datetime

print("="*80)
print("BITCOIN PRICE PREDICTION - WITH FEATURE SELECTION")
print("="*80)
print()

# Configuration
DATA_PATH = "datasets/Bitcoin/bitcoin data csvs/btc-ohlc.csv"
TARGET_COL = "close"
TASK_TYPE = "REGRESSION"
TIME_COL = "d"
IS_TIMESERIES = True
OUTPUT_DIR = "ml_pipeline/bitcoin_results_feature_selected"

# Load data
df = pd.read_csv(DATA_PATH)
df[TIME_COL] = pd.to_datetime(df[TIME_COL], format='mixed', errors='coerce')
df = df.sort_values(TIME_COL).reset_index(drop=True)

print(f"Data shape: {df.shape}")
print(f"Date range: {df[TIME_COL].min()} to {df[TIME_COL].max()}")
print()

# Create time series features (all lagged)
print("Creating time series features (all lagged)...")
print()

lag = 1

# Price features
df['prev_close'] = df['close'].shift(lag)
df['prev_open'] = df['open'].shift(lag)
df['prev_high'] = df['high'].shift(lag)
df['prev_low'] = df['low'].shift(lag)
df['prev_volume'] = df['volume'].shift(lag)

# Returns
df['return_1d'] = df['close'].pct_change(1).shift(lag)
df['return_5d'] = df['close'].pct_change(5).shift(lag)
df['return_20d'] = df['close'].pct_change(20).shift(lag)

# Volatility
df['volatility_5d'] = df['close'].shift(1).rolling(window=5).std()
df['volatility_20d'] = df['close'].shift(1).rolling(window=20).std()

# Moving averages
df['sma_20'] = df['close'].shift(1).rolling(window=20).mean()
df['sma_50'] = df['close'].shift(1).rolling(window=50).mean()
df['sma_200'] = df['close'].shift(1).rolling(window=200).mean()

# Price ratios
df['price_sma20_ratio'] = df['prev_close'] / df['sma_20']
df['price_sma50_ratio'] = df['prev_close'] / df['sma_50']
df['price_sma200_ratio'] = df['prev_close'] / df['sma_200']

# Volume features
df['volume_sma_20'] = df['volume'].shift(1).rolling(window=20).mean()
df['volume_ratio'] = df['prev_volume'] / df['volume_sma_20']

# Additional lagged features
for lag_days in [2, 3, 5, 10]:
    df[f'close_lag_{lag_days}'] = df['close'].shift(lag_days)

# Target
target = df['close'].copy()

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

# Try to load FRED data if available
fred_paths = [
    "analysis/bitcoin/results/fred_analysis/fred_bitcoin_combined.csv",
    "trading algo/bitcoin/results/fred_analysis/fred_bitcoin_combined.csv",
]

fred_data = None
for path in fred_paths:
    if Path(path).exists():
        try:
            fred_df = pd.read_csv(path)
            fred_df['Date'] = pd.to_datetime(fred_df['Date'], format='mixed', errors='coerce')
            fred_df = fred_df.set_index('Date')
            
            # Merge FRED data
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
            
            print(f"  ✓ Loaded FRED data: {len([f for f in initial_features if any(x in f.upper() for x in ['M2', 'NET', 'FED', 'CPI', 'GFDEBT', 'DGS', 'WALCL', 'UNRATE'])])} features")
            fred_data = True
            break
        except Exception as e:
            print(f"  ⚠ Could not load FRED data: {e}")
            continue

if not fred_data:
    print("  ⚠ FRED data not available - using technical features only")

# Prepare data
df_features = df[initial_features + [TIME_COL]].copy()
df_features = df_features.dropna()
target = target.loc[df_features.index]

print(f"\nInitial features: {len(initial_features)}")
print()

# Categorize features
categories = identify_feature_categories(initial_features)
print("Feature categories:")
for cat, features in categories.items():
    if features:
        print(f"  {cat}: {len(features)} features")
        print(f"    {', '.join(features[:5])}{'...' if len(features) > 5 else ''}")
print()

# Apply feature selection
print("="*80)
print("APPLYING FEATURE SELECTION")
print("="*80)
print()

X = df_features[initial_features].values
y = target.values

selected_features, selection_report = comprehensive_feature_selection(
    X, y,
    feature_names=initial_features,
    correlation_threshold=0.95,  # Remove features with >95% correlation
    cross_category_threshold=0.90,  # Remove FRED/Technical conflicts >90%
    use_mutual_info=True,
    top_k=15  # Keep top 15 features
)

print("Feature Selection Report:")
for step in selection_report['steps']:
    print(f"\n{step['step']}:")
    print(f"  Removed: {step['removed']} features")
    print(f"  Remaining: {step['remaining']} features")
    if 'removed_features' in step and step['removed_features']:
        print(f"  Removed: {', '.join(step['removed_features'][:5])}")
    if 'conflicts' in step:
        print(f"  FRED/Technical conflicts resolved: {step['conflicts']}")
        if step.get('conflict_details'):
            for conflict in step['conflict_details']:
                print(f"    - {conflict['fred_feature']} vs {conflict['technical_feature']} "
                      f"(corr={conflict['correlation']:.3f}) -> Removed {conflict['removed']}")

print(f"\nFinal selected features: {len(selected_features)}")
print(f"  Reduction: {len(initial_features)} -> {len(selected_features)} "
      f"({(1 - len(selected_features)/len(initial_features))*100:.1f}% reduction)")

final_categories = identify_feature_categories(selected_features)
print("\nFinal feature categories:")
for cat, features in final_categories.items():
    if features:
        print(f"  {cat}: {len(features)} - {', '.join(features)}")

# Save prepared data with selected features
prepared_data_path = Path(OUTPUT_DIR) / "bitcoin_prepared_feature_selected.csv"
prepared_data_path.parent.mkdir(parents=True, exist_ok=True)
df_features_selected = df_features[selected_features + [TIME_COL]].copy()
df_features_selected['close'] = target
df_features_selected.to_csv(prepared_data_path, index=False)
print(f"\nSaved prepared data to: {prepared_data_path}")

# Save feature selection report
selection_report_path = Path(OUTPUT_DIR) / "feature_selection_report.json"
with open(selection_report_path, 'w') as f:
    json.dump(selection_report, f, indent=2, default=str)
print(f"Feature selection report saved to: {selection_report_path}")
print()

# Run model selection
print("="*80)
print("RUNNING MODEL SELECTION WITH SELECTED FEATURES")
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

overfitting_report = comprehensive_overfitting_check(
    X_train_val, y_train_val,
    train_val_score, test_score, test_score,
    time_col=TIME_COL, target_col='close',
    threshold=0.10
)

# Print report
print("Data Leakage Check:")
if overfitting_report['data_leakage']['has_leakage']:
    print("  ❌ FAIL - Data leakage detected!")
    for issue in overfitting_report['data_leakage']['issues'][:5]:
        print(f"    - {issue}")
else:
    print("  ✅ PASS - No data leakage detected")

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

print(f"\nOverall Status: {overfitting_report['overall_status']}")

# Save overfitting report
overfitting_path = Path(OUTPUT_DIR) / "overfitting_report.json"
with open(overfitting_path, 'w') as f:
    json.dump(overfitting_report, f, indent=2, default=str)

# Only proceed if model passes checks
if overfitting_report['overall_status'].startswith('FAIL'):
    print("\n" + "="*80)
    print("WARNING: Model failed overfitting/data leakage checks!")
    print("This model should NOT be used for predictions.")
    print("="*80)
    sys.exit(1)

# Make prediction
print("\n" + "="*80)
print("MAKING PREDICTION ON MOST RECENT DATA")
print("="*80)

most_recent = df_features_selected.iloc[-1:].copy()
current_price = target.iloc[-1]
current_date = df_features_selected[TIME_COL].iloc[-1]

predicted_price = best_model.predict(most_recent.drop(columns=[TIME_COL, 'close']))[0]
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
    'selected_features': selected_features,
    'prediction_date': datetime.now().isoformat()
}

prediction_path = Path(OUTPUT_DIR) / 'latest_prediction.json'
with open(prediction_path, 'w') as f:
    json.dump(prediction_result, f, indent=2)

print(f"\nPrediction saved to: {prediction_path}")
print()



