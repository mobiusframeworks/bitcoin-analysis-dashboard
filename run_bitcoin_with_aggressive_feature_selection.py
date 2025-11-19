#!/usr/bin/env python3
"""
Run Bitcoin Prediction with Aggressive Feature Selection

This version uses multiple strategies to aggressively remove redundant features:
1. VIF-based removal (multicollinearity)
2. Correlation-based removal (more aggressive)
3. Linear combination removal
4. Recursive Feature Elimination
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from model_selection_pipeline import run_model_selection
from overfitting_detector import comprehensive_overfitting_check
from aggressive_feature_selection import (
    aggressive_feature_selection,
    analyze_feature_redundancy,
    calculate_vif
)
from feature_selector import identify_feature_categories
import pandas as pd
import numpy as np
import json
from datetime import datetime

print("="*80)
print("BITCOIN PREDICTION - AGGRESSIVE FEATURE SELECTION")
print("="*80)
print()

# Configuration
DATA_PATH = "datasets/Bitcoin/bitcoin data csvs/btc-ohlc.csv"
TARGET_LAG_DAYS = 20
TASK_TYPE = "REGRESSION"
TIME_COL = "d"
IS_TIMESERIES = True
OUTPUT_DIR = "ml_pipeline/bitcoin_results_aggressive_fs"

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
        except Exception as e:
            continue

# Prepare data
df_features = df[initial_features + [TIME_COL]].copy()
valid_mask = ~target.isna()
df_features = df_features[valid_mask].copy()
target = target[valid_mask].copy()
df_features = df_features.dropna()
target = target.loc[df_features.index]

print(f"\nInitial features: {len(initial_features)}")
print(f"Target: {TARGET_LAG_DAYS}-day forward returns")
print()

# Analyze redundancy BEFORE selection
print("="*80)
print("ANALYZING FEATURE REDUNDANCY")
print("="*80)
print()

X = df_features[initial_features].values
y = target.values

redundancy_analysis = analyze_feature_redundancy(X, y, initial_features)

print("Correlation Analysis:")
print(f"  High correlation pairs (>0.85): {redundancy_analysis['correlation_analysis']['num_high_corr_pairs']}")
if redundancy_analysis['correlation_analysis']['high_correlation_pairs']:
    print("  Top redundant pairs:")
    for pair in redundancy_analysis['correlation_analysis']['high_correlation_pairs'][:10]:
        print(f"    - {pair['feature1']} vs {pair['feature2']}: {pair['correlation']:.3f}")

print("\nVIF Analysis:")
vif_df = calculate_vif(X, initial_features)
high_vif = vif_df[vif_df['vif'] > 5.0]
print(f"  Features with VIF > 5.0: {len(high_vif)}")
if len(high_vif) > 0:
    print("  High VIF features:")
    for _, row in high_vif.head(10).iterrows():
        print(f"    - {row['feature']}: VIF = {row['vif']:.2f}")

print("\nRecommendations:")
for rec in redundancy_analysis['recommendations']:
    print(f"  - {rec}")
print()

# Apply aggressive feature selection
print("="*80)
print("APPLYING AGGRESSIVE FEATURE SELECTION")
print("="*80)
print()

selected_features, selection_report = aggressive_feature_selection(
    X, y,
    feature_names=initial_features,
    correlation_threshold=0.85,  # Remove features with >85% correlation
    vif_threshold=5.0,  # Remove features with VIF > 5
    use_rfe=True,
    n_features_final=8  # Final number of features
)

print("\nFeature Selection Summary:")
for step in selection_report['steps']:
    print(f"\n{step['step']}:")
    print(f"  Removed: {step['removed']} features")
    print(f"  Remaining: {step['remaining']} features")
    if 'removed_features' in step and step['removed_features']:
        print(f"  Removed: {', '.join(step['removed_features'][:5])}")

print(f"\nFinal selected features: {len(selected_features)}")
print(f"  Reduction: {len(initial_features)} -> {len(selected_features)} "
      f"({(1 - len(selected_features)/len(initial_features))*100:.1f}% reduction)")

final_categories = identify_feature_categories(selected_features)
print("\nFinal feature categories:")
for cat, features in final_categories.items():
    if features:
        print(f"  {cat}: {len(features)} - {', '.join(features)}")

# Check VIF of final features
print("\nVIF of Final Features:")
X_final = df_features[selected_features].values
vif_final = calculate_vif(X_final, selected_features)
print(vif_final.to_string(index=False))
print()

# Save prepared data
prepared_data_path = Path(OUTPUT_DIR) / "bitcoin_prepared_aggressive_fs.csv"
prepared_data_path.parent.mkdir(parents=True, exist_ok=True)
df_features_selected = df_features[selected_features + [TIME_COL]].copy()
df_features_selected['target_return'] = target
df_features_selected['close'] = df.loc[df_features.index, 'close'].values
df_features_selected.to_csv(prepared_data_path, index=False)
print(f"Saved prepared data to: {prepared_data_path}")

# Save feature selection report
selection_report_path = Path(OUTPUT_DIR) / "aggressive_feature_selection_report.json"
with open(selection_report_path, 'w') as f:
    json.dump(selection_report, f, indent=2, default=str)
print(f"Feature selection report saved to: {selection_report_path}")
print()

# Run model selection
print("="*80)
print("RUNNING MODEL SELECTION WITH AGGRESSIVELY SELECTED FEATURES")
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
print("COMPREHENSIVE OVERFITTING AND DATA LEAKAGE CHECK")
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
print(f"  Train Score (R²): {train_val_score:.4f}")
print(f"  Test Score (R²): {test_score:.4f}")
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
current_price = df_features_selected['close'].iloc[-1]
current_date = df_features_selected[TIME_COL].iloc[-1]

predicted_return = best_model.predict(most_recent.drop(columns=[TIME_COL, 'target_return', 'close']))[0]
predicted_price = current_price * (1 + predicted_return)
predicted_change = predicted_price - current_price
predicted_change_pct = predicted_return * 100

test_pred = best_model.predict(X_test)
test_errors = y_test - test_pred
error_std = test_errors.std()
price_error_std = current_price * error_std

ci_68_low = predicted_price - price_error_std
ci_68_high = predicted_price + price_error_std
ci_95_low = predicted_price - 2 * price_error_std
ci_95_high = predicted_price + 2 * price_error_std

print(f"\nCurrent Date: {current_date}")
print(f"Current Price: USD {current_price:,.2f}")
print(f"\nPredicted {TARGET_LAG_DAYS}-Day Return: {predicted_return*100:+.2f}%")
print(f"Predicted Price: USD {predicted_price:,.2f}")
print(f"Predicted Change: USD {predicted_change:,.2f} ({predicted_change_pct:+.2f}%)")
print(f"Direction: {'UP' if predicted_return > 0 else 'DOWN'}")
print(f"\nConfidence Intervals:")
print(f"  68% CI: USD {ci_68_low:,.2f} - USD {ci_68_high:,.2f}")
print(f"  95% CI: USD {ci_95_low:,.2f} - USD {ci_95_high:,.2f}")

# Save prediction
prediction_result = {
    'current_date': str(current_date),
    'current_price': float(current_price),
    'predicted_return_pct': float(predicted_return * 100),
    'predicted_price': float(predicted_price),
    'predicted_change': float(predicted_change),
    'predicted_change_pct': float(predicted_change_pct),
    'direction': 'UP' if predicted_return > 0 else 'DOWN',
    'prediction_horizon_days': TARGET_LAG_DAYS,
    'confidence_intervals': {
        '68_low': float(ci_68_low),
        '68_high': float(ci_68_high),
        '95_low': float(ci_95_low),
        '95_high': float(ci_95_high)
    },
    'return_error_std': float(error_std),
    'model': results['best_model'],
    'overfitting_status': overfitting_report['overall_status'],
    'test_r2': float(test_score),
    'selected_features': selected_features,
    'num_features': len(selected_features),
    'prediction_date': datetime.now().isoformat()
}

prediction_path = Path(OUTPUT_DIR) / 'latest_prediction.json'
with open(prediction_path, 'w') as f:
    json.dump(prediction_result, f, indent=2)

print(f"\nPrediction saved to: {prediction_path}")
print()



