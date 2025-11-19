#!/usr/bin/env python3
"""
Run ML Model Selection Pipeline on Bitcoin Data and Make Predictions
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from model_selection_pipeline import run_model_selection
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

print("="*80)
print("BITCOIN PRICE PREDICTION USING ML PIPELINE")
print("="*80)
print()

# Configuration
DATA_PATH = "data/btc-ohlc.csv"  # Adjust path as needed
TARGET_COL = "close"  # Lowercase in the data
TASK_TYPE = "REGRESSION"
TIME_COL = "d"  # The actual column name in the data
IS_TIMESERIES = True
OUTPUT_DIR = "ml_pipeline/bitcoin_results"

# Check if data file exists, try alternative paths
possible_paths = [
    "data/btc-ohlc.csv",
    "datasets/Bitcoin/bitcoin data csvs/btc-ohlc.csv",
    "analysis/bitcoin/results/comprehensive_enhanced/bitcoin_with_features.csv",
    "analysis/bitcoin/results/comprehensive/bitcoin_with_features.csv",
]

data_path = None
for path in possible_paths:
    if Path(path).exists():
        data_path = path
        break

if not data_path:
    print("ERROR: Bitcoin data file not found. Checked:")
    for path in possible_paths:
        print(f"  - {path}")
    sys.exit(1)

print(f"Using data file: {data_path}")
print()

# Run model selection
results, best_model, X_test, y_test, X_train_val, y_train_val = run_model_selection(
    data_path=data_path,
    target_col=TARGET_COL,
    task_type=TASK_TYPE,
    time_col=TIME_COL,
    is_timeseries=IS_TIMESERIES,
    output_dir=OUTPUT_DIR
)

# Make prediction on most recent data
print("\n" + "="*80)
print("MAKING PREDICTION ON MOST RECENT DATA")
print("="*80)

# Load full dataset to get most recent row
df = pd.read_csv(data_path)
df[TIME_COL] = pd.to_datetime(df[TIME_COL], format='mixed', errors='coerce')
df = df.sort_values(TIME_COL)

# Get most recent row (excluding target for prediction)
most_recent = df.iloc[-1:].copy()
X_recent = most_recent.drop(columns=[TARGET_COL, TIME_COL] if TIME_COL in most_recent.columns else [TARGET_COL])
current_price = most_recent[TARGET_COL].iloc[0]
current_date = most_recent[TIME_COL].iloc[0] if TIME_COL in most_recent.columns else None

# Make prediction
predicted_price = best_model.predict(X_recent)[0]
predicted_change = predicted_price - current_price
predicted_change_pct = (predicted_change / current_price) * 100

# Calculate confidence interval based on test set errors
test_pred = best_model.predict(X_test)
test_errors = y_test - test_pred
error_std = test_errors.std()

ci_68_low = predicted_price - error_std
ci_68_high = predicted_price + error_std
ci_95_low = predicted_price - 2 * error_std
ci_95_high = predicted_price + 2 * error_std

print(f"\nCurrent Date: {current_date}")
print(f"Current Price: ${current_price:,.2f}")
print(f"\nPredicted Price: ${predicted_price:,.2f}")
print(f"Predicted Change: ${predicted_change:,.2f} ({predicted_change_pct:+.2f}%)")
print(f"Direction: {'UP' if predicted_change > 0 else 'DOWN'}")
print(f"\nConfidence Intervals:")
print(f"  68% CI: ${ci_68_low:,.2f} - ${ci_68_high:,.2f}")
print(f"  95% CI: ${ci_95_low:,.2f} - ${ci_95_high:,.2f}")

# Save prediction
prediction_result = {
    'current_date': str(current_date) if current_date else None,
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
    'prediction_date': datetime.now().isoformat()
}

output_path = Path(OUTPUT_DIR)
with open(output_path / 'latest_prediction.json', 'w') as f:
    json.dump(prediction_result, f, indent=2)

print(f"\nPrediction saved to: {output_path / 'latest_prediction.json'}")
print()

