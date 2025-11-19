#!/usr/bin/env python3
"""
Prepare Bitcoin Data and Run ML Pipeline

This script:
1. Loads Bitcoin data with features
2. Creates target variable (20-day forward returns)
3. Runs the ML model selection pipeline
4. Makes predictions on current data
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from model_selection_pipeline import run_model_selection
import json

print("="*80)
print("BITCOIN ML MODEL SELECTION - DATA PREPARATION & PIPELINE")
print("="*80)
print()

# Find Bitcoin data file
data_paths = [
    Path("analysis/bitcoin/results/comprehensive_enhanced/bitcoin_with_features.csv"),
    Path("analysis/bitcoin/results/comprehensive/bitcoin_with_features.csv"),
    Path("trading algo/bitcoin/results/comprehensive_enhanced/bitcoin_with_features.csv"),
    Path("data/btc-ohlc.csv"),
]

data_path = None
for path in data_paths:
    if path.exists():
        data_path = path
        break

if not data_path:
    print("ERROR: Bitcoin data file not found!")
    sys.exit(1)

print(f"Loading data from: {data_path}")
print()

# Load data
df = pd.read_csv(data_path)

# Handle date column
date_col = None
for col in ['Date', 'date', 'DATE', 'timestamp', 'Timestamp']:
    if col in df.columns:
        date_col = col
        break

if date_col:
    df[date_col] = pd.to_datetime(df[date_col], format='mixed', errors='coerce')
    df = df.sort_values(date_col)
else:
    print("Warning: No date column found, assuming data is already sorted")

# Get price column
price_col = None
for col in ['Close', 'close', 'BTC_Price', 'price']:
    if col in df.columns:
        price_col = col
        break

if not price_col:
    # Use first numeric column
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        price_col = numeric_cols[0]
    else:
        print("ERROR: No price column found!")
        sys.exit(1)

print(f"Using price column: {price_col}")
print()

# Create target: 20-day forward returns
target_lag = 20
df['Target_Return'] = df[price_col].pct_change(target_lag).shift(-target_lag)

# Remove rows with missing target
df = df.dropna(subset=['Target_Return'])

print(f"✓ Created target: {target_lag}-day forward returns")
print(f"✓ Data shape: {df.shape}")
print(f"✓ Date range: {df[date_col].min()} to {df[date_col].max()}")
print()

# Save prepared data
prepared_data_path = Path('ml_pipeline/data/bitcoin_prepared.csv')
prepared_data_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(prepared_data_path, index=False)
print(f"✓ Saved prepared data to: {prepared_data_path}")
print()

# Run the pipeline
print("="*80)
print("RUNNING ML MODEL SELECTION PIPELINE")
print("="*80)
print()

pipeline = run_model_selection(
    data_path=str(prepared_data_path),
    target_col='Target_Return',
    time_col=date_col,
    is_timeseries=True,
    task_type='regression'
)

# Save results
results_dir = Path('ml_pipeline/results')
results_dir.mkdir(parents=True, exist_ok=True)

def convert_to_serializable(obj):
    """Convert numpy types to native Python types for JSON."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj

with open(results_dir / 'bitcoin_ml_results.json', 'w') as f:
    serializable_results = convert_to_serializable(pipeline.results)
    json.dump(serializable_results, f, indent=2)

print(f"\n✓ Results saved to: {results_dir / 'bitcoin_ml_results.json'}")

# Make prediction on most recent data
print()
print("="*80)
print("MAKING PREDICTION ON CURRENT DATA")
print("="*80)
print()

# Get most recent row (excluding target)
X_current = df.iloc[[-1]].drop(columns=['Target_Return'])
if date_col:
    current_date = df.iloc[-1][date_col]
    current_price = df.iloc[-1][price_col]
else:
    current_date = "Unknown"
    current_price = df.iloc[-1][price_col]

# Make prediction
predicted_return = pipeline.best_model.predict(X_current)[0]
predicted_price = current_price * (1 + predicted_return)

print(f"Current Date: {current_date}")
print(f"Current Price: ${current_price:,.2f}")
print(f"Predicted Return: {predicted_return*100:.2f}%")
print(f"Predicted Price: ${predicted_price:,.2f}")
print(f"Expected Move: ${predicted_price - current_price:,.2f}")
print()

# Save prediction
prediction_result = {
    'current_date': str(current_date),
    'current_price': float(current_price),
    'predicted_return_pct': float(predicted_return * 100),
    'predicted_price': float(predicted_price),
    'expected_move': float(predicted_price - current_price),
    'target_lag_days': target_lag,
    'model': pipeline.best_model_name,
    'best_params': convert_to_serializable(pipeline.best_params)
}

with open(results_dir / 'bitcoin_current_prediction.json', 'w') as f:
    json.dump(prediction_result, f, indent=2)

print(f"✓ Prediction saved to: {results_dir / 'bitcoin_current_prediction.json'}")
print()

