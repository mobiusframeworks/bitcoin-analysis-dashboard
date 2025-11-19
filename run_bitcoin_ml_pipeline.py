#!/usr/bin/env python3
"""
Run ML Pipeline on Bitcoin Data

This script runs the model selection pipeline on Bitcoin price prediction data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from model_selection_pipeline import run_model_selection
import json

print("="*80)
print("BITCOIN ML MODEL SELECTION PIPELINE")
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

print(f"Using data file: {data_path}")
print()

# We'll predict 20-day forward returns
# First, we need to prepare the data with the target
# For now, let's use a prepared dataset or create target on the fly

# Run the pipeline
pipeline = run_model_selection(
    data_path=str(data_path),
    target_col='Target_Return',  # We'll need to create this
    time_col='Date',
    is_timeseries=True,
    task_type='regression'
)

# Save results
results_dir = Path('ml_pipeline/results')
results_dir.mkdir(parents=True, exist_ok=True)

with open(results_dir / 'bitcoin_ml_results.json', 'w') as f:
    # Convert numpy types to native Python types for JSON
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    serializable_results = convert_to_serializable(pipeline.results)
    json.dump(serializable_results, f, indent=2)

print(f"\n✓ Results saved to: {results_dir / 'bitcoin_ml_results.json'}")

