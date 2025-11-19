#!/usr/bin/env python3
"""
Test Different Feature Selection Strategies

Tests multiple combinations of thresholds and methods to find the best approach
for reducing overfitting by eliminating redundant features.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from aggressive_feature_selection import aggressive_feature_selection
from overfitting_detector import comprehensive_overfitting_check
from model_selection_pipeline import run_model_selection
import pandas as pd
import numpy as np
import json
from itertools import product

print("="*80)
print("TESTING FEATURE SELECTION STRATEGIES")
print("="*80)
print()

# Load and prepare data (same as before)
DATA_PATH = "datasets/Bitcoin/bitcoin data csvs/btc-ohlc.csv"
df = pd.read_csv(DATA_PATH)
df['d'] = pd.to_datetime(df['d'], format='mixed', errors='coerce')
df = df.sort_values('d').reset_index(drop=True)

# Create features (same as before)
lag = 1
df['prev_close'] = df['close'].shift(lag)
df['prev_open'] = df['open'].shift(lag)
df['prev_high'] = df['high'].shift(lag)
df['prev_low'] = df['low'].shift(lag)
df['prev_volume'] = df['volume'].shift(lag)
df['return_1d'] = df['close'].pct_change(1).shift(lag)
df['return_5d'] = df['close'].pct_change(5).shift(lag)
df['return_20d'] = df['close'].pct_change(20).shift(lag)
df['volatility_5d'] = df['close'].shift(1).rolling(window=5).std()
df['volatility_20d'] = df['close'].shift(1).rolling(window=20).std()
df['sma_20'] = df['close'].shift(1).rolling(window=20).mean()
df['sma_50'] = df['close'].shift(1).rolling(window=50).mean()
df['sma_200'] = df['close'].shift(1).rolling(window=200).mean()
df['price_sma20_ratio'] = df['prev_close'] / df['sma_20']
df['price_sma50_ratio'] = df['prev_close'] / df['sma_50']
df['price_sma200_ratio'] = df['prev_close'] / df['sma_200']
df['volume_sma_20'] = df['volume'].shift(1).rolling(window=20).mean()
df['volume_ratio'] = df['prev_volume'] / df['volume_sma_20']
for lag_days in [2, 3, 5, 10]:
    df[f'close_lag_{lag_days}'] = df['close'].shift(lag_days)

target = df['close'].pct_change(20).shift(-20)

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
            df_with_fred = df.set_index('d')
            for col in ['Net_Liquidity', 'M2SL', 'M2_Growth_Rate', 'WALCL', 
                       'FEDFUNDS', 'CPIAUCSL', 'CPI_YoY', 'UNRATE', 'GFDEBTN', 'DGS10']:
                if col in fred_df.columns:
                    merged = pd.merge_asof(
                        df_with_fred.reset_index(),
                        fred_df[[col]].reset_index(),
                        left_on='d',
                        right_on='Date',
                        direction='backward',
                        tolerance=pd.Timedelta('30d')
                    )
                    df[col] = merged[col].values
                    df[col] = df[col].ffill()
                    initial_features.append(col)
            break
        except:
            continue

df_features = df[initial_features + ['d']].copy()
valid_mask = ~target.isna()
df_features = df_features[valid_mask].copy()
target = target[valid_mask].copy()
df_features = df_features.dropna()
target = target.loc[df_features.index]

X = df_features[initial_features].values
y = target.values

print(f"Initial features: {len(initial_features)}")
print(f"Data shape: {X.shape}")
print()

# Test different strategies
strategies = [
    {'corr_threshold': 0.80, 'vif_threshold': 5.0, 'n_features': 10, 'name': 'Moderate'},
    {'corr_threshold': 0.85, 'vif_threshold': 5.0, 'n_features': 8, 'name': 'Aggressive'},
    {'corr_threshold': 0.90, 'vif_threshold': 3.0, 'n_features': 8, 'name': 'Very Aggressive'},
    {'corr_threshold': 0.85, 'vif_threshold': 3.0, 'n_features': 6, 'name': 'Extreme'},
    {'corr_threshold': 0.90, 'vif_threshold': 5.0, 'n_features': 10, 'name': 'High Corr, Mod VIF'},
]

results = []

for strategy in strategies:
    print(f"\n{'='*80}")
    print(f"Testing Strategy: {strategy['name']}")
    print(f"  Correlation threshold: {strategy['corr_threshold']}")
    print(f"  VIF threshold: {strategy['vif_threshold']}")
    print(f"  Final features: {strategy['n_features']}")
    print(f"{'='*80}")
    
    try:
        selected_features, report = aggressive_feature_selection(
            X, y,
            feature_names=initial_features,
            correlation_threshold=strategy['corr_threshold'],
            vif_threshold=strategy['vif_threshold'],
            use_rfe=True,
            n_features_final=strategy['n_features']
        )
        
        print(f"Selected {len(selected_features)} features")
        print(f"Features: {', '.join(selected_features)}")
        
        # Quick model test (simplified)
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score
        from sklearn.preprocessing import StandardScaler
        
        X_selected = df_features[selected_features].values
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = Ridge(alpha=100.0)  # Strong regularization
        model.fit(X_train_scaled, y_train)
        
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
        
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        gap = abs(train_r2 - test_r2) / abs(test_r2) if test_r2 != 0 else float('inf')
        
        results.append({
            'strategy': strategy['name'],
            'corr_threshold': strategy['corr_threshold'],
            'vif_threshold': strategy['vif_threshold'],
            'n_features': len(selected_features),
            'selected_features': selected_features,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'gap': gap,
            'overfitting': gap > 0.15
        })
        
        print(f"  Train R²: {train_r2:.4f}")
        print(f"  Test R²: {test_r2:.4f}")
        print(f"  Gap: {gap:.2%}")
        print(f"  Overfitting: {'YES' if gap > 0.15 else 'NO'}")
        
    except Exception as e:
        print(f"  Error: {e}")
        continue

# Summary
print(f"\n{'='*80}")
print("STRATEGY COMPARISON")
print(f"{'='*80}")
print()

results_df = pd.DataFrame(results)
if len(results_df) > 0:
    print(results_df[['strategy', 'n_features', 'train_r2', 'test_r2', 'gap', 'overfitting']].to_string(index=False))
    
    # Find best strategy (lowest gap, reasonable test R²)
    results_df['score'] = results_df['test_r2'] - results_df['gap']  # Higher is better
    best = results_df.loc[results_df['score'].idxmax()]
    
    print(f"\nBest Strategy: {best['strategy']}")
    print(f"  Features: {best['n_features']}")
    print(f"  Test R²: {best['test_r2']:.4f}")
    print(f"  Gap: {best['gap']:.2%}")
    print(f"  Selected features: {', '.join(best['selected_features'])}")
    
    # Save results
    output_path = Path("ml_pipeline/feature_selection_strategy_comparison.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

print()



