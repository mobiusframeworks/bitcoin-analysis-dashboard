#!/usr/bin/env python3
"""
Fetch live Bitcoin data from API and merge with historical data
"""

import sys
from pathlib import Path
import pandas as pd
import requests
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("FETCHING LIVE BITCOIN DATA")
print("="*80)
print()

# Paths
DATA_PATH = Path(__file__).parent / "data" / "bitcoin_prepared.csv"
OUTPUT_PATH = Path(__file__).parent / "data" / "bitcoin_live.csv"

# Fetch from API
API_URL = "https://bitcoin-data.com/v1/btc-ohlc"

print(f"📡 Fetching live data from: {API_URL}")

try:
    response = requests.get(API_URL, headers={'accept': 'application/hal+json'}, timeout=10)
    response.raise_for_status()

    api_data = response.json()
    print(f"✅ API response received")

    # Parse API data - check structure first
    print(f"   API structure: {type(api_data)}")
    if isinstance(api_data, dict):
        print(f"   Keys: {list(api_data.keys())[:5]}")

    if isinstance(api_data, dict) and '_embedded' in api_data:
        records = api_data['_embedded'].get('ohlc', [])
    elif isinstance(api_data, dict) and 'data' in api_data:
        records = api_data['data']
    elif isinstance(api_data, list):
        records = api_data
    else:
        records = [api_data]

    if not records:
        raise ValueError("No data records found in API response")

    # Convert to DataFrame
    api_df = pd.DataFrame(records)
    print(f"   DataFrame shape: {api_df.shape}")
    print(f"   Columns: {list(api_df.columns)}")

    # Ensure proper column names (case insensitive)
    api_df.columns = [col.lower() for col in api_df.columns]

    # Handle date column priority: d > unixts > timestamp
    if 'd' in api_df.columns:
        api_df['Date'] = pd.to_datetime(api_df['d'])
        api_df = api_df.drop(['d'], axis=1)
    elif 'unixts' in api_df.columns:
        api_df['Date'] = pd.to_datetime(api_df['unixts'], unit='s')
        api_df = api_df.drop(['unixts'], axis=1)
    elif 'timestamp' in api_df.columns:
        if pd.api.types.is_numeric_dtype(api_df['timestamp']):
            api_df['Date'] = pd.to_datetime(api_df['timestamp'], unit='s')
        else:
            api_df['Date'] = pd.to_datetime(api_df['timestamp'])
        api_df = api_df.drop(['timestamp'], axis=1)

    # Drop any remaining timestamp columns
    timestamp_cols = [col for col in api_df.columns if col in ['unixts', 'timestamp', 't', 'time']]
    if timestamp_cols:
        api_df = api_df.drop(timestamp_cols, axis=1)

    # Rename OHLCV columns
    column_mapping = {
        'open': 'Open',
        'o': 'Open',
        'high': 'High',
        'h': 'High',
        'low': 'Low',
        'l': 'Low',
        'close': 'Close',
        'c': 'Close',
        'volume': 'Volume',
        'v': 'Volume'
    }

    api_df = api_df.rename(columns={k: v for k, v in column_mapping.items() if k in api_df.columns})

    # Ensure required columns exist
    required_cols = ['Date', 'Open', 'High', 'Low', 'Close']
    missing_cols = [col for col in required_cols if col not in api_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print(f"✅ Parsed {len(api_df)} records from API")
    print(f"   Latest: {api_df['Date'].max()}")
    print(f"   Current Price: ${api_df['Close'].iloc[-1]:,.2f}")

except Exception as e:
    print(f"❌ Error fetching from API: {e}")
    print(f"⚠️  Using historical data only")
    api_df = pd.DataFrame()

# Load historical data
if DATA_PATH.exists():
    print(f"\n📊 Loading historical data from: {DATA_PATH}")
    hist_df = pd.read_csv(DATA_PATH)
    hist_df['Date'] = pd.to_datetime(hist_df['Date'], format='mixed')
    print(f"✅ Loaded {len(hist_df)} historical records")
    print(f"   Latest historical: {hist_df['Date'].max()}")
else:
    print(f"⚠️  No historical data found")
    hist_df = pd.DataFrame()

# Merge data
if not api_df.empty and not hist_df.empty:
    # Combine and remove duplicates
    combined_df = pd.concat([hist_df, api_df], ignore_index=True)
    combined_df = combined_df.sort_values('Date')
    combined_df = combined_df.drop_duplicates(subset=['Date'], keep='last')

    print(f"\n✅ Merged data: {len(combined_df)} total records")
    print(f"   Date range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")

    # Save
    combined_df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved to: {OUTPUT_PATH}")

elif not api_df.empty:
    api_df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved API data to: {OUTPUT_PATH}")

elif not hist_df.empty:
    hist_df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Using historical data: {OUTPUT_PATH}")

else:
    print(f"❌ No data available")
    sys.exit(1)

print("\n" + "="*80)
print("DATA FETCH COMPLETE")
print("="*80)
