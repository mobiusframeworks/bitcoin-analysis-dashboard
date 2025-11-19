#!/usr/bin/env python3
"""
Fetch hourly Bitcoin data from yfinance for real-time updates
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime

print("="*80)
print("FETCHING HOURLY BITCOIN DATA (yfinance)")
print("="*80)
print()

# Paths
BASE_DIR = Path(__file__).parent.parent
OUTPUT_FILE = BASE_DIR / "datasets" / "btc-ohlc.csv"

print(f"📡 Fetching BTC-USD from Yahoo Finance...")
print(f"   Interval: 1 hour")
print(f"   Period: 730 days (2 years hourly data)")
print()

try:
    # Fetch hourly data for last 2 years
    btc = yf.Ticker("BTC-USD")
    df = btc.history(period="730d", interval="1h")

    if df.empty:
        raise ValueError("No data returned from yfinance")

    # Reset index to get timestamp column
    df = df.reset_index()

    # Rename columns to match expected format
    df = df.rename(columns={
        'Datetime': 'timestamp',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })

    # Add unix timestamp
    df['unix_time'] = df['timestamp'].astype(int) // 10**9

    # Reorder columns
    df = df[['timestamp', 'unix_time', 'open', 'high', 'low', 'close', 'volume']]

    # Sort by timestamp
    df = df.sort_values('timestamp')

    # Remove duplicates
    df = df.drop_duplicates(subset=['timestamp'], keep='last')

    print(f"✅ Fetched {len(df)} hourly records")
    print(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"💰 Latest price: ${df['close'].iloc[-1]:,.2f}")
    print(f"⏰ Last update: {df['timestamp'].iloc[-1]}")
    print()

    # Save to CSV
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Saved to: {OUTPUT_FILE}")

except Exception as e:
    print(f"❌ Error fetching data: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print()
print("="*80)
print("HOURLY DATA FETCH COMPLETE")
print("="*80)
print()
print(f"🔄 Next update: Every hour on the hour")
print(f"📊 Total records: {len(df):,}")
print(f"💾 File size: {OUTPUT_FILE.stat().st_size / 1024 / 1024:.2f} MB")
