#!/usr/bin/env python3
"""
Fetch current Bitcoin price from free APIs and update data
"""

import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
import json

print("="*80)
print("FETCHING CURRENT BITCOIN PRICE")
print("="*80)

# Try multiple free APIs
current_price = None
current_date = datetime.now()

# Try Coinbase API
try:
    print("\n📡 Trying Coinbase API...")
    response = requests.get('https://api.coinbase.com/v2/prices/BTC-USD/spot', timeout=10)
    data = response.json()
    current_price = float(data['data']['amount'])
    print(f"✅ Coinbase: ${current_price:,.2f}")
except Exception as e:
    print(f"❌ Coinbase failed: {e}")

# Try CoinGecko as backup
if not current_price:
    try:
        print("\n📡 Trying CoinGecko API...")
        response = requests.get('https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd', timeout=10)
        data = response.json()
        current_price = float(data['bitcoin']['usd'])
        print(f"✅ CoinGecko: ${current_price:,.2f}")
    except Exception as e:
        print(f"❌ CoinGecko failed: {e}")

if not current_price:
    print("\n❌ Could not fetch current price from any API")
    exit(1)

print(f"\n✅ Current Bitcoin Price: ${current_price:,.2f}")
print(f"   Date: {current_date.strftime('%Y-%m-%d %H:%M:%S')}")

# Load existing data
DATA_PATH = Path(__file__).parent / "data" / "bitcoin_prepared.csv"
df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'], format='mixed')

print(f"\n📊 Existing data:")
print(f"   Records: {len(df)}")
print(f"   Latest: {df['Date'].max()}")
print(f"   Last price: ${df['Close'].iloc[-1]:,.2f}")

# Add current price as new row
new_row = {
    'Date': current_date,
    'Open': current_price,
    'High': current_price,
    'Low': current_price,
    'Close': current_price,
    'Volume': df['Volume'].iloc[-1] if 'Volume' in df.columns else 0
}

# Append and save
df_updated = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
df_updated = df_updated.sort_values('Date')
df_updated = df_updated.drop_duplicates(subset=['Date'], keep='last')

OUTPUT_PATH = Path(__file__).parent / "data" / "bitcoin_current.csv"
df_updated.to_csv(OUTPUT_PATH, index=False)

print(f"\n✅ Updated data saved to: {OUTPUT_PATH.name}")
print(f"   Total records: {len(df_updated)}")
print(f"   Latest: {df_updated['Date'].max()}")
print(f"   Current price: ${df_updated['Close'].iloc[-1]:,.2f}")

# Calculate price change
price_change = current_price - df['Close'].iloc[-1]
price_change_pct = (price_change / df['Close'].iloc[-1]) * 100

print(f"\n📉 Price Change Since Last Data:")
print(f"   Change: ${price_change:+,.2f} ({price_change_pct:+.2f}%)")

# 4th Halving context
halving_date = datetime(2024, 4, 19)
days_since_halving = (current_date - halving_date).days

print(f"\n🔄 Halving Cycle:")
print(f"   Last halving: {halving_date.strftime('%Y-%m-%d')}")
print(f"   Days since: {days_since_halving}")
print(f"   Current phase: {'POST-PEAK DISTRIBUTION' if days_since_halving > 550 else 'PEAK ZONE'}")

print("\n" + "="*80)
print("COMPLETE")
print("="*80)
