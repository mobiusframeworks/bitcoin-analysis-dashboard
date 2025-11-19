#!/usr/bin/env python3
"""
Clean and prepare Bitcoin data with proper feature engineering
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

print("="*80)
print("CLEANING AND PREPARING BITCOIN DATA")
print("="*80)

# Load current data
DATA_PATH = Path(__file__).parent / "data" / "bitcoin_current.csv"
df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'], format='mixed')
df = df.sort_values('Date')

print(f"\n📊 Loaded {len(df)} records")
print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")

# Remove the incomplete last record if it's just a price snapshot
# (all OHLC values are the same)
last_row = df.iloc[-1]
if (last_row['Open'] == last_row['High'] == last_row['Low'] == last_row['Close']):
    print(f"\n⚠️  Removing incomplete snapshot at {last_row['Date']}")
    df = df.iloc[:-1]
    print(f"   New latest: {df['Date'].max()}")

# Remove any duplicate dates (keep last)
duplicates = df['Date'].duplicated().sum()
if duplicates > 0:
    print(f"\n⚠️  Removing {duplicates} duplicate dates")
    df = df.drop_duplicates(subset=['Date'], keep='last')

# Sort by date
df = df.sort_values('Date').reset_index(drop=True)

# Calculate technical indicators
print("\n📈 Calculating technical indicators...")

# Moving averages
df['SMA_50'] = df['Close'].rolling(50).mean()
df['SMA_200'] = df['Close'].rolling(200).mean()
df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

# Volatility
df['Volatility_20d'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(365)

# Price ratios
df['Price_SMA50_Ratio'] = df['Close'] / df['SMA_50']
df['Price_SMA200_Ratio'] = df['Close'] / df['SMA_200']

# MACD
ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema_12 - ema_26
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

# Halving cycle feature
HALVING_DATE = datetime(2024, 4, 19)
df['Days_Since_Halving'] = (df['Date'] - HALVING_DATE).dt.days

# Save cleaned data
OUTPUT_PATH = Path(__file__).parent / "data" / "bitcoin_clean.csv"
df.to_csv(OUTPUT_PATH, index=False)

print(f"\n✅ Cleaned data saved to: {OUTPUT_PATH.name}")
print(f"   Total records: {len(df)}")
print(f"   Latest date: {df['Date'].max()}")

# Print summary of latest values
latest = df.iloc[-1]
print(f"\n📊 Latest Data Point:")
print(f"   Date: {latest['Date']}")
print(f"   Close: ${latest['Close']:,.2f}")
print(f"   SMA-50: ${latest['SMA_50']:,.2f} ({((latest['Close']/latest['SMA_50']-1)*100):+.2f}%)")
print(f"   SMA-200: ${latest['SMA_200']:,.2f} ({((latest['Close']/latest['SMA_200']-1)*100):+.2f}%)")
print(f"   EMA-50: ${latest['EMA_50']:,.2f}")
print(f"   Volatility: {latest['Volatility_20d']*100:.2f}%")
print(f"   Days since halving: {latest['Days_Since_Halving']:.0f}")

print("\n" + "="*80)
print("DATA PREPARATION COMPLETE")
print("="*80)
