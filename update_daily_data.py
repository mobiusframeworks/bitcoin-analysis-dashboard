#!/usr/bin/env python3
"""
Daily Data Update Script
- Fetches latest Bitcoin price from Coinbase API
- Fetches latest FRED economic data (M2, Fed Funds, CPI, etc.)
- Merges all data into comprehensive dataset
- Updates btc-ohlc.csv with new data
- Regenerates all reports and website
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import requests
import time

# Configuration
BASE_DIR = Path("/Users/alexhorton/quant connect dev environment")
DATASETS_DIR = BASE_DIR / "datasets"
ML_PIPELINE_DIR = BASE_DIR / "ml_pipeline"
REPORTS_DIR = ML_PIPELINE_DIR / "reports"

BTC_OHLC_FILE = DATASETS_DIR / "btc-ohlc.csv"
COMPREHENSIVE_DATA_FILE = DATASETS_DIR / "btc_comprehensive_data.csv"

# FRED API (free tier, no key needed for public data via alternative endpoint)
# For production, get a free API key from https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY = "YOUR_FRED_API_KEY_HERE"  # Replace with actual key

print("=" * 80)
print("DAILY DATA UPDATE SCRIPT")
print("=" * 80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ============================================================================
# STEP 1: FETCH LATEST BITCOIN PRICE
# ============================================================================
print("📊 STEP 1: Fetching latest Bitcoin price from Coinbase...")

def fetch_latest_bitcoin_price():
    """Fetch current Bitcoin price from Coinbase API"""
    try:
        url = "https://api.coinbase.com/v2/prices/BTC-USD/spot"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        price = float(data['data']['amount'])
        print(f"✅ Current Bitcoin price: ${price:,.2f}")
        return price
    except Exception as e:
        print(f"❌ Error fetching Coinbase price: {e}")
        return None

def fetch_bitcoin_ohlc_today():
    """Fetch today's OHLC from Coinbase Pro API"""
    try:
        # Coinbase Pro (Advanced Trade) API for OHLC
        # Granularity: 86400 = 1 day
        url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
        params = {
            'granularity': 86400,  # 1 day
            'start': (datetime.utcnow() - timedelta(days=1)).isoformat(),
            'end': datetime.utcnow().isoformat()
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data and len(data) > 0:
            # Coinbase returns: [time, low, high, open, close, volume]
            latest = data[0]

            ohlc = {
                'timestamp': datetime.utcfromtimestamp(latest[0]),
                'unix_time': latest[0],
                'open': latest[3],
                'high': latest[2],
                'low': latest[1],
                'close': latest[4],
                'volume': latest[5]
            }

            print(f"✅ Latest OHLC data:")
            print(f"   Date: {ohlc['timestamp']}")
            print(f"   Open: ${ohlc['open']:,.2f}")
            print(f"   High: ${ohlc['high']:,.2f}")
            print(f"   Low: ${ohlc['low']:,.2f}")
            print(f"   Close: ${ohlc['close']:,.2f}")
            print(f"   Volume: {ohlc['volume']:,.2f} BTC")

            return ohlc
        else:
            print("⚠️  No OHLC data returned from Coinbase")
            return None

    except Exception as e:
        print(f"❌ Error fetching OHLC: {e}")
        return None

latest_ohlc = fetch_bitcoin_ohlc_today()

# ============================================================================
# STEP 2: UPDATE BTC-OHLC.CSV
# ============================================================================
print("\n📊 STEP 2: Updating btc-ohlc.csv...")

def update_btc_ohlc_csv(new_ohlc):
    """Add latest OHLC to CSV if not already present"""
    try:
        # Load existing data
        df = pd.read_csv(BTC_OHLC_FILE)
        df.columns = ['timestamp', 'unix_time', 'open', 'high', 'low', 'close', 'volume']
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Check if today's data already exists
        new_date = new_ohlc['timestamp'].date()
        df['date'] = pd.to_datetime(df['timestamp']).dt.date

        if new_date in df['date'].values:
            print(f"ℹ️  Data for {new_date} already exists in CSV")
            # Update the existing row with latest data
            mask = df['date'] == new_date
            df.loc[mask, 'open'] = new_ohlc['open']
            df.loc[mask, 'high'] = new_ohlc['high']
            df.loc[mask, 'low'] = new_ohlc['low']
            df.loc[mask, 'close'] = new_ohlc['close']
            df.loc[mask, 'volume'] = new_ohlc['volume']
            print(f"✅ Updated existing row for {new_date}")
        else:
            # Append new row
            new_row = pd.DataFrame([{
                'timestamp': new_ohlc['timestamp'],
                'unix_time': new_ohlc['unix_time'],
                'open': new_ohlc['open'],
                'high': new_ohlc['high'],
                'low': new_ohlc['low'],
                'close': new_ohlc['close'],
                'volume': new_ohlc['volume']
            }])
            df = pd.concat([df, new_row], ignore_index=True)
            print(f"✅ Added new row for {new_date}")

        # Remove temporary date column and sort
        df = df.drop('date', axis=1)
        df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='last')

        # Save back to CSV
        df.to_csv(BTC_OHLC_FILE, index=False)
        print(f"✅ Saved updated data to {BTC_OHLC_FILE}")
        print(f"   Total records: {len(df)}")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        return df

    except Exception as e:
        print(f"❌ Error updating CSV: {e}")
        return None

if latest_ohlc:
    btc_df = update_btc_ohlc_csv(latest_ohlc)
else:
    print("⚠️  Skipping CSV update - no new OHLC data")
    btc_df = pd.read_csv(BTC_OHLC_FILE)
    btc_df.columns = ['timestamp', 'unix_time', 'open', 'high', 'low', 'close', 'volume']
    btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'])

# ============================================================================
# STEP 3: FETCH FRED ECONOMIC DATA
# ============================================================================
print("\n📊 STEP 3: Fetching FRED economic indicators...")

def fetch_fred_series(series_id, series_name):
    """Fetch a FRED data series"""
    try:
        if FRED_API_KEY == "YOUR_FRED_API_KEY_HERE":
            print(f"⚠️  FRED API key not configured - using fallback method for {series_name}")
            # Use FRED CSV download endpoint (public, no key needed)
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
            df = pd.read_csv(url)
            df.columns = ['date', series_id]
            df['date'] = pd.to_datetime(df['date'])
            df = df[df[series_id] != '.']  # Remove missing values marked as '.'
            df[series_id] = pd.to_numeric(df[series_id], errors='coerce')
            print(f"✅ Fetched {series_name} ({len(df)} records)")
            return df
        else:
            # Use official FRED API
            url = f"https://api.stlouisfed.org/fred/series/observations"
            params = {
                'series_id': series_id,
                'api_key': FRED_API_KEY,
                'file_type': 'json'
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            df = pd.DataFrame(data['observations'])
            df['date'] = pd.to_datetime(df['date'])
            df[series_id] = pd.to_numeric(df['value'], errors='coerce')
            df = df[['date', series_id]]
            print(f"✅ Fetched {series_name} ({len(df)} records)")
            return df

    except Exception as e:
        print(f"❌ Error fetching {series_name}: {e}")
        return None

# Fetch key FRED series
fred_series = {
    'M2SL': 'M2 Money Supply',
    'FEDFUNDS': 'Federal Funds Rate',
    'DGS10': '10-Year Treasury Rate',
    'CPIAUCSL': 'Consumer Price Index',
    'WALCL': 'Fed Balance Sheet',
    'DEXUSEU': 'USD/EUR Exchange Rate',
    'DCOILWTICO': 'WTI Crude Oil Price'
}

fred_data = {}
for series_id, series_name in fred_series.items():
    df = fetch_fred_series(series_id, series_name)
    if df is not None:
        fred_data[series_id] = df
    time.sleep(0.5)  # Rate limiting

# ============================================================================
# STEP 4: CREATE COMPREHENSIVE DATASET
# ============================================================================
print("\n📊 STEP 4: Creating comprehensive dataset (BTC + FRED)...")

def create_comprehensive_dataset(btc_df, fred_data):
    """Merge Bitcoin OHLC with all FRED indicators"""
    try:
        # Start with Bitcoin data
        df = btc_df.copy()
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        df['date'] = pd.to_datetime(df['date'])

        print(f"Starting with {len(df)} Bitcoin records")

        # Merge each FRED series
        for series_id, fred_df in fred_data.items():
            fred_df = fred_df.copy()
            fred_df['date'] = pd.to_datetime(fred_df['date']).dt.date
            fred_df['date'] = pd.to_datetime(fred_df['date'])

            # Merge on date
            df = pd.merge(df, fred_df, on='date', how='left')

            # Forward fill FRED data (monthly/weekly data -> daily)
            df[series_id] = df[series_id].fillna(method='ffill')

            print(f"  Merged {series_id}: {df[series_id].notna().sum()} valid values")

        # Calculate derived indicators
        print("Calculating derived indicators...")

        # CPI Year-over-Year % change
        if 'CPIAUCSL' in df.columns:
            df['CPI_YoY'] = df['CPIAUCSL'].pct_change(365) * 100

        # Real interest rate (Fed Funds - Inflation)
        if 'FEDFUNDS' in df.columns and 'CPI_YoY' in df.columns:
            df['Real_Rate'] = df['FEDFUNDS'] - df['CPI_YoY']

        # M2 growth rate
        if 'M2SL' in df.columns:
            df['M2_Growth_30d'] = df['M2SL'].pct_change(30) * 100
            df['M2_Growth_90d'] = df['M2SL'].pct_change(90) * 100
            df['M2_Growth_365d'] = df['M2SL'].pct_change(365) * 100

        # Bitcoin technical indicators
        df['BTC_Returns_1d'] = df['close'].pct_change(1)
        df['BTC_Returns_7d'] = df['close'].pct_change(7)
        df['BTC_Returns_30d'] = df['close'].pct_change(30)
        df['BTC_MA_20'] = df['close'].rolling(window=20).mean()
        df['BTC_MA_50'] = df['close'].rolling(window=50).mean()
        df['BTC_MA_200'] = df['close'].rolling(window=200).mean()
        df['BTC_Volatility_30d'] = df['close'].pct_change().rolling(window=30).std() * np.sqrt(365) * 100

        # Save comprehensive dataset
        df.to_csv(COMPREHENSIVE_DATA_FILE, index=False)
        print(f"\n✅ Comprehensive dataset created!")
        print(f"   File: {COMPREHENSIVE_DATA_FILE}")
        print(f"   Records: {len(df)}")
        print(f"   Columns: {len(df.columns)}")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"\n   Columns included:")
        for col in df.columns:
            non_null = df[col].notna().sum()
            print(f"     - {col}: {non_null}/{len(df)} values ({non_null/len(df)*100:.1f}% coverage)")

        return df

    except Exception as e:
        print(f"❌ Error creating comprehensive dataset: {e}")
        import traceback
        traceback.print_exc()
        return None

if fred_data:
    comprehensive_df = create_comprehensive_dataset(btc_df, fred_data)
else:
    print("⚠️  No FRED data available - skipping comprehensive dataset creation")
    comprehensive_df = None

# ============================================================================
# STEP 5: REGENERATE REPORTS
# ============================================================================
print("\n📊 STEP 5: Regenerating all reports with updated data...")

def regenerate_reports():
    """Run report generation scripts with updated data"""
    import subprocess
    import os

    scripts_to_run = [
        {
            'name': 'M2 Interest Rate Study',
            'script': 'generate_m2_interest_rate_bitcoin_study.py',
            'desc': 'M2 and interest rate analysis with charts'
        },
        {
            'name': 'Trading Strategy Analysis',
            'script': 'generate_trading_strategy_analysis.py',
            'desc': '50-week SMA trading strategy with risk management'
        },
        {
            'name': 'Comprehensive Bitcoin Website',
            'script': 'generate_comprehensive_bitcoin_website.py',
            'desc': 'Main dashboard with all integrated analysis'
        }
    ]

    for item in scripts_to_run:
        print(f"\n🔄 Running: {item['name']}...")
        print(f"   ({item['desc']})")

        try:
            script_path = ML_PIPELINE_DIR / item['script']

            # Run script
            result = subprocess.run(
                [
                    'python3',
                    str(script_path)
                ],
                cwd=str(ML_PIPELINE_DIR),
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode == 0:
                print(f"✅ {item['name']} completed successfully")
            else:
                print(f"❌ {item['name']} failed:")
                print(result.stderr)

        except subprocess.TimeoutExpired:
            print(f"⏱️  {item['name']} timed out (>120s)")
        except Exception as e:
            print(f"❌ Error running {item['name']}: {e}")

regenerate_reports()

# ============================================================================
# STEP 6: SAVE UPDATE LOG
# ============================================================================
print("\n📊 STEP 6: Saving update log...")

update_log = {
    'timestamp': datetime.now().isoformat(),
    'btc_ohlc_records': len(btc_df) if btc_df is not None else 0,
    'latest_btc_price': btc_df['close'].iloc[-1] if btc_df is not None else None,
    'latest_date': btc_df['timestamp'].iloc[-1].isoformat() if btc_df is not None else None,
    'fred_series_fetched': list(fred_data.keys()) if fred_data else [],
    'comprehensive_dataset_created': comprehensive_df is not None,
    'comprehensive_records': len(comprehensive_df) if comprehensive_df is not None else 0
}

log_file = ML_PIPELINE_DIR / 'daily_update_log.json'
with open(log_file, 'w') as f:
    json.dump(update_log, f, indent=2)

print(f"✅ Update log saved: {log_file}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("DAILY UPDATE COMPLETE")
print("=" * 80)
print(f"\n📊 Summary:")
print(f"   Latest Bitcoin Price: ${update_log['latest_btc_price']:,.2f}" if update_log['latest_btc_price'] else "   Latest Bitcoin Price: N/A")
print(f"   Data Updated Through: {update_log['latest_date'][:10]}" if update_log['latest_date'] else "   Data Updated Through: N/A")
print(f"   Total BTC Records: {update_log['btc_ohlc_records']}")
print(f"   FRED Series Fetched: {len(update_log['fred_series_fetched'])}")
print(f"   Comprehensive Dataset: {'✅ Created' if update_log['comprehensive_dataset_created'] else '❌ Not created'}")

print(f"\n📁 Files Updated:")
print(f"   - {BTC_OHLC_FILE}")
if update_log['comprehensive_dataset_created']:
    print(f"   - {COMPREHENSIVE_DATA_FILE}")
print(f"   - {log_file}")

print(f"\n🌐 View Updated Dashboard:")
print(f"   http://localhost:8080/Bitcoin_Comprehensive_Dashboard.html")

print(f"\n✅ All systems updated!")
print("=" * 80)
