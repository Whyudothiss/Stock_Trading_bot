import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import json
import requests
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

def get_twelvedata_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetches 5-min intraday data from Twelve Data for a single day."""
    print(f"\nFetching 5-min data for {ticker} from Twelve Data on {start_date}...")
    load_dotenv()
    api_key = os.getenv("TWELVEDATA_API_KEY")
    if not api_key:
        print("❌ Twelve Data API key not found. Set TWELVEDATA_API_KEY in your environment.")
        return pd.DataFrame()

    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": ticker,
        "interval": "5min",
        "start_date": f"{start_date} 09:30:00",
        "end_date": f"{end_date} 16:00:00",
        "apikey": api_key,
        "timezone": "America/New_York",
        "format": "JSON"
    }
    try:
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "ok" or "values" not in data:
            print(f"⚠️  Twelve Data returned no data for {ticker}: {data.get('message', 'No message')}")
            return pd.DataFrame()
        bars = data["values"]
        df = pd.DataFrame(bars)
        if df.empty:
            print(f"⚠️  Twelve Data returned no bars for {ticker}.")
            return pd.DataFrame()
        # Convert columns to numeric
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        # Convert datetime to pandas datetime and set as index
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        # Rename columns to match yfinance
        df.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume"
        }, inplace=True)
        print(f"✅ Found {len(df)} 5-min bars from Twelve Data.")
        return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        print(f"❌ Error fetching from Twelve Data: {e}")
        return pd.DataFrame()
def compare_yfinance_vs_twelvedata(ticker: str, start_date: str, end_date: str):
    """Fetches 5-min data from yfinance and Twelve Data for a single day and prints a comparison."""
    yf_end_date = (datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    yf_df = get_yfinance_data(ticker, start_date, yf_end_date)
    td_df = get_twelvedata_data(ticker, start_date, end_date)

    if yf_df.empty or td_df.empty:
        print(f"\nOne of the sources returned no data for {ticker} on {start_date}.")
        print("\nComparison skipped due to missing data from one or both sources.")
        return

    print("\n" + "="*80)
    print(f"5-MIN DATA COMPARISON REPORT (yfinance vs. Twelve Data) for {ticker} on {start_date}")
    print("="*80)

    # Align dataframes on the index (datetime)
    aligned_yf, aligned_td = yf_df.align(td_df, join='outer', axis=0)

    # Find times present in one but not the other
    yf_only_times = aligned_yf[aligned_td['Open'].isna()].index
    td_only_times = aligned_td[aligned_yf['Open'].isna()].index

    if len(yf_only_times) > 0:
        print(f"\n⚠️ Times found only in yfinance data ({len(yf_only_times)}):")
        print([str(t) for t in yf_only_times])
    if len(td_only_times) > 0:
        print(f"\n⚠️ Times found only in Twelve Data data ({len(td_only_times)}):")
        print([str(t) for t in td_only_times])

    # Compare common times
    common_yf = aligned_yf.dropna()
    common_td = aligned_td.dropna()
    if common_yf.empty:
        print("\nNo common 5-min bars found to compare.")
        return

    # Flatten MultiIndex columns from yfinance if present
    if isinstance(common_yf.columns, pd.MultiIndex):
        common_yf.columns = [col[0] for col in common_yf.columns]

    # Ensure columns match and are in the same order (intersection only)
    common_cols = [col for col in ['Open', 'High', 'Low', 'Close', 'Volume'] if col in common_yf.columns and col in common_td.columns]
    common_yf = common_yf[common_cols]
    common_td = common_td[common_cols]

    # Align only on index (axis=0), not columns
    common_yf, common_td = common_yf.align(common_td, join='inner', axis=0)

    # Standardize index names
    common_yf.index.name = 'Datetime'
    common_td.index.name = 'Datetime'

    # Debug: print columns, index, and index names to help diagnose issues
    print("\n[DEBUG] Columns in yfinance DataFrame:", common_yf.columns.tolist())
    print("[DEBUG] Columns in Twelve Data DataFrame:", common_td.columns.tolist())
    print("[DEBUG] Index in yfinance DataFrame:", common_yf.index)
    print("[DEBUG] Index in Twelve Data DataFrame:", common_td.index)
    print("[DEBUG] Index name in yfinance DataFrame:", common_yf.index.name)
    print("[DEBUG] Index name in Twelve Data DataFrame:", common_td.index.name)

    # Use pandas compare, rounding to handle minor float precision differences
    comparison = common_yf.round(4).compare(common_td.round(4))

    if comparison.empty:
        print(f"\n✅ SUCCESS: Data for {len(common_yf)} common 5-min bars is identical.")
    else:
        print(f"\n❌ DIFFERENCES FOUND in data for common 5-min bars.")
        print("Showing rows with mismatched values (yfinance vs. Twelve Data):")
        print("-" * 80)
        with pd.option_context('display.max_rows', 20):
            print(comparison)
        print("-" * 80)

        # Set thresholds for what counts as a 'real' difference
        price_threshold = 0.05  # $0.05 for OHLC
        volume_threshold = 100  # 100 shares for volume

        # Print a side-by-side comparison for mismatched rows, highlighting only significant differences
        mismatched_times = comparison.index.get_level_values(0).unique()
        sample_times = mismatched_times  # Show all mismatched times
        print("\nSample mismatched rows (side-by-side, significant diffs highlighted):")
        for dt in sample_times:
            yf_row = common_yf.loc[dt]
            td_row = common_td.loc[dt]
            print(f"\nDatetime: {dt}")
            print(f"{'Field':<8} | {'yfinance':>12} | {'TwelveData':>12} | {'Diff':>10} | {'Significant?'}")
            print("-" * 55)
            for col in common_yf.columns:
                yf_val = yf_row[col]
                td_val = td_row[col]
                diff = abs(yf_val - td_val)
                if col == 'Volume':
                    significant = diff > volume_threshold
                else:
                    significant = diff > price_threshold
                sig_str = 'YES' if significant else ''
                print(f"{col:<8} | {yf_val:12.4f} | {td_val:12.4f} | {diff:10.4f} | {sig_str}")
        print("\n(Only differences above $0.05 for prices or 100 shares for volume are marked as significant.)")
#!/usr/bin/env python3
"""
Data Source Comparator

A utility script to fetch and compare historical daily stock data from two different sources:
1. yfinance (Yahoo Finance)
2. Alpaca Markets

This helps in identifying discrepancies, missing data, or differences in data adjustment
(e.g., for splits and dividends) between the two providers.

**Prerequisites:**
- Install necessary libraries: pip install pandas yfinance alpaca-trade-api python-dotenv
- Set up your Alpaca API keys in a .env file in the project root, or as environment variables:
  APCA_API_KEY_ID="YOUR_KEY_ID"
  APCA_API_SECRET_KEY="YOUR_SECRET_KEY"
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import requests
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))


def get_yfinance_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetches 5-min intraday data from yfinance for a single day."""
    print(f"Fetching 5-min data for {ticker} from yfinance on {start_date}...")
    try:
        # yfinance intraday data: interval='5m', period max 60d, but for a single day, use start/end
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval='5m',
            progress=False,
            auto_adjust=True
        )
        if df.empty:
            print(f"⚠️  yfinance returned no data for {ticker}.")
            return pd.DataFrame()

        # CRITICAL FIX: Convert yfinance's (often UTC) index to New York time, then remove timezone info for comparison.
        df.index = df.index.tz_convert('America/New_York').tz_localize(None)

        print(f"✅ Found {len(df)} 5-min bars from yfinance.")
        return df
    except Exception as e:
        print(f"❌ Error fetching from yfinance: {e}")
        return pd.DataFrame()



def get_alpaca_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetches 5-min intraday data from Alpaca using direct HTTP requests for a single day."""
    print(f"\nFetching 5-min data for {ticker} from Alpaca on {start_date}...")
    load_dotenv()
    api_key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        print("❌ Alpaca API keys not found.")
        print("   Please set either (APCA_API_KEY_ID, APCA_API_SECRET_KEY) or (ALPACA_API_KEY, ALPACA_SECRET_KEY) in your environment.")
        return pd.DataFrame()

    url = f"https://data.alpaca.markets/v2/stocks/{ticker}/bars"
    params = {
        "start": f"{start_date}T09:30:00-05:00",  # NYSE open
        "end": f"{end_date}T16:00:00-05:00",      # NYSE close
        "timeframe": "5Min",
        "adjustment": "all",
        "limit": 1000
    }
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key
    }
    try:
        resp = requests.get(url, params=params, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        bars = data.get("bars", [])
        if not bars:
            print(f"⚠️  Alpaca returned no data for {ticker}.")
            return pd.DataFrame()
        df = pd.DataFrame(bars)
        # Convert timestamp to datetime and set as index
        # CRITICAL FIX: Convert Alpaca's UTC index to New York time, then remove timezone info for comparison.
        df['timestamp'] = pd.to_datetime(df['t']).dt.tz_convert('America/New_York').dt_localize(None)

        df.set_index('timestamp', inplace=True)
        # Rename columns to match yfinance
        df.rename(columns={
            'o': 'Open',
            'h': 'High',
            'l': 'Low',
            'c': 'Close',
            'v': 'Volume'
        }, inplace=True)
        print(f"✅ Found {len(df)} 5-min bars from Alpaca.")
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        print(f"❌ Error fetching from Alpaca: {e}")
        return pd.DataFrame()

def get_twelvedata_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetches 5-min intraday data from Twelve Data for a single day."""
    print(f"\nFetching 5-min data for {ticker} from Twelve Data on {start_date}...")
    load_dotenv()
    api_key = os.getenv("TWELVEDATA_API_KEY")
    if not api_key:
        print("❌ Twelve Data API key not found. Set TWELVEDATA_API_KEY in your environment.")
        return pd.DataFrame()

    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": ticker, "interval": "5min",
        "start_date": f"{start_date} 09:30:00", "end_date": f"{end_date} 16:00:00",
        "apikey": api_key, "timezone": "America/New_York", "format": "JSON"
    }
    try:
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "ok" or "values" not in data:
            print(f"⚠️  Twelve Data returned no data for {ticker}: {data.get('message', 'No message')}")
            return pd.DataFrame()
        
        df = pd.DataFrame(data["values"])
        if df.empty: return pd.DataFrame()

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Twelve Data already provides NY time, so just convert and remove timezone info.
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        df.rename(columns={
            "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"
        }, inplace=True)
        print(f"✅ Found {len(df)} 5-min bars from Twelve Data.")
        return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        print(f"❌ Error fetching from Twelve Data: {e}")
        return pd.DataFrame()

def compare_data_sources(ticker: str, start_date: str, end_date: str):
    """Fetches 5-min data from both sources for a single day and prints a comparison."""
    # For 5-min data, both start_date and end_date should be the same (single day)
    # yfinance's end is exclusive, so add one day
    yf_end_date = (datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')

    # Fetch data
    yf_df = get_yfinance_data(ticker, start_date, yf_end_date)
    td_df = get_twelvedata_data(ticker, start_date, end_date)

    if yf_df.empty or td_df.empty:
        print(f"\nOne of the sources returned no data for {ticker} on {start_date}.")
        print("\nComparison skipped due to missing data from one or both sources.")
        return

    print("\n" + "="*80)
    print(f"5-MIN DATA COMPARISON REPORT (yfinance vs. Twelve Data) for {ticker} on {start_date}")
    print("="*80)

    # Align dataframes on the index (datetime)
    aligned_yf, aligned_td = yf_df.align(td_df, join='outer', axis=0)

    # Find times present in one but not the other
    yf_only_times = aligned_yf[aligned_td['Open'].isna()].index
    td_only_times = aligned_td[aligned_yf['Open'].isna()].index

    if len(yf_only_times) > 0:
        print(f"\n⚠️ Times found only in yfinance data ({len(yf_only_times)}):")
        print([str(t) for t in yf_only_times])
    if len(td_only_times) > 0:
        print(f"\n⚠️ Times found only in Twelve Data data ({len(td_only_times)}):")
        print([str(t) for t in td_only_times])

    # Compare common times
    common_yf = aligned_yf.dropna()
    common_td = aligned_td.dropna()
    if common_yf.empty:
        print("\nNo common 5-min bars found to compare.")
        return

    # Flatten MultiIndex columns from yfinance if present
    if isinstance(common_yf.columns, pd.MultiIndex):
        common_yf.columns = [col[0] for col in common_yf.columns]

    # Ensure columns match and are in the same order (intersection only)
    common_cols = [col for col in ['Open', 'High', 'Low', 'Close', 'Volume'] if col in common_yf.columns and col in common_td.columns]
    common_yf = common_yf[common_cols]
    common_td = common_td[common_cols]

    # Align only on index (axis=0), not columns
    common_yf, common_td = common_yf.align(common_td, join='inner', axis=0)

    # Standardize index names
    common_yf.index.name = 'Datetime'
    common_td.index.name = 'Datetime'

    # Debug: print columns, index, and index names to help diagnose issues
    print("\n[DEBUG] Columns in yfinance DataFrame:", common_yf.columns.tolist())
    print("[DEBUG] Columns in Twelve Data DataFrame:", common_td.columns.tolist())
    print("[DEBUG] Index in yfinance DataFrame:", common_yf.index)
    print("[DEBUG] Index in Twelve Data DataFrame:", common_td.index)
    print("[DEBUG] Index name in yfinance DataFrame:", common_yf.index.name)
    print("[DEBUG] Index name in Twelve Data DataFrame:", common_td.index.name)

    # Use pandas compare, rounding to handle minor float precision differences
    comparison = common_yf.round(4).compare(common_td.round(4))

    if comparison.empty:
        print(f"\n✅ SUCCESS: Data for {len(common_yf)} common 5-min bars is identical.")
    else:
        print(f"\n❌ DIFFERENCES FOUND in data for common 5-min bars.")
        print("Showing rows with mismatched values (yfinance vs. Twelve Data):")
        print("-" * 80)
        with pd.option_context('display.max_rows', 20):
            print(comparison)
        print("-" * 80)

        # Set thresholds for what counts as a 'real' difference
        price_threshold = 0.05  # $0.05 for OHLC
        volume_threshold = 100  # 100 shares for volume

        # Print a side-by-side comparison for mismatched rows, highlighting only significant differences
        mismatched_times = comparison.index.get_level_values(0).unique()
        sample_times = mismatched_times 
        print("\nSample mismatched rows (side-by-side, significant diffs highlighted):")
        for dt in sample_times:
            yf_row = common_yf.loc[dt]
            td_row = common_td.loc[dt]
            print(f"\nDatetime: {dt}")
            print(f"{'Field':<8} | {'yfinance':>12} | {'TwelveData':>12} | {'Diff':>10} | {'Significant?'}")
            print("-" * 55)
            for col in common_yf.columns:
                yf_val = yf_row[col]
                td_val = td_row[col]
                diff = abs(yf_val - td_val)
                if col == 'Volume':
                    significant = diff > volume_threshold
                else:
                    significant = diff > price_threshold
                sig_str = 'YES' if significant else ''
                print(f"{col:<8} | {yf_val:12.4f} | {td_val:12.4f} | {diff:10.4f} | {sig_str}")
        print("\n(Only differences above $0.05 for prices or 100 shares for volume are marked as significant.)")


if __name__ == "__main__":
    # --- Parameters for comparison ---
    TICKER_TO_CHECK = "RGTI"
    DATE_TO_CHECK = "2025-11-14"  # Single day for 5-min bars
    # ---------------------------------

    # Compare yfinance vs. Twelve Data
    compare_data_sources(TICKER_TO_CHECK, DATE_TO_CHECK, DATE_TO_CHECK)