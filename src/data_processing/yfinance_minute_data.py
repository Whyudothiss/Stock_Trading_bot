#!/usr/bin/env python3
"""
YFinance Minute Data Fetcher - Downloads 5-min intraday data using yfinance
"""

from pathlib import Path
import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import Optional


class YFinanceMinuteDownloader:
    """Download 5-minute data using yfinance"""
    
    def __init__(self):
        # Cache directory
        self.cache_dir = Path('data/cache/yfinance_minute_data')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def download_minute_data(
        self, 
        ticker: str, 
        date: str, 
        timeframe: str = '5Min'
    ) -> Optional[pd.DataFrame]:
        """
        Download 5-minute data for a specific date using yfinance
        
        Args:
            ticker: Stock symbol
            date: Date string (YYYY-MM-DD) or datetime object
            timeframe: '5Min' (yfinance uses '5m')
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        # Convert date to string if datetime
        if isinstance(date, datetime):
            date_str = date.strftime('%Y-%m-%d')
        else:
            date_str = date
        
        # Check cache first
        cache_file = self.cache_dir / f"{ticker}_{date_str}_{timeframe}.parquet"
        if cache_file.exists():
            try:
                return pd.read_parquet(cache_file)
            except Exception as e:
                print(f"⚠️  Cache read failed for {ticker} {date_str}: {e}")
        
        # Convert timeframe format (5Min -> 5m)
        yf_timeframe = timeframe.replace('Min', 'm')
        
        try:
            # Download data for this date
            ticker_obj = yf.Ticker(ticker)
            
            # Get next day for end date
            from datetime import timedelta
            date_dt = datetime.strptime(date_str, '%Y-%m-%d')
            next_day = (date_dt + timedelta(days=1)).strftime('%Y-%m-%d')
            
            # Download
            df = ticker_obj.history(start=date_str, end=next_day, interval=yf_timeframe)
            
            if df is None or df.empty:
                # print(f"⚠️  No data returned for {ticker} on {date_str}")
                return None
            
            # Rename columns to match expected format
            df = df.rename(columns={
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            })
            
            # Keep only OHLCV columns
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Filter to only this date (yfinance might return extra data)
            df = df[df.index.date == date_dt.date()]
            
            if df.empty:
                return None
            
            # Cache the data
            df.to_parquet(cache_file)
            
            # print(f"✅ Downloaded {len(df)} bars for {ticker} on {date_str} ({timeframe})")
            return df
            
        except Exception as e:
            # print(f"❌ Error downloading {ticker} for {date_str}: {e}")
            return None
