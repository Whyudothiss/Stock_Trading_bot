#!/usr/bin/env python3
"""
Minute Data Downloader - Downloads intraday historical data using Alpaca API
Supports both 1-minute and 5-minute timeframes
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
import time
import requests
from typing import Optional

# Load environment variables
load_dotenv()


class AlpacaMinuteDownloader:
    """Download minute-level historical data from Alpaca"""
    
    def __init__(self):
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        # Use the data API endpoint (not trading endpoint)
        self.base_url = 'https://data.alpaca.markets'
        
        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Alpaca API keys not found. Please:\n"
                "1. Sign up at https://alpaca.markets/\n"
                "2. Get your API keys from dashboard\n"
                "3. Copy .env.example to .env\n"
                "4. Add your keys to .env file"
            )
        
        self.headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.secret_key
        }
        
        # Cache directory
        self.cache_dir = Path('data/cache/minute_data')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def download_minute_data(
        self, 
        ticker: str, 
        date: str, 
        timeframe: str = '1Min'
    ) -> Optional[pd.DataFrame]:
        """
        Download minute data for a specific date
        
        Args:
            ticker: Stock symbol
            date: Date string (YYYY-MM-DD) or datetime object
            timeframe: '1Min' or '5Min'
            
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
        
        # Parse date
        try:
            trade_date = datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            print(f"❌ Invalid date format: {date_str}")
            return None
        
        # Set start and end times (market hours: 9:30 AM - 4:00 PM ET)
        start_time = trade_date.replace(hour=9, minute=30, second=0)
        end_time = trade_date.replace(hour=16, minute=0, second=0)
        
        # Format for API
        start_str = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_str = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        # Alpaca API endpoint
        url = f"{self.base_url}/v2/stocks/{ticker}/bars"
        
        params = {
            'start': start_str,
            'end': end_str,
            'timeframe': timeframe,
            'adjustment': 'split',
            'feed': 'sip',  # Use SIP feed for consolidated market data (all exchanges)
            'limit': 10000
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'bars' not in data or len(data['bars']) == 0:
                    print(f"⚠️  No data returned for {ticker} on {date_str}")
                    return None
                
                # Convert to DataFrame
                bars = data['bars']
                df = pd.DataFrame(bars)
                
                # Rename columns to match yfinance format
                df = df.rename(columns={
                    't': 'Datetime',
                    'o': 'Open',
                    'h': 'High',
                    'l': 'Low',
                    'c': 'Close',
                    'v': 'Volume'
                })
                
                # Convert timestamp to datetime
                df['Datetime'] = pd.to_datetime(df['Datetime'])
                df.set_index('Datetime', inplace=True)
                
                # Cache the data
                df.to_parquet(cache_file)
                
                print(f"✅ Downloaded {len(df)} bars for {ticker} on {date_str} ({timeframe})")
                return df
                
            elif response.status_code == 404:
                print(f"⚠️  {ticker} not found for {date_str}")
                return None
            elif response.status_code == 429:
                print(f"⚠️  Rate limit hit. Waiting 60 seconds...")
                time.sleep(60)
                return self.download_minute_data(ticker, date, timeframe)
            else:
                print(f"❌ API error {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"⚠️  Timeout downloading {ticker} for {date_str}")
            return None
        except Exception as e:
            print(f"❌ Error downloading {ticker} for {date_str}: {e}")
            return None
    
    def test_connection(self) -> bool:
        """Test API connection with a simple request"""
        print("🔍 Testing Alpaca API connection...")
        
        try:
            # Try to get recent bars for SPY - use a known trading day
            # Go back up to 7 days to ensure we hit a trading day
            for days_back in range(1, 8):
                test_date = (datetime.now() - timedelta(days=days_back))
                # Skip weekends
                if test_date.weekday() >= 5:  # 5=Saturday, 6=Sunday
                    continue
                test_date_str = test_date.strftime('%Y-%m-%d')
                break
            
            print(f"Testing with date: {test_date_str}")
            df = self.download_minute_data('SPY', test_date_str, '1Min')
            
            if df is not None and len(df) > 0:
                print(f"✅ Connection successful! Retrieved {len(df)} bars for SPY")
                return True
            else:
                print("❌ Connection test failed - no data returned")
                return False
                
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False


def main():
    """Test the downloader"""
    print("\n" + "="*60)
    print("ALPACA MINUTE DATA DOWNLOADER - TEST")
    print("="*60 + "\n")
    
    try:
        downloader = AlpacaMinuteDownloader()
        
        # Test connection
        if not downloader.test_connection():
            print("\n❌ Please check your API keys in .env file")
            return
        
        # Test downloading specific date
        print("\n" + "-"*60)
        print("Testing historical download...")
        print("-"*60 + "\n")
        
        test_ticker = "AAPL"
        test_date = "2024-11-14"
        
        # Test 1-minute data
        print(f"Downloading {test_ticker} 1-minute data for {test_date}...")
        df_1min = downloader.download_minute_data(test_ticker, test_date, '1Min')
        
        if df_1min is not None:
            print(f"\n1-Minute Data Sample:")
            print(df_1min.head())
            print(f"\nTotal bars: {len(df_1min)}")
        
        # Test 5-minute data
        print(f"\nDownloading {test_ticker} 5-minute data for {test_date}...")
        df_5min = downloader.download_minute_data(test_ticker, test_date, '5Min')
        
        if df_5min is not None:
            print(f"\n5-Minute Data Sample:")
            print(df_5min.head())
            print(f"\nTotal bars: {len(df_5min)}")
        
        print("\n✅ Test completed successfully!")
        
    except ValueError as e:
        print(f"\n❌ Configuration error: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
