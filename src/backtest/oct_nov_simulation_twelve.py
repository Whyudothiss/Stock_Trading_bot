#!/usr/bin/env python3
"""
October-November 2025 SIMULATION using yfinance data
Simulates live trading from Oct 1 - Nov 17, 2025
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import joblib
import numpy as np
from datetime import datetime, timedelta
import json
import os
import pandas as pd
import requests
from dotenv import load_dotenv
import yfinance as yf
import argparse

# Import feature extractors
from ml_pipeline.consolidation_features import ConsolidationFeatureExtractor
from ml_pipeline.breakout_features import BreakoutFeatureExtractor
from data_processing.download_minute_data import TwelveDataMinuteDownloader
from pathlib import Path
import threading

# Module-level TwelveData downloader singleton (uses cache + rate limiting)
_TD_DOWNLOADER_LOCK = threading.Lock()
_TD_DOWNLOADER = None

def _get_td_downloader():
    global _TD_DOWNLOADER
    with _TD_DOWNLOADER_LOCK:
        if _TD_DOWNLOADER is None:
            _TD_DOWNLOADER = TwelveDataMinuteDownloader()
        return _TD_DOWNLOADER


def td_get_intraday(ticker: str, date: str, api_key: str = None) -> pd.DataFrame:
    """Fetch 5-min intraday bars for a single date from Twelve Data.

    Returns DataFrame indexed by timezone-aware datetime (America/New_York) with columns Open/High/Low/Close/Volume.
    """
    # Use the shared downloader which handles rate limiting and caching
    downloader = _get_td_downloader()
    # downloader.download_minute_data returns a DataFrame or None
    df = downloader.download_minute_data(ticker, date, timeframe='5min')
    if df is None:
        return pd.DataFrame()
    # Ensure canonical column names
    cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
    return df[cols]


def td_get_daily(ticker: str, start_date: str, end_date: str, api_key: str = None) -> pd.DataFrame:
    """Fetch daily bars from Twelve Data between start_date and end_date (inclusive).

    Returns DataFrame with DateTime index (America/New_York) and Open/High/Low/Close/Volume.
    """
    # Use yfinance for daily bars (lighter, avoids Twelve Data daily quota)
    cache_dir = Path('data/cache/twelvedata_daily')
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{ticker}_{start_date}_{end_date}.parquet"
    if cache_file.exists():
        try:
            return pd.read_parquet(cache_file)
        except Exception:
            pass

    # yfinance end date is exclusive, so add one day to include `end_date`
    try:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
        end_plus = end_dt.strftime('%Y-%m-%d')
    except Exception:
        # fall back to using provided strings
        end_plus = end_date

    try:
        df = yf.download(ticker, start=start_date, end=end_plus, interval='1d', progress=False, auto_adjust=False)
    except Exception as e:
        print(f'Error fetching daily data from yfinance for {ticker}: {e}')
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # Ensure canonical column names
    # yfinance may return a MultiIndex columns (e.g., when multiple tickers are requested)
    # Flatten to single level if needed so column selection returns Series, not DataFrame
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.get_level_values(0)
        except Exception:
            # Fall back to stringifying column tuples
            df.columns = ['_'.join(map(str, c)) for c in df.columns]

    df = df.rename(columns={
        'Adj Close': 'Close'
    })

    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if c in df.columns:
            # Ensure we operate on a 1D Series; if df[c] is a DataFrame, pick the first column
            col = df[c]
            if hasattr(col, 'shape') and getattr(col, 'ndim', 1) > 1:
                # convert to first column
                try:
                    col = col.iloc[:, 0]
                except Exception:
                    # last resort: skip numeric conversion for this column
                    continue
            df[c] = pd.to_numeric(col, errors='coerce')

    # Normalize timezone to America/New_York
    try:
        if df.index.tz is None:
            # yfinance often returns UTC-naive dates; localize to UTC then convert
            try:
                df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')
            except Exception:
                df.index = df.index.tz_localize('America/New_York')
        else:
            df.index = df.index.tz_convert('America/New_York')
    except Exception:
        pass

    cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
    out = df[cols].copy()
    try:
        out.to_parquet(cache_file)
    except Exception:
        pass
    return out


class OctNovSimulator:
    """Simulate Oct-Nov 2025 with yfinance data"""
    
    def __init__(self, starting_cash=30000):
        # Load models
        print("Loading models...")
        pipeline1_data = joblib.load('models/consolidation_detector.pkl')
        pipeline2_model = joblib.load('models/5R_predictor_5min_twelve.pkl')
        
        self.pipeline1_model = pipeline1_data['model']
        self.pipeline1_features = pipeline1_data['feature_names']
        
        self.pipeline2_model = pipeline2_model
        with open('models/5R_predictor_5min_features.txt', 'r') as f:
            self.pipeline2_features = [line.strip() for line in f]
        
        # Feature extractors
        self.consol_extractor = ConsolidationFeatureExtractor()
        # Use the shared Twelve Data downloader (singleton) so we respect caching and rate limits
        self.breakout_extractor = BreakoutFeatureExtractor(timeframe='5Min', downloader=_get_td_downloader())
        
        # Twelve Data API key (loaded lazily by helper functions)
        load_dotenv()
        self.td_api_key = os.getenv('TWELVEDATA_API_KEY')
        
        # Thresholds
        self.pipeline1_threshold = 0.70
        self.pipeline2_threshold = 0.17
        
        # Capital management
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.risk_per_trade_pct = 0.5
        
        # Track trades
        self.trades = []
        self.positions = []
        # Checkpointing
        self.checkpoint_path = Path('data/cache/octnov_sim_checkpoint.pkl')
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.last_completed_day = None
    
    def get_trading_days(self, start_date='2025-10-01', end_date='2025-10-31'):
        """Get all trading days between start and end"""
        trading_days = []
        current = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        while current <= end:
            # Skip weekends
            if current.weekday() < 5:  # Monday=0, Friday=4
                trading_days.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)
        
        return trading_days
    
    def scan_for_consolidations(self, tickers, date):
        """Run Pipeline 1 on all tickers for this date (parallelized)"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        watchlist = []
        scan_date_dt = datetime.strptime(date, '%Y-%m-%d')

        def process_ticker(ticker):
            try:
                features_dict = self.consol_extractor.extract_features(ticker, scan_date_dt)
                if not features_dict:
                    return None
                feature_values = [features_dict.get(f, 0) for f in self.pipeline1_features]
                # Build a DataFrame with column names to match training input
                try:
                    X_df = pd.DataFrame([feature_values], columns=self.pipeline1_features)
                    prob = self.pipeline1_model.predict_proba(X_df)[0][1]
                except Exception:
                    # Fallback to numpy array if DataFrame fails
                    X = np.array([feature_values])
                    prob = self.pipeline1_model.predict_proba(X)[0][1]
                if prob >= self.pipeline1_threshold:
                    return {
                        'ticker': ticker,
                        'prob': prob,
                        'consol_high': features_dict.get('price_high', 0),
                        'consol_low': features_dict.get('price_low', 0),
                        'range_pct': features_dict.get('price_range_pct', 0)
                    }
            except Exception:
                return None

        # Use up to 8 threads (tune as needed for your CPU)
        max_workers = min(8, len(tickers))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_ticker, ticker): ticker for ticker in tickers}
            for i, future in enumerate(as_completed(futures)):
                if (i + 1) % 20 == 0:
                    print(f"   Progress: {i+1}/{len(tickers)}")
                result = future.result()
                if result:
                    watchlist.append(result)
        return watchlist
        
        print(f"✅ Found {len(watchlist)} consolidations (>={self.pipeline1_threshold*100:.0f}%)")
        return watchlist
    
    def check_for_breakout(self, ticker, date, consol_high, consol_low, prev_day_high):
        """Check if ticker broke out using yfinance data"""
        try:
            data = td_get_intraday(ticker, date, api_key=self.td_api_key)
            
            if data is None or data.empty:
                return None
            
            # Check if gapping up
            open_price = data['Open'].iloc[0]
            gap_pct = ((open_price - prev_day_high) / prev_day_high) * 100 if prev_day_high > 0 else 0
            is_gap = gap_pct > 2.0
            
            # For gaps, get first candle high (9:30-9:35)
            if is_gap and len(data) > 0:
                first_candle_high = data['High'].iloc[0]
            
            # Check each 5-min bar for breakout
            for i, (idx, row) in enumerate(data.iterrows()):
                # Skip first candle if gap (wait for 9:35)
                if is_gap and i == 0:
                    continue
                
                # Determine breakout condition
                if is_gap:
                    # Must break first candle high
                    breakout = row['High'] > first_candle_high
                else:
                    # Must break previous day high
                    breakout = row['High'] > prev_day_high
                
                if breakout:
                    # Entry = trigger level + 0.5% slippage
                    if is_gap:
                        trigger_level = first_candle_high
                    else:
                        trigger_level = prev_day_high
                    entry_price = trigger_level * 1.005
                    
                    # Stop = entry day low (lowest low on entry day so far)
                    entry_day_low = data.loc[:idx]['Low'].min()
                    stop_price = entry_day_low
                    
                    # Target = 5R
                    risk = entry_price - stop_price
                    target_price = entry_price + 5 * risk
                    
                    if risk <= 0:
                        continue
                    
                    return {
                        'breakout_time': idx,
                        'breakout_bar_index': data.index.get_loc(idx),
                        'entry_price': entry_price,
                        'stop_price': stop_price,
                        'entry_day_low': entry_day_low,
                        'target_price': target_price,
                        'risk_per_share': risk,
                        'is_gap': is_gap,
                        'gap_pct': gap_pct,
                        'data': data
                    }
            
            return None
            
        except Exception as e:
            return None
    
    def check_pipeline2(self, ticker, date, entry_price, stop_price, target_price):
        """Check if trade passes Pipeline 2"""
        try:
            features_dict = self.breakout_extractor.extract_features(
                ticker, date, entry_price, stop_price, target_price
            )
            if not features_dict:
                return False, 0.0
            
            feature_values = [features_dict.get(f, 0) for f in self.pipeline2_features]
            try:
                X_df = pd.DataFrame([feature_values], columns=self.pipeline2_features)
                prob = self.pipeline2_model.predict_proba(X_df)[0][1]
            except Exception:
                X = np.array([feature_values])
                prob = self.pipeline2_model.predict_proba(X)[0][1]
            
            return prob >= self.pipeline2_threshold, prob
            
        except Exception as e:
            return False, 0.0
    
    def manage_position_intraday(self, position, date, intraday_data, daily_data):
        """Manage a position for one trading day"""
        try:
            if intraday_data is None or intraday_data.empty:
                return None  # Hold position
            
            current_stop = position['stop_price']
            hit_5R = position['hit_5R']
            position_size = position['position_size']
            partial_exits = position.get('partial_exits', [])  # Carry forward from previous days
            
            # Recalculate total_exit_price from all previous partial exits
            total_exit_price = sum(exit['price'] * exit['size'] for exit in partial_exits)
            
            # Check each 5-min bar
            for idx, row in intraday_data.iterrows():
                # Check if hit 5R target
                if not hit_5R and position_size == 1.0 and row['High'] >= position['target_price']:
                    # Take 20% profit at 5R
                    partial_exits.append({
                        'time': idx,
                        'price': position['target_price'],
                        'size': 0.20,
                        'reason': 'PARTIAL_5R'
                    })
                    total_exit_price += position['target_price'] * 0.20
                    position_size = 0.80
                    # Move stop to breakeven (entry price)
                    current_stop = position['entry_price']
                    hit_5R = True
                    continue
                
                # Check stop loss (with 0.5% slippage)
                if row['Low'] <= current_stop:
                    exit_price_with_slippage = current_stop * 0.995
                    partial_exits.append({
                        'time': idx,
                        'price': exit_price_with_slippage,
                        'size': position_size,
                        'reason': 'STOPPED_OUT'
                    })
                    total_exit_price += exit_price_with_slippage * position_size
                    position_size = 0
                    break
                
                # Check 3:50 PM 10 SMA exit
                time_str = idx.strftime('%H:%M')
                if time_str == '15:50' and position_size > 0:
                    # Check if 3:50 PM close < 10 SMA
                    if len(daily_data) >= 11 and 'SMA10' in daily_data.columns:
                        current_sma = daily_data['SMA10'].iloc[-1]
                        if pd.notna(current_sma) and row['Close'] < current_sma:
                            # Exit at next candle (3:55 PM) open
                            next_candles = intraday_data.loc[idx:].iloc[1:]
                            if len(next_candles) > 0:
                                next_candle = next_candles.iloc[0]
                                exit_price = next_candle['Open']
                                partial_exits.append({
                                    'time': next_candles.index[0],
                                    'price': exit_price,
                                    'size': position_size,
                                    'reason': 'BELOW_10SMA'
                                })
                                total_exit_price += exit_price * position_size
                                position_size = 0
                                break
            
            # If fully exited
            if position_size == 0:
                # total_exit_price is already the weighted average (sum of price * size)
                # where size fractions sum to 1.0
                avg_exit_price = total_exit_price
                return {
                    'exited': True,
                    'exit_price': avg_exit_price,
                    'exit_time': partial_exits[-1]['time'],
                    'reason': partial_exits[-1]['reason'],
                    'partial_exits': partial_exits
                }
            
            # Update position for next day
            return {
                'exited': False,
                'position_size': position_size,
                'stop_price': current_stop,
                'hit_5R': hit_5R,
                'partial_exits': partial_exits
            }
            
        except Exception as e:
            return None
    
    def calculate_position_size(self, entry_price, stop_price):
        """Calculate shares based on risk % of STARTING capital"""
        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share == 0:
            return 0
        
        # Risk should be based on STARTING capital, not remaining cash
        dollar_risk = self.starting_cash * (self.risk_per_trade_pct / 100.0)
        shares = int(dollar_risk / risk_per_share)
        
        # Cap by available cash WITHOUT changing risk-based sizing
        max_shares = int(self.cash / entry_price)
        shares = min(shares, max_shares)
        
        # If we don't have enough cash for even the risk-based position, skip trade
        if shares < int(dollar_risk / risk_per_share) * 0.5:  # Less than 50% of intended size
            return 0
        
        return max(shares, 0)
    
    def run_simulation(self, tickers, start_from_date: str = None):
        """Run Oct-Nov 2025 simulation"""
        trading_days = self.get_trading_days()
        # If resuming from a particular date (exclusive), slice trading_days accordingly
        if start_from_date:
            try:
                # accept string dates like '2025-08-01'
                start_from_dt = datetime.strptime(start_from_date, '%Y-%m-%d')
                # find first trading day strictly after start_from_date
                idx = 0
                for i, d in enumerate(trading_days):
                    d_dt = datetime.strptime(d, '%Y-%m-%d')
                    if d_dt > start_from_dt:
                        idx = i
                        break
                trading_days = trading_days[idx:]
            except Exception:
                pass
        active_positions = []
        entered_tickers_this_month = set()  # Track tickers entered during the entire month

        print(f"\n{'='*80}")
        print(f"JANUARY-NOVEMBER 2025 SIMULATION (yfinance data)")
        print(f"{'='*80}")
        print(f"Tickers: {len(tickers)}")
        print(f"Trading days: {len(trading_days)}")
        print(f"Period: {trading_days[0]} to {trading_days[-1]}")
        print(f"Starting capital: ${self.starting_cash:,}")
        print(f"{'='*80}\n")
        
        for day_idx, date in enumerate(trading_days, 1):
            print(f"\n[Day {day_idx}/{len(trading_days)}] {date}")
            print("-" * 80)
            
            # Track tickers entered today to prevent same-day re-entry
            entered_tickers_today = set()
            
            # Check existing positions first
            if active_positions:
                print(f"\n   📊 Managing {len(active_positions)} open position(s)...")
                remaining_positions = []
                
                for position in active_positions:
                    ticker = position['ticker']
                    
                    # Check daily 10 SMA FIRST (before intraday) using Twelve Data daily bars
                    date_dt = datetime.strptime(date, '%Y-%m-%d')
                    start_date = (date_dt - timedelta(days=15)).strftime('%Y-%m-%d')
                    daily_data = td_get_daily(ticker, start_date, date, api_key=self.td_api_key)
                    
                    # Check if yesterday closed below 10 SMA
                    if len(daily_data) >= 11:
                        daily_data['SMA10'] = daily_data['Close'].rolling(10).mean()
                        # Yesterday is the last bar in daily_data (download end date excludes today)
                        yesterday_close = daily_data['Close'].iloc[-1].item()
                        yesterday_sma = daily_data['SMA10'].iloc[-1].item()
                        
                        if pd.notna(yesterday_sma) and yesterday_close < yesterday_sma:
                            # Exit at today's open
                            intraday_data = td_get_intraday(ticker, date, api_key=self.td_api_key)
                            if intraday_data is not None and len(intraday_data) > 0:
                                exit_price = intraday_data['Open'].iloc[0]
                                cost = position['shares'] * position['entry_price']
                                proceeds = position['shares'] * exit_price
                                pnl = proceeds - cost
                                pnl_pct = (pnl / cost) * 100
                                self.cash += pnl
                                
                                print(f"      ✅ {ticker}: ${position['entry_price']:.2f} → ${exit_price:.2f}")
                                print(f"         P&L: ${pnl:,.0f} ({pnl_pct:+.1f}%) | BELOW_10SMA")
                                
                                self.trades.append({
                                    'date': position['entry_date'],
                                    'time': position['entry_time'],
                                    'exit_date': date,
                                    'exit_time': '09:30',
                                    'ticker': ticker,
                                    'entry': position['entry_price'],
                                    'exit': exit_price,
                                    'shares': position['shares'],
                                    'pnl': pnl,
                                    'pnl_pct': pnl_pct,
                                    'outcome': 'BELOW_10SMA',
                                    'p2_prob': position['p2_prob']
                                })
                                continue
                    
                    # Get intraday data for regular position management
                    intraday_data = td_get_intraday(ticker, date, api_key=self.td_api_key)
                    
                    result = self.manage_position_intraday(position, date, intraday_data, daily_data)
                    
                    if result and result['exited']:
                        # Position closed
                        cost = position['shares'] * position['entry_price']
                        proceeds = position['shares'] * result['exit_price']
                        pnl = proceeds - cost
                        pnl_pct = (pnl / cost) * 100
                        self.cash += pnl
                        
                        print(f"      ✅ {ticker}: ${position['entry_price']:.2f} → ${result['exit_price']:.2f}")
                        print(f"         P&L: ${pnl:,.0f} ({pnl_pct:+.1f}%) | {result['reason']}")
                        
                        self.trades.append({
                            'date': position['entry_date'],
                            'time': position['entry_time'],
                            'exit_date': date,
                            'exit_time': result['exit_time'].strftime('%H:%M'),
                            'ticker': ticker,
                            'entry': position['entry_price'],
                            'exit': result['exit_price'],
                            'shares': position['shares'],
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                            'outcome': result['reason'],
                            'p2_prob': position['p2_prob']
                        })
                    elif result and not result['exited']:
                        # Update position and hold
                        position['position_size'] = result['position_size']
                        position['stop_price'] = result['stop_price']
                        position['hit_5R'] = result['hit_5R']
                        remaining_positions.append(position)
                        print(f"      📌 {ticker}: Holding (stop: ${result['stop_price']:.2f})")
                    else:
                        # Keep position unchanged
                        remaining_positions.append(position)
                
                active_positions = remaining_positions
            
            # Pipeline 1 scan
            watchlist = self.scan_for_consolidations(tickers, date)
            
            if not watchlist:
                print("   No consolidations found")
                continue
            
            print(f"   📋 Watchlist: {', '.join([w['ticker'] for w in watchlist[:10]])}" + 
                  (f" + {len(watchlist)-10} more" if len(watchlist) > 10 else ""))
            
            # Get previous day highs (using Twelve Data daily bars)
            prev_day_data = {}
            for stock in watchlist:
                try:
                    ticker = stock['ticker']
                    date_dt = datetime.strptime(date, '%Y-%m-%d')
                    prev_date = (date_dt - timedelta(days=3)).strftime('%Y-%m-%d')
                    df_prev = td_get_daily(ticker, prev_date, date, api_key=self.td_api_key)
                    if df_prev is not None and len(df_prev) >= 1:
                        high_val = df_prev['High'].iloc[-1]
                        prev_day_data[ticker] = float(high_val)
                except Exception:
                    pass
            
            # NEW APPROACH: Continuously evaluate Pipeline 2 every 5 minutes
            print(f"\n   🔍 Continuously evaluating watchlist stocks...")
            
            trades_entered_today = 0
            max_trades_per_day = 3
            hot_list = {}  # ticker -> {last_prob, last_check_time}
            
            # Sort watchlist by Pipeline 1 probability (highest first)
            # This ensures we prioritize the best setups when we hit the 3-trade limit
            watchlist = sorted(watchlist, key=lambda x: x['prob'], reverse=True)
            
            # Get intraday data for all watchlist stocks
            for stock in watchlist:
                ticker = stock['ticker']
                prev_day_high = prev_day_data.get(ticker, stock['consol_high'])
                
                # Download 5-min data for this day (Twelve Data)
                intraday_data = td_get_intraday(ticker, date, api_key=self.td_api_key)
                if intraday_data is None or len(intraday_data) == 0:
                    continue
                
                # Iterate through each 5-min bar
                for bar_idx, (timestamp, row) in enumerate(intraday_data.iterrows()):
                    if trades_entered_today >= max_trades_per_day:
                        break
                    
                    # Calculate hypothetical entry as high-of-day so far
                    high_so_far = intraday_data.loc[:timestamp, 'High'].max()
                    # Stop loss: low from start of day up to current bar (entry bar)
                    low_so_far = intraday_data.loc[:timestamp, 'Low'].min()
                    
                    # Hypothetical entry = high_so_far + 0.5% slippage
                    hypothetical_entry = high_so_far * 1.005
                    hypothetical_stop = low_so_far
                    risk = hypothetical_entry - hypothetical_stop
                    
                    if risk <= 0:
                        continue
                    
                    hypothetical_target = hypothetical_entry + 5 * risk
                    
                    # Evaluate Pipeline 2 with hypothetical prices
                    passed, prob = self.check_pipeline2(
                        ticker, date,
                        hypothetical_entry,
                        hypothetical_stop,
                        hypothetical_target
                    )
                    
                    # Update hot list
                    if passed and prob >= self.pipeline2_threshold:
                        hot_list[ticker] = {
                            'prob': prob,
                            'time': timestamp,
                            'trigger': high_so_far,
                            'entry': hypothetical_entry,
                            'stop': hypothetical_stop,
                            'target': hypothetical_target
                        }
                    
                    # Check if actual breakout happened (price broke trigger)
                    if ticker in hot_list and row['High'] > hot_list[ticker]['trigger']:
                        # Check if already entered this ticker today or this month
                        if ticker in entered_tickers_today:
                            continue  # Skip - already entered today
                        if ticker in entered_tickers_this_month:
                            continue  # Skip - already entered this month
                        
                        # BREAKOUT! Recalculate entry/stop with current candle included
                        current_high = intraday_data.loc[:timestamp, 'High'].max()
                        current_low = intraday_data.loc[:timestamp, 'Low'].min()
                        entry_price = current_high * 1.005
                        stop_price = current_low
                        risk = entry_price - stop_price
                        target_price = entry_price + 5 * risk
                        prob = hot_list[ticker]['prob']
                        
                        shares = self.calculate_position_size(entry_price, stop_price)
                        if shares == 0:
                            continue
                        
                        # Create position
                        position = {
                            'ticker': ticker,
                            'entry_date': date,
                            'entry_time': timestamp.strftime('%H:%M'),
                            'entry_price': entry_price,
                            'stop_price': stop_price,
                            'target_price': target_price,
                            'shares': shares,
                            'position_size': 1.0,
                            'hit_5R': False,
                            'p2_prob': prob,
                            'partial_exits': []  # Track all exits across days
                        }
                        
                        active_positions.append(position)
                        # Don't subtract cash on entry - we're not actually spending it yet
                        # Cash will be updated when position closes based on P&L
                        trades_entered_today += 1
                        
                        # Mark ticker as entered
                        entered_tickers_today.add(ticker)
                        entered_tickers_this_month.add(ticker)
                        
                        print(f"\n   🚨 {ticker} at {timestamp.strftime('%H:%M')}")
                        print(f"      P2 (pre-screened): {prob*100:.1f}% ✅")
                        print(f"      ✅ ENTERED: ${entry_price:.2f}")
                        print(f"      Shares: {shares} | Stop: ${stop_price:.2f} | Cash: ${self.cash:,.0f}")
                        
                        # Remove from hot list after entering
                        del hot_list[ticker]
                        break  # Move to next stock after entering
        
        # Final results
            # Save checkpoint at end of each day so we can resume if interrupted
            try:
                self.last_completed_day = date
                joblib.dump(self, self.checkpoint_path)
                print(f"   [Checkpoint] Saved checkpoint for {date} -> {self.checkpoint_path}")
            except Exception as e:
                print(f"⚠️ Failed to write checkpoint: {e}")

        self.print_results()
    
    def print_results(self):
        """Print final results"""
        print(f"\n\n{'='*80}")
        print(f"📊 JANUARY-NOVEMBER 2025 RESULTS")
        print(f"{'='*80}")
        print(f"Starting Capital: ${self.starting_cash:,}")
        print(f"Final Capital: ${self.cash:,.0f}")
        
        total_return = self.cash - self.starting_cash
        return_pct = (total_return / self.starting_cash) * 100
        print(f"Total Return: ${total_return:,.0f} ({return_pct:+.2f}%)")
        
        print(f"\nTrades Taken: {len(self.trades)}")
        
        if len(self.trades) > 0:
            winners = [t for t in self.trades if t['pnl'] > 0]
            losers = [t for t in self.trades if t['pnl'] < 0]
            
            print(f"Winners: {len(winners)} ({len(winners)/len(self.trades)*100:.1f}%)")
            print(f"Losers: {len(losers)} ({len(losers)/len(self.trades)*100:.1f}%)")
            
            print(f"\n📋 All Trades:")
            print("-" * 80)
            for trade in self.trades:
                exit_date_str = trade.get('exit_date', trade['date'])
                print(f"{trade['date']} {trade['time']} {trade['ticker']:6} "
                      f"${trade['entry']:7.2f} → ${trade['exit']:7.2f} (exit: {exit_date_str}) "
                      f"P&L: ${trade['pnl']:7.0f} ({trade['pnl_pct']:+.1f}%) | {trade.get('outcome', 'N/A')}")
        
        print(f"{'='*80}\n")


def main():
    """Run the simulation with optional resume support"""
    parser = argparse.ArgumentParser(description='Run Oct-Nov simulator')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint if present')
    parser.add_argument('--reset-checkpoint', action='store_true', help='Delete existing checkpoint and start fresh')
    args = parser.parse_args()

    # Load all tickers with trades since June 2025
    tickers = []
    with open('data/processed/unique_traded_jun1_oct31_2025.txt', 'r') as f:
        for line in f:
            ticker = line.strip()
            if ticker:
                tickers.append(ticker)

    print(f"Loaded {len(tickers)} tickers with trades since January 2025")
    print("Priority sorting enabled: highest Pipeline 1 probabilities evaluated first")

    checkpoint_path = Path('data/cache/octnov_sim_checkpoint.pkl')
    if args.reset_checkpoint and checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
            print(f"Removed checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"Could not remove checkpoint: {e}")

    if args.resume and checkpoint_path.exists():
        try:
            print(f"Resuming from checkpoint: {checkpoint_path}")
            sim = joblib.load(checkpoint_path)
            # Determine the resume start date (start after last_completed_day)
            start_from = getattr(sim, 'last_completed_day', None)
            if start_from is None:
                print("Checkpoint present but last completed day not recorded; resuming from beginning")
                sim.run_simulation(tickers)
            else:
                print(f"Resuming simulation from the day after {start_from}")
                sim.run_simulation(tickers, start_from_date=start_from)
            return
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")

    # No resume: start fresh
    sim = OctNovSimulator(starting_cash=30000)
    sim.run_simulation(tickers)


if __name__ == '__main__':
    main()
