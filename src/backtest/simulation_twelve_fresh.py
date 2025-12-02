#!/usr/bin/env python3
"""
Fresh October-November 2025 simulation using yfinance data
Single-threaded, no checkpoints, fully deterministic
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import joblib
import numpy as np
import random
from datetime import datetime, timedelta
import json
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
import os

# Import feature extractors
from ml_pipeline.consolidation_features import ConsolidationFeatureExtractor
from ml_pipeline.breakout_features import BreakoutFeatureExtractor
from data_processing.download_minute_data import TwelveDataMinuteDownloader


def td_get_intraday(ticker: str, date: str, downloader) -> pd.DataFrame:
    """Fetch 5-min intraday bars for a single date from Twelve Data."""
    df = downloader.download_minute_data(ticker, date, timeframe='5min')
    if df is None:
        return pd.DataFrame()
    cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
    return df[cols]


def td_get_daily(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily bars using yfinance (faster, cached)."""
    cache_dir = Path('data/cache/twelvedata_daily')
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{ticker}_{start_date}_{end_date}.parquet"
    
    if cache_file.exists():
        try:
            return pd.read_parquet(cache_file)
        except Exception:
            pass

    try:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
        end_plus = end_dt.strftime('%Y-%m-%d')
    except Exception:
        end_plus = end_date

    try:
        df = yf.download(ticker, start=start_date, end=end_plus, interval='1d', 
                        progress=False, auto_adjust=False)
    except Exception as e:
        print(f'   Error fetching daily data for {ticker}: {e}')
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns={'Adj Close': 'Close'})

    # Convert to numeric, handling MultiIndex column selections
    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if c in df.columns:
            col = df[c]
            if isinstance(col, pd.DataFrame):
                col = col.iloc[:, 0]
            df[c] = pd.to_numeric(col, errors='coerce')

    # Normalize timezone
    try:
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')
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


class FreshSimulator:
    """Single-threaded, deterministic simulator"""
    
    def __init__(self, starting_cash=30000, random_seed=42):
        # Set random seed
        np.random.seed(random_seed)
        random.seed(random_seed)
        
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
        self.td_downloader = TwelveDataMinuteDownloader()
        self.breakout_extractor = BreakoutFeatureExtractor(
            timeframe='5Min', 
            downloader=self.td_downloader
        )
        
        # Thresholds
        self.pipeline1_threshold = 0.70
        self.pipeline2_threshold = 0.17
        
        # Capital
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.risk_per_trade_pct = 0.5
        
        # Tracking
        self.trades = []
        self.positions = []
    
    def get_trading_days(self, start_date='2025-10-01', end_date='2025-10-02'):
        """Get all trading days between start and end"""
        trading_days = []
        current = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        while current <= end:
            if current.weekday() < 5:  # Skip weekends
                trading_days.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)
        
        return trading_days
    
    def scan_for_consolidations(self, tickers, date):
        """Run Pipeline 1 on all tickers - single-threaded for full determinism"""
        scan_date_dt = datetime.strptime(date, '%Y-%m-%d')
        watchlist = []
        
        # Sort tickers alphabetically for deterministic order
        tickers = sorted(tickers)
        
        print(f"   Scanning {len(tickers)} tickers for consolidations...")
        
        for i, ticker in enumerate(tickers):
            if (i + 1) % 20 == 0:
                print(f"   Progress: {i+1}/{len(tickers)}")
            
            try:
                features_dict = self.consol_extractor.extract_features(ticker, scan_date_dt)
                if not features_dict:
                    continue
                
                feature_values = [features_dict.get(f, 0) for f in self.pipeline1_features]
                X_df = pd.DataFrame([feature_values], columns=self.pipeline1_features)
                prob = self.pipeline1_model.predict_proba(X_df)[0][1]
                
                if prob >= self.pipeline1_threshold:
                    result = {
                        'ticker': ticker,
                        'prob': prob,
                        'consol_high': features_dict.get('price_high', 0),
                        'consol_low': features_dict.get('price_low', 0),
                        'range_pct': features_dict.get('price_range_pct', 0)
                    }
                    watchlist.append(result)
                    print(f"      [P1] {ticker}: {prob*100:.1f}% ✓")
            except Exception as e:
                print(f"      [P1] {ticker}: ERROR - {e}")
                continue
        
        print(f"\n   ✅ Pipeline 1: {len(watchlist)} stocks passed threshold (≥{self.pipeline1_threshold*100:.0f}%)")
        return watchlist
    
    def check_pipeline2(self, ticker, date, entry_price, stop_price, target_price):
        """Check if trade passes Pipeline 2"""
        try:
            features_dict = self.breakout_extractor.extract_features(
                ticker, date, entry_price, stop_price, target_price
            )
            if not features_dict:
                return False, 0.0
            
            feature_values = [features_dict.get(f, 0) for f in self.pipeline2_features]
            X_df = pd.DataFrame([feature_values], columns=self.pipeline2_features)
            prob = self.pipeline2_model.predict_proba(X_df)[0][1]
            
            return prob >= self.pipeline2_threshold, prob
        except Exception as e:
            return False, 0.0
    
    def manage_position(self, position, date, daily_data):
        """Manage a position using daily data only"""
        try:
            if daily_data is None or daily_data.empty:
                return None
            
            current_stop = position['stop_price']
            hit_5R = position['hit_5R']
            position_size = position['position_size']
            partial_exits = position.get('partial_exits', [])
            
            total_exit_price = sum(exit['price'] * exit['size'] for exit in partial_exits)
            
            # Get daily low and close
            low_col = daily_data['Low']
            close_col = daily_data['Close']
            
            if isinstance(low_col, pd.DataFrame):
                low_col = low_col.iloc[:, 0]
            if isinstance(close_col, pd.DataFrame):
                close_col = close_col.iloc[:, 0]
            
            today_low = float(low_col.iloc[-1])
            today_close = float(close_col.iloc[-1])
            
            # 1. Check if close >= 5R (take 20% profit)
            if not hit_5R and position_size == 1.0 and today_close >= position['target_price']:
                partial_exits.append({
                    'time': daily_data.index[-1],
                    'price': today_close,
                    'size': 0.20,
                    'reason': 'PARTIAL_5R'
                })
                total_exit_price += today_close * 0.20
                position_size = 0.80
                current_stop = position['entry_price']  # Move to breakeven
                hit_5R = True
            
            # 2. Check stop using daily low
            if today_low <= current_stop and position_size > 0:
                exit_price = current_stop * 0.995  # 0.5% slippage
                partial_exits.append({
                    'time': daily_data.index[-1],
                    'price': exit_price,
                    'size': position_size,
                    'reason': 'STOPPED_OUT'
                })
                total_exit_price += exit_price * position_size
                position_size = 0
            
            # 3. Check SMA10
            elif 'SMA10' in daily_data.columns and position_size > 0:
                sma_col = daily_data['SMA10']
                if isinstance(sma_col, pd.DataFrame):
                    sma_col = sma_col.iloc[:, 0]
                current_sma = float(sma_col.iloc[-1])
                
                if today_close < current_sma:
                    partial_exits.append({
                        'time': daily_data.index[-1],
                        'price': today_close,
                        'size': position_size,
                        'reason': 'BELOW_10SMA'
                    })
                    total_exit_price += today_close * position_size
                    position_size = 0
            
            # If fully exited
            if position_size == 0:
                return {
                    'exited': True,
                    'exit_price': total_exit_price,
                    'exit_time': partial_exits[-1]['time'],
                    'reason': partial_exits[-1]['reason'],
                    'partial_exits': partial_exits
                }
            
            # Otherwise hold
            return {
                'exited': False,
                'position_size': position_size,
                'stop_price': current_stop,
                'hit_5R': hit_5R,
                'partial_exits': partial_exits
            }
        
        except Exception as e:
            print(f"   Error managing position: {e}")
            return None
    
    def calculate_position_size(self, entry_price, stop_price):
        """Calculate shares based on risk % of starting capital"""
        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share == 0:
            return 0
        
        dollar_risk = self.starting_cash * (self.risk_per_trade_pct / 100.0)
        slippage_factor = 1.005
        effective_risk = risk_per_share * slippage_factor
        shares = int(dollar_risk / effective_risk)
        
        max_shares = int(self.cash / entry_price)
        shares = min(shares, max_shares)
        
        return shares if shares > 0 else 0
    
    def run_simulation(self, tickers):
        """Run the full simulation"""
        trading_days = self.get_trading_days()
        active_positions = []
        entered_tickers_this_month = set()
        
        print(f"\n{'='*80}")
        print(f"FRESH OCTOBER-NOVEMBER 2025 SIMULATION")
        print(f"{'='*80}")
        print(f"Tickers: {len(tickers)}")
        print(f"Trading days: {len(trading_days)}")
        print(f"Period: {trading_days[0]} to {trading_days[-1]}")
        print(f"Starting capital: ${self.starting_cash:,}")
        print(f"{'='*80}\n")
        
        for day_idx, date in enumerate(trading_days, 1):
            print(f"\n[Day {day_idx}/{len(trading_days)}] {date}")
            print("-" * 80)
            
            entered_tickers_today = set()
            
            # Manage existing positions
            if active_positions:
                print(f"\n   📊 Managing {len(active_positions)} open position(s)...")
                remaining_positions = []
                
                for position in active_positions:
                    ticker = position['ticker']
                    date_dt = datetime.strptime(date, '%Y-%m-%d')
                    start_date = (date_dt - timedelta(days=15)).strftime('%Y-%m-%d')
                    daily_data = td_get_daily(ticker, start_date, date)
                    
                    # Calculate SMA10
                    if len(daily_data) >= 11:
                        close_col = daily_data['Close']
                        if isinstance(close_col, pd.DataFrame):
                            close_col = close_col.iloc[:, 0]
                        daily_data['SMA10'] = close_col.rolling(10).mean()
                    
                    result = self.manage_position(position, date, daily_data)
                    
                    if result and result['exited']:
                        cost = position['shares'] * position['entry_price']
                        proceeds = position['shares'] * result['exit_price']
                        pnl = proceeds - cost
                        pnl_pct = (pnl / cost) * 100
                        self.cash += proceeds
                        
                        print(f"      ✅ {ticker}: ${position['entry_price']:.2f} → ${result['exit_price']:.2f}")
                        print(f"         P&L: ${pnl:,.0f} ({pnl_pct:+.1f}%) | {result['reason']}")
                        
                        self.trades.append({
                            'date': position['entry_date'],
                            'exit_date': date,
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
                        position.update({
                            'position_size': result['position_size'],
                            'stop_price': result['stop_price'],
                            'hit_5R': result['hit_5R'],
                            'partial_exits': result['partial_exits']
                        })
                        remaining_positions.append(position)
                        print(f"      📌 {ticker}: Holding (stop: ${result['stop_price']:.2f})")
                    else:
                        remaining_positions.append(position)
                
                active_positions = remaining_positions
            
            # Pipeline 1 scan
            watchlist = self.scan_for_consolidations(tickers, date)
            
            if not watchlist:
                print("   No consolidations found")
                continue
            
            # Sort by probability (highest first), then ticker for determinism
            watchlist = sorted(watchlist, key=lambda x: (-x['prob'], x['ticker']))
            
            print(f"   📋 Watchlist: {', '.join([w['ticker'] for w in watchlist[:10]])}" + 
                  (f" + {len(watchlist)-10} more" if len(watchlist) > 10 else ""))
            
            # Get previous day highs
            prev_day_data = {}
            for stock in watchlist:
                try:
                    ticker = stock['ticker']
                    date_dt = datetime.strptime(date, '%Y-%m-%d')
                    prev_date = (date_dt - timedelta(days=3)).strftime('%Y-%m-%d')
                    df_prev = td_get_daily(ticker, prev_date, date)
                    if df_prev is not None and len(df_prev) >= 1:
                        high_col = df_prev['High']
                        if isinstance(high_col, pd.DataFrame):
                            high_col = high_col.iloc[:, 0]
                        prev_day_data[ticker] = float(high_col.iloc[-1])
                except Exception:
                    pass
            
            # Scan for breakouts
            print(f"\n   🔍 Scanning watchlist for breakouts...")
            candidates = []
            max_trades_per_day = 3
            
            for stock in watchlist:
                ticker = stock['ticker']
                prev_day_high = prev_day_data.get(ticker, stock.get('consol_high'))
                
                # Get intraday data
                intraday_data = td_get_intraday(ticker, date, self.td_downloader)
                if intraday_data is None or len(intraday_data) == 0:
                    continue
                
                intraday_data = intraday_data.sort_index()
                
                hot = None
                breakout_info = None
                
                # Scan each bar
                for bar_idx, (timestamp, row) in enumerate(intraday_data.iterrows()):
                    high_so_far = intraday_data.loc[:timestamp, 'High'].max()
                    low_so_far = intraday_data.loc[:timestamp, 'Low'].min()
                    
                    hypothetical_entry = high_so_far * 1.005
                    hypothetical_stop = low_so_far
                    risk = hypothetical_entry - hypothetical_stop
                    
                    if risk <= 0:
                        continue
                    
                    hypothetical_target = hypothetical_entry + 5 * risk
                    
                    # Check Pipeline 2
                    passed, prob = self.check_pipeline2(
                        ticker, date, hypothetical_entry, hypothetical_stop, hypothetical_target
                    )
                    
                    if passed:
                        if hot is None or prob > hot['prob']:
                            hot = {
                                'prob': prob,
                                'time': timestamp,
                                'trigger': high_so_far,
                                'entry': hypothetical_entry,
                                'stop': hypothetical_stop,
                                'target': hypothetical_target
                            }
                    
                    # Check for breakout
                    if hot is not None and row['High'] > hot['trigger']:
                        current_high = intraday_data.loc[:timestamp, 'High'].max()
                        current_low = intraday_data.loc[:timestamp, 'Low'].min()
                        entry_price = current_high * 1.005
                        stop_price = current_low
                        target_price = entry_price + 5 * (entry_price - stop_price)
                        
                        breakout_info = {
                            'ticker': ticker,
                            'entry_time': timestamp,
                            'entry_price': entry_price,
                            'stop_price': stop_price,
                            'target_price': target_price,
                            'p2_prob': hot['prob']
                        }
                        break
                
                if breakout_info:
                    candidates.append(breakout_info)
            
            # Select top candidates
            if not candidates:
                print('   No breakouts found')
                continue
            
            candidates = sorted(candidates, key=lambda x: (-x['p2_prob'], x['ticker']))
            to_enter = candidates[:max_trades_per_day]
            
            for info in to_enter:
                ticker = info['ticker']
                
                if ticker in entered_tickers_today or ticker in entered_tickers_this_month:
                    continue
                
                shares = self.calculate_position_size(info['entry_price'], info['stop_price'])
                if shares == 0:
                    continue
                
                cost = shares * info['entry_price']
                if cost > self.cash:
                    continue
                
                position = {
                    'ticker': ticker,
                    'entry_date': date,
                    'entry_price': info['entry_price'],
                    'stop_price': info['stop_price'],
                    'target_price': info['target_price'],
                    'shares': shares,
                    'position_size': 1.0,
                    'hit_5R': False,
                    'p2_prob': info['p2_prob'],
                    'partial_exits': []
                }
                
                self.cash -= cost
                active_positions.append(position)
                entered_tickers_today.add(ticker)
                entered_tickers_this_month.add(ticker)
                
                print(f"\n   🚨 {ticker} at {info['entry_time'].strftime('%H:%M')}")
                print(f"      P2: {info['p2_prob']*100:.1f}% ✅")
                print(f"      ENTERED: ${info['entry_price']:.2f}")
                print(f"      Shares: {shares} | Stop: ${info['stop_price']:.2f} | Cash: ${self.cash:,.0f}")
        
        self.print_results()
    
    def print_results(self):
        """Print final results"""
        print(f"\n\n{'='*80}")
        print(f"📊 FINAL RESULTS")
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
                print(f"{trade['date']} {trade['ticker']:6} "
                      f"${trade['entry']:7.2f} → ${trade['exit']:7.2f} "
                      f"P&L: ${trade['pnl']:7.0f} ({trade['pnl_pct']:+.1f}%) | {trade['outcome']}")
        
        print(f"{'='*80}\n")


def main():
    """Run the fresh simulation"""
    # Load tickers
    tickers = []
    with open('data/processed/unique_traded_jun1_oct31_2025.txt', 'r') as f:
        for line in f:
            ticker = line.strip()
            if ticker:
                tickers.append(ticker)
    
    print(f"Loaded {len(tickers)} tickers")
    
    sim = FreshSimulator(starting_cash=30000, random_seed=42)
    sim.run_simulation(tickers)


if __name__ == '__main__':
    main()
