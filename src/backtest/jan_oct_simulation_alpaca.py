#!/usr/bin/env python3
"""
January-October 2025 SIMULATION using Alpaca API data
Simulates live trading from Jan 1 - Oct 31, 2025, using direct HTTP requests.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import joblib
import numpy as np
import pickle
from datetime import datetime, timedelta
import pandas as pd
import requests
from dotenv import load_dotenv
import os

# Import feature extractors
from ml_pipeline.consolidation_features_alpaca import ConsolidationFeatureExtractorAlpaca
from ml_pipeline.breakout_features_alpaca import BreakoutFeatureExtractorAlpaca

class JanOctSimulatorAlpaca:
    """Simulate Jan-Oct 2025 with Alpaca API data"""

    CHECKPOINT_PATH = "../../data/simulation/alpaca_sim_checkpoint.pkl"
    TRADES_PATH = "../../data/simulation/alpaca_sim_trades.csv"

    def save_checkpoint(self, day_idx, date, active_positions):
        os.makedirs(os.path.dirname(self.CHECKPOINT_PATH), exist_ok=True)
        checkpoint = {
            'day_idx': day_idx,
            'date': date,
            'cash': self.cash,
            'active_positions': active_positions,
            'trades': self.trades
        }
        with open(self.CHECKPOINT_PATH, 'wb') as f:
            pickle.dump(checkpoint, f)

    def load_checkpoint(self, ignore_checkpoint=False):
        if not ignore_checkpoint and os.path.exists(self.CHECKPOINT_PATH):
            with open(self.CHECKPOINT_PATH, 'rb') as f:
                checkpoint = pickle.load(f)
            self.cash = checkpoint['cash']
            self.trades = checkpoint['trades']
            return checkpoint['day_idx'], checkpoint['date'], checkpoint['active_positions']
        return 0, None, []

    def save_trades(self):
        os.makedirs(os.path.dirname(self.TRADES_PATH), exist_ok=True)
        pd.DataFrame(self.trades).to_csv(self.TRADES_PATH, index=False)

    def __init__(self, starting_cash=30000):
        # Load models
        print("Loading models...")
        pipeline1_data = joblib.load('models/consolidation_detector.pkl')
        pipeline2_model = joblib.load('models/5R_predictor_5min.pkl')
        self.pipeline1_model = pipeline1_data['model']
        self.pipeline1_features = pipeline1_data['feature_names']
        self.pipeline2_model = pipeline2_model
        with open('models/5R_predictor_5min_features.txt', 'r') as f:
            self.pipeline2_features = [line.strip() for line in f]
        # Feature extractors
        self.consol_extractor = ConsolidationFeatureExtractorAlpaca()
        self.breakout_extractor = BreakoutFeatureExtractorAlpaca(timeframe='5Min')
        
        # Capital management & thresholds
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.risk_per_trade_pct = 0.5
        self.pipeline1_threshold = 0.70
        self.pipeline2_threshold = 0.45
        
        self.trades = []

    def get_trading_days(self):
        # Use pandas_market_calendars to get NYSE trading days (skip weekends and holidays)
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar('NYSE')
        sched = nyse.schedule(start_date='2025-01-01', end_date='2025-10-31')
        return [d.strftime('%Y-%m-%d') for d in sched.index]

    def scan_for_consolidations(self, tickers, scan_date):
        """Run Pipeline 1 to find good consolidations (parallelized, with Alpaca rate limit handling)"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        scan_date_dt = datetime.strptime(scan_date, '%Y-%m-%d')
        watchlist = []

        def process_ticker(ticker):
            try:
                features_dict = self.consol_extractor.extract_features(ticker, scan_date_dt)
                if not features_dict:
                    return None
                feature_values = [features_dict.get(f, 0) for f in self.pipeline1_features]
                X = np.array([feature_values])
                prob = self.pipeline1_model.predict_proba(X)[0][1]
                if prob >= 0.7:
                    return {
                        'ticker': ticker,
                        'prob': prob,
                        'consol_high': features_dict.get('price_high', 0),
                        'consol_low': features_dict.get('price_low', 0),
                        'range_pct': features_dict.get('price_range_pct', 0)
                    }
            except requests.exceptions.HTTPError as e:
                if e.response and e.response.status_code == 429:
                    print("[Alpaca] Rate limit hit. Waiting 60 seconds...")
                    import time; time.sleep(60)
                    return process_ticker(ticker)
                else: # Other HTTP errors
                    return None
            except Exception:
                return None

        max_workers = min(8, len(tickers))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_ticker, ticker): ticker for ticker in tickers}
            for i, future in enumerate(as_completed(futures)):
                if (i + 1) % 20 == 0:
                    print(f"   Progress: {i+1}/{len(tickers)}")
                result = future.result()
                if result:
                    watchlist.append(result)
        print(f"✅ Found {len(watchlist)} consolidations (>=70%)")
        return watchlist

    def download_minute_data_http(self, ticker: str, date_str: str, timeframe: str = '5Min') -> pd.DataFrame:
        """Fetches minute data from Alpaca using direct HTTP requests with rate limit handling."""
        load_dotenv()
        api_key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            print("❌ Alpaca API keys not found for minute data download.")
            return pd.DataFrame()

        url = f"https://data.alpaca.markets/v2/stocks/{ticker}/bars"
        start_time = f"{date_str}T09:30:00Z"
        end_time = f"{date_str}T16:00:00Z"

        params = {"start": start_time, "end": end_time, "timeframe": timeframe, "adjustment": "all", "limit": 1000}
        headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key}
        
        try:
            resp = requests.get(url, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            bars = data.get("bars", [])
            if not bars: return pd.DataFrame()
            
            df = pd.DataFrame(bars)
            df['timestamp'] = pd.to_datetime(df['t']).dt.tz_convert('America/New_York')
            df.set_index('timestamp', inplace=True)
            df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}, inplace=True)
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        except requests.exceptions.HTTPError as e:
            if e.response and e.response.status_code == 429:
                print(f"   [Alpaca] Rate limit hit for {ticker}. Waiting 60s...")
                import time; time.sleep(60)
                return self.download_minute_data_http(ticker, date_str, timeframe) # Retry
            print(f"⚠️  HTTP error fetching minute data for {ticker}: {e}")
            return pd.DataFrame()
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Request error for {ticker}: {e}")
            return pd.DataFrame()

    def check_pipeline2(self, ticker, date, entry_price, stop_price, target_price):
        """Check if trade passes Pipeline 2"""
        try:
            features_dict = self.breakout_extractor.extract_features(
                ticker, date, entry_price, stop_price, target_price
            )
            if not features_dict:
                return False, 0.0
            
            feature_values = [features_dict.get(f, 0) for f in self.pipeline2_features]
            X = np.array([feature_values])
            prob = self.pipeline2_model.predict_proba(X)[0][1]
            
            return prob >= self.pipeline2_threshold, prob
        except Exception:
            return False, 0.0

    def manage_position_intraday(self, position, intraday_data, daily_data):
        """Manage a position for one trading day, logic from oct_nov_simulation.py"""
        try:
            if intraday_data is None or intraday_data.empty:
                return None  # Hold position

            current_stop = position['stop_price']
            hit_5R = position['hit_5R']
            position_size = position['position_size']
            partial_exits = position.get('partial_exits', [])
            total_exit_price = sum(exit['price'] * exit['size'] for exit in partial_exits)

            for idx, row in intraday_data.iterrows():
                # 1. Check for 5R partial profit take
                if not hit_5R and position_size == 1.0 and row['High'] >= position['target_price']:
                    partial_exits.append({'time': idx, 'price': position['target_price'], 'size': 0.20, 'reason': 'PARTIAL_5R'})
                    total_exit_price += position['target_price'] * 0.20
                    position_size = 0.80
                    current_stop = position['entry_price'] # Move stop to breakeven
                    hit_5R = True
                    continue

                # 2. Check stop loss (with slippage)
                if row['Low'] <= current_stop:
                    exit_price_with_slippage = current_stop * 0.995
                    partial_exits.append({'time': idx, 'price': exit_price_with_slippage, 'size': position_size, 'reason': 'STOPPED_OUT'})
                    total_exit_price += exit_price_with_slippage * position_size
                    position_size = 0
                    break

                # 3. Check 3:50 PM 10 SMA exit rule
                if idx.strftime('%H:%M') == '15:50' and position_size > 0:
                    if len(daily_data) >= 11 and 'SMA10' in daily_data.columns:
                        current_sma = daily_data['SMA10'].iloc[-1]
                        if pd.notna(current_sma) and row['Close'] < current_sma:
                            next_candles = intraday_data.loc[idx:].iloc[1:]
                            if len(next_candles) > 0:
                                exit_price = next_candles.iloc[0]['Open']
                                partial_exits.append({'time': next_candles.index[0], 'price': exit_price, 'size': position_size, 'reason': 'BELOW_10SMA'})
                                total_exit_price += exit_price * position_size
                                position_size = 0
                                break
            
            if position_size == 0: # Fully exited
                return {
                    'exited': True, 'exit_price': total_exit_price,
                    'exit_time': partial_exits[-1]['time'], 'reason': partial_exits[-1]['reason'],
                    'partial_exits': partial_exits
                }
            
            # Update position for next day
            return {
                'exited': False, 'position_size': position_size, 'stop_price': current_stop,
                'hit_5R': hit_5R, 'partial_exits': partial_exits
            }
        except Exception:
            return None

    def calculate_position_size(self, entry_price, stop_price):
        """Calculate shares based on risk % of STARTING capital"""
        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share == 0:
            return 0
        
        dollar_risk = self.starting_cash * (self.risk_per_trade_pct / 100.0)
        shares = int(dollar_risk / risk_per_share)
        
        max_shares_by_cash = int(self.cash / entry_price)
        shares = min(shares, max_shares_by_cash)
        
        if shares < int(dollar_risk / risk_per_share) * 0.5:
            return 0 # Skip if can't afford at least 50% of intended size
        
        return max(shares, 0)

    def print_results(self):
        """Print final results"""
        print(f"\n\n{'='*80}\n📊 JANUARY-OCTOBER 2025 RESULTS (Alpaca)\n{'='*80}")
        print(f"Starting Capital: ${self.starting_cash:,}")
        print(f"Final Capital: ${self.cash:,.0f}")
        
        total_return = self.cash - self.starting_cash
        return_pct = (total_return / self.starting_cash) * 100
        print(f"Total Return: ${total_return:,.0f} ({return_pct:+.2f}%)")
        print(f"\nTrades Taken: {len(self.trades)}")
        
        if len(self.trades) > 0:
            winners = [t for t in self.trades if t['pnl'] > 0]
            losers = [t for t in self.trades if t['pnl'] <= 0]
            print(f"Winners: {len(winners)} ({len(winners)/len(self.trades)*100:.1f}%)")
            print(f"Losers: {len(losers)} ({len(losers)/len(self.trades)*100:.1f}%)")
            
            print(f"\n📋 All Trades:")
            print("-" * 80)
            for trade in self.trades:
                exit_date_str = trade.get('exit_date', trade['date'])
                print(f"{trade['date']} {trade['time']} {trade['ticker']:6} "
                      f"${trade['entry']:7.2f} → ${trade['exit']:7.2f} (exit: {exit_date_str}) "
                      f"P&L: ${trade['pnl']:7.0f} ({trade.get('pnl_pct', 0):+.1f}%) | {trade.get('outcome', 'N/A')}")
        print(f"{'='*80}\n")

    def run_simulation(self, tickers, ignore_checkpoint=False):
        trading_days = self.get_trading_days()
        if ignore_checkpoint:
            start_day_idx = 0
            start_date = None
            active_positions = []
        else:
            start_day_idx, start_date, active_positions = self.load_checkpoint(ignore_checkpoint=False)
            if start_day_idx > 0:
                print(f"Resuming from checkpoint: Day {start_day_idx}/{len(trading_days)} ({start_date})")
            else:
                active_positions = []
        
        print(f"\n{'='*80}")
        print(f"JANUARY-OCTOBER 2025 SIMULATION (Alpaca data)")
        print(f"{'='*80}")
        print(f"Tickers: {len(tickers)}")
        print(f"Trading days: {len(trading_days)}")
        print(f"Starting capital: ${self.starting_cash:,}")
        print(f"{'='*80}\n")

        for day_idx, date in enumerate(trading_days[start_day_idx:], start=start_day_idx):
            print(f"\n[Day {day_idx}/{len(trading_days)}] {date}")
            print("-" * 80)
            
            entered_tickers_today = set()
            
            try:
                # --- 1. MANAGE ACTIVE POSITIONS ---
                if active_positions:
                    print(f"\n   📊 Managing {len(active_positions)} open position(s)...")
                    remaining_positions = []
                    for position in active_positions:
                        ticker = position['ticker']
                        
                        # For the 10SMA rule, we need daily data. We use the breakout_extractor's internal downloader.
                        daily_data = self.breakout_extractor._download_daily_data(ticker, datetime.strptime(date, '%Y-%m-%d'))
                        if daily_data is not None and len(daily_data) > 10:
                            daily_data['SMA10'] = daily_data['Close'].rolling(10).mean()

                        intraday_data = self.download_minute_data_http(ticker, date, '5Min')
                        result = self.manage_position_intraday(position, intraday_data, daily_data)
                        
                        if result and result['exited']: # Position closed
                            cost = position['shares'] * position['entry_price']
                            proceeds = position['shares'] * result['exit_price']
                            pnl = proceeds - cost
                            pnl_pct = (pnl / cost) * 100
                            self.cash += proceeds # Add full proceeds back to cash
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
                            self.save_trades()
                        elif result and not result['exited']:
                            position['position_size'] = result['position_size']
                            position['stop_price'] = result['stop_price'] # Stop may have moved to B/E
                            position['hit_5R'] = result['hit_5R']
                            remaining_positions.append(position)
                            print(f"      📌 {ticker}: Holding (stop: ${result['stop_price']:.2f})")
                        else: # No data or error
                            remaining_positions.append(position)
                    active_positions = remaining_positions

                # --- 2. SCAN FOR NEW OPPORTUNITIES ---
                watchlist = self.scan_for_consolidations(tickers, date)
                if not watchlist:
                    print("   No consolidations found")
                    self.save_checkpoint(day_idx, date, active_positions)
                    continue
                print(f"   📋 Watchlist: {', '.join([w['ticker'] for w in watchlist[:10]])}" + \
                      (f" + {len(watchlist)-10} more" if len(watchlist) > 10 else ""))

                # --- 3. MONITOR WATCHLIST FOR ENTRIES ---
                print(f"\n   🔍 Continuously evaluating watchlist stocks...")
                trades_entered_today = 0
                max_trades_per_day = 3
                hot_list = {}
                watchlist = sorted(watchlist, key=lambda x: x['prob'], reverse=True)

                sample_ticker = watchlist[0]['ticker']
                market_bars = self.download_minute_data_http(sample_ticker, date, '5Min')
                if market_bars.empty:
                    self.save_checkpoint(day_idx + 1, date, active_positions)
                    continue

                # Pre-fetch all intraday data for the watchlist to avoid repeated downloads
                intraday_data_cache = {
                    stock['ticker']: self.download_minute_data_http(stock['ticker'], date, '5Min')
                    for stock in watchlist
                }

                for timestamp, _ in market_bars.iterrows():
                    if trades_entered_today >= max_trades_per_day: break

                    # 1. Check for triggers on stocks already in the hot list
                    for ticker in list(hot_list.keys()):
                        if trades_entered_today >= max_trades_per_day: break
                        intraday_data = intraday_data_cache.get(ticker)
                        if intraday_data is None or timestamp not in intraday_data.index: continue

                        current_bar = intraday_data.loc[timestamp]
                        if current_bar['High'] >= hot_list[ticker]['trigger']:
                            if ticker in entered_tickers_today: continue
                            
                            entry_details = hot_list[ticker]
                            shares = self.calculate_position_size(entry_details['entry'], entry_details['stop'])
                            if shares > 0 and (shares * entry_details['entry']) <= self.cash:
                                self.cash -= (shares * entry_details['entry'])
                                active_positions.append({
                                    'ticker': ticker, 'entry_date': date, 'entry_time': timestamp.strftime('%H:%M'),
                                    'entry_price': entry_details['entry'], 'stop_price': entry_details['stop'],
                                    'target_price': entry_details['target'], 'shares': shares,
                                    'position_size': 1.0, 'hit_5R': False, 'p2_prob': entry_details['prob'],
                                    'partial_exits': []
                                })
                                trades_entered_today += 1
                                entered_tickers_today.add(ticker)
                                print(f"\n   🚨 {ticker} at {timestamp.strftime('%H:%M')}")
                                print(f"      P2 (pre-screened): {entry_details['prob']*100:.1f}% ✅")
                                print(f"      ✅ ENTERED: ${entry_details['entry']:.2f}")
                                print(f"      Shares: {shares} | Stop: ${entry_details['stop']:.2f} | Cash: ${self.cash:,.0f}")
                                del hot_list[ticker]

                    # 2. Screen the main watchlist to add new stocks to the hot list
                    for stock in watchlist:
                        ticker = stock['ticker']
                        if ticker in hot_list or ticker in entered_tickers_today: continue
                        
                        intraday_data = intraday_data_cache.get(ticker)
                        if intraday_data is None or timestamp not in intraday_data.index: continue
                        
                        current_bar_idx = intraday_data.index.get_loc(timestamp)
                        if current_bar_idx == 0: continue

                        prev_bar = intraday_data.iloc[current_bar_idx - 1]
                        trigger_price, entry_price, stop_price = prev_bar['High'], prev_bar['High'], prev_bar['Low']
                        if stop_price >= entry_price: continue

                        target_price = entry_price + 5 * (entry_price - stop_price)
                        passed, prob = self.check_pipeline2(ticker, date, entry_price, stop_price, target_price)
                        if passed:
                            print(f"   🔥 {ticker} added to hot list at {timestamp.strftime('%H:%M')}. P2 Prob: {prob:.2f}. Trigger > ${trigger_price:.2f}")
                            hot_list[ticker] = {'prob': prob, 'trigger': trigger_price, 'entry': entry_price, 'stop': stop_price, 'target': target_price}

                self.save_checkpoint(day_idx + 1, date, active_positions)
            except Exception as e:
                print(f"[ERROR] Exception on day {day_idx} ({date}): {e}")
                self.save_checkpoint(day_idx + 1, date, active_positions)
                self.save_trades()
                raise
        self.print_results()

def main():
    """Run the simulation using Alpaca API data"""
    # Load all tickers with trades since January 2025
    tickers = []
    import argparse
    parser = argparse.ArgumentParser(description="Run Jan-Oct 2025 Alpaca simulation.")
    parser.add_argument('--ignore-checkpoint', action='store_true', help='Ignore checkpoint and start from scratch')
    args = parser.parse_args()

    try:
        with open('data/processed/tickers_since_jan.txt', 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(tickers)} tickers for simulation.")
        simulator = JanOctSimulatorAlpaca(starting_cash=30000)
        simulator.run_simulation(tickers, ignore_checkpoint=args.ignore_checkpoint)
    except FileNotFoundError:
        print("Error: `data/processed/tickers_since_jan.txt` not found.")

if __name__ == '__main__':
    main()
