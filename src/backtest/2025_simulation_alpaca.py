
#!/usr/bin/env python3
"""
2025 SIMULATION using Alpaca API for 5-min data, yfinance for daily data
Simulates live trading from Jan 2 - Oct 31, 2025
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import joblib
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import requests
from dotenv import load_dotenv
import os

# Import feature extractors
from ml_pipeline.consolidation_features import ConsolidationFeatureExtractor
from ml_pipeline.breakout_features import BreakoutFeatureExtractor

def download_minute_data_alpaca(ticker, date, timeframe='5Min'):
	"""Fetches minute data from Alpaca using HTTP requests."""
	load_dotenv()
	api_key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
	secret_key = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
	if not api_key or not secret_key:
		print(f"❌ Alpaca API keys not found for {ticker} {date}")
		return pd.DataFrame()
	url = f"https://data.alpaca.markets/v2/stocks/{ticker}/bars"
	start_time = f"{date}T09:30:00Z"
	end_time = f"{date}T16:00:00Z"
	params = {"start": start_time, "end": end_time, "timeframe": timeframe, "adjustment": "all", "limit": 1000}
	headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key}
	try:
		resp = requests.get(url, params=params, headers=headers)
		resp.raise_for_status()
		data = resp.json()
		bars = data.get("bars", [])
		if not bars:
			return pd.DataFrame()
		df = pd.DataFrame(bars)
		df['timestamp'] = pd.to_datetime(df['t']).dt.tz_convert('America/New_York')
		df.set_index('timestamp', inplace=True)
		df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}, inplace=True)
		return df[['Open', 'High', 'Low', 'Close', 'Volume']]
	except requests.exceptions.HTTPError as e:
		if e.response and e.response.status_code == 429:
			print(f"   [Alpaca] Rate limit hit for {ticker}. Waiting 60s...")
			import time; time.sleep(60)
			return download_minute_data_alpaca(ticker, date, timeframe)
		print(f"⚠️  HTTP error fetching minute data for {ticker}: {e}")
		return pd.DataFrame()
	except requests.exceptions.RequestException as e:
		print(f"⚠️  Request error for {ticker}: {e}")
		return pd.DataFrame()

import pandas_market_calendars as mcal

class Alpaca2025Simulator:
	def __init__(self, starting_cash=30000):
		print("Loading models...")
		pipeline1_data = joblib.load('models/consolidation_detector.pkl')
		pipeline2_model = joblib.load('models/5R_predictor_5min.pkl')
		self.pipeline1_model = pipeline1_data['model']
		self.pipeline1_features = pipeline1_data['feature_names']
		self.pipeline2_model = pipeline2_model
		with open('models/5R_predictor_5min_features.txt', 'r') as f:
			self.pipeline2_features = [line.strip() for line in f]
		self.consol_extractor = ConsolidationFeatureExtractor()
		self.breakout_extractor = BreakoutFeatureExtractor(timeframe='5Min')
		self.pipeline1_threshold = 0.70
		self.pipeline2_threshold = 0.45
		self.starting_cash = starting_cash
		self.cash = starting_cash
		self.risk_per_trade_pct = 0.5
		self.trades = []

	def get_trading_days(self, start_date='2025-01-02', end_date='2025-10-31'):
		nyse = mcal.get_calendar('NYSE')
		sched = nyse.schedule(start_date=start_date, end_date=end_date)
		return [d.strftime('%Y-%m-%d') for d in sched.index]

	def scan_for_consolidations(self, tickers, date):
		from concurrent.futures import ThreadPoolExecutor, as_completed
		scan_date_dt = datetime.strptime(date, '%Y-%m-%d')
		watchlist = []
		def process_ticker(ticker):
			try:
				features_dict = self.consol_extractor.extract_features(ticker, scan_date_dt)
				if not features_dict:
					return None
				feature_values = [features_dict.get(f, 0) for f in self.pipeline1_features]
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

	def check_pipeline2(self, ticker, date, entry_price, stop_price, target_price):
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

	def calculate_position_size(self, entry_price, stop_price):
		risk_per_share = abs(entry_price - stop_price)
		if risk_per_share == 0:
			return 0
		dollar_risk = self.starting_cash * (self.risk_per_trade_pct / 100.0)
		shares = int(dollar_risk / risk_per_share)
		max_shares = int(self.cash / entry_price)
		shares = min(shares, max_shares)
		if shares < int(dollar_risk / risk_per_share) * 0.5:
			return 0
		return max(shares, 0)

	def run_simulation(self, tickers):
		trading_days = self.get_trading_days()
		active_positions = []
		entered_tickers_this_month = set()
		print(f"\n{'='*80}")
		print(f"JANUARY-OCTOBER 2025 SIMULATION (Alpaca 5-min, yfinance daily)")
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
			# Manage open positions
			if active_positions:
				print(f"\n   📊 Managing {len(active_positions)} open position(s)...")
				remaining_positions = []
				for position in active_positions:
					ticker = position['ticker']
					date_dt = datetime.strptime(date, '%Y-%m-%d')
					start_date = (date_dt - timedelta(days=15)).strftime('%Y-%m-%d')
					daily_data = yf.download(ticker, start=start_date, end=date, progress=False)
					if len(daily_data) >= 11:
						# Handle MultiIndex columns from yfinance
						if isinstance(daily_data.columns, pd.MultiIndex):
							close_col = [col for col in daily_data.columns if col[0] == 'Close']
							if close_col:
								close_col = close_col[0]
								daily_data['SMA10'] = daily_data[close_col].rolling(10).mean()
								yesterday_close = daily_data[close_col].iloc[-1].item()
								yesterday_sma = daily_data['SMA10'].iloc[-1].item()
							else:
								yesterday_close = yesterday_sma = None
						else:
							daily_data['SMA10'] = daily_data['Close'].rolling(10).mean()
							yesterday_close = daily_data['Close'].iloc[-1].item()
							yesterday_sma = daily_data['SMA10'].iloc[-1].item()
						if pd.notna(yesterday_sma) and yesterday_close < yesterday_sma:
							intraday_data = download_minute_data_alpaca(ticker, date, '5Min')
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
					intraday_data = download_minute_data_alpaca(ticker, date, '5Min')
					result = self.manage_position_intraday(position, date, intraday_data, daily_data)
					if result and result['exited']:
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
						position['position_size'] = result['position_size']
						position['stop_price'] = result['stop_price']
						position['hit_5R'] = result['hit_5R']
						remaining_positions.append(position)
						print(f"      📌 {ticker}: Holding (stop: ${result['stop_price']:.2f})")
					else:
						remaining_positions.append(position)
				active_positions = remaining_positions
			# Pipeline 1 scan (parallel)
			watchlist = self.scan_for_consolidations(tickers, date)
			if not watchlist:
				print("   No consolidations found")
				continue
			print(f"   📋 Watchlist: {', '.join([w['ticker'] for w in watchlist[:10]])}" + (f" + {len(watchlist)-10} more" if len(watchlist) > 10 else ""))
			# Get previous day highs
			prev_day_data = {}
			for stock in watchlist:
				try:
					ticker = stock['ticker']
					date_dt = datetime.strptime(date, '%Y-%m-%d')
					prev_date = (date_dt - timedelta(days=3)).strftime('%Y-%m-%d')
					df_prev = yf.download(ticker, start=prev_date, end=date, progress=False)
					if len(df_prev) >= 1:
						high_val = df_prev['High'].iloc[-1]
						if hasattr(high_val, 'item'):
							prev_day_data[ticker] = float(high_val.item())
						else:
							prev_day_data[ticker] = float(high_val)
				except:
					pass
			print(f"\n   🔍 Continuously evaluating watchlist stocks...")
			trades_entered_today = 0
			max_trades_per_day = 3
			hot_list = {}
			watchlist = sorted(watchlist, key=lambda x: x['prob'], reverse=True)
			# Pre-fetch all intraday data for the watchlist in parallel
			from concurrent.futures import ThreadPoolExecutor
			intraday_data_cache = {}
			def fetch_intraday(ticker):
				return ticker, download_minute_data_alpaca(ticker, date, '5Min')
			with ThreadPoolExecutor(max_workers=8) as executor:
				results = executor.map(fetch_intraday, [stock['ticker'] for stock in watchlist])
				for ticker, data in results:
					intraday_data_cache[ticker] = data
			for stock in watchlist:
				ticker = stock['ticker']
				prev_day_high = prev_day_data.get(ticker, stock['consol_high'])
				intraday_data = intraday_data_cache.get(ticker)
				if intraday_data is None or len(intraday_data) == 0:
					continue
				for bar_idx, (timestamp, row) in enumerate(intraday_data.iterrows()):
					if trades_entered_today >= max_trades_per_day:
						break
					high_so_far = intraday_data.loc[:timestamp, 'High'].max()
					low_so_far = intraday_data.loc[:timestamp, 'Low'].min()
					hypothetical_entry = high_so_far * 1.005
					hypothetical_stop = low_so_far
					risk = hypothetical_entry - hypothetical_stop
					if risk <= 0:
						continue
					hypothetical_target = hypothetical_entry + 5 * risk
					passed, prob = self.check_pipeline2(
						ticker, date, hypothetical_entry, hypothetical_stop, hypothetical_target
					)
					if passed and prob >= self.pipeline2_threshold:
						hot_list[ticker] = {
							'prob': prob,
							'time': timestamp,
							'trigger': high_so_far,
							'entry': hypothetical_entry,
							'stop': hypothetical_stop,
							'target': hypothetical_target
						}
					if ticker in hot_list and row['High'] > hot_list[ticker]['trigger']:
						if ticker in entered_tickers_today or ticker in entered_tickers_this_month:
							continue
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
							'partial_exits': []
						}
						active_positions.append(position)
						trades_entered_today += 1
						entered_tickers_today.add(ticker)
						entered_tickers_this_month.add(ticker)
						print(f"\n   🚨 {ticker} at {timestamp.strftime('%H:%M')}")
						print(f"      P2 (pre-screened): {prob*100:.1f}% ✅")
						print(f"      ✅ ENTERED: ${entry_price:.2f}")
						print(f"      Shares: {shares} | Stop: ${stop_price:.2f} | Cash: ${self.cash:,.0f}")
						del hot_list[ticker]
						break
		self.print_results()

	def manage_position_intraday(self, position, date, intraday_data, daily_data):
		try:
			if intraday_data is None or intraday_data.empty:
				return None
			current_stop = position['stop_price']
			hit_5R = position['hit_5R']
			position_size = position['position_size']
			partial_exits = position.get('partial_exits', [])
			total_exit_price = sum(exit['price'] * exit['size'] for exit in partial_exits)
			for idx, row in intraday_data.iterrows():
				if not hit_5R and position_size == 1.0 and row['High'] >= position['target_price']:
					partial_exits.append({'time': idx, 'price': position['target_price'], 'size': 0.20, 'reason': 'PARTIAL_5R'})
					total_exit_price += position['target_price'] * 0.20
					position_size = 0.80
					current_stop = position['entry_price']
					hit_5R = True
					continue
				if row['Low'] <= current_stop:
					exit_price_with_slippage = current_stop * 0.995
					partial_exits.append({'time': idx, 'price': exit_price_with_slippage, 'size': position_size, 'reason': 'STOPPED_OUT'})
					total_exit_price += exit_price_with_slippage * position_size
					position_size = 0
					break
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
			if position_size == 0:
				return {
					'exited': True, 'exit_price': total_exit_price,
					'exit_time': partial_exits[-1]['time'], 'reason': partial_exits[-1]['reason'],
					'partial_exits': partial_exits
				}
			return {
				'exited': False, 'position_size': position_size, 'stop_price': current_stop,
				'hit_5R': hit_5R, 'partial_exits': partial_exits
			}
		except Exception:
			return None

	def print_results(self):
		print(f"\n\n{'='*80}")
		print(f"📊 JANUARY-OCTOBER 2025 RESULTS (Alpaca 5-min, yfinance daily)")
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
	tickers = []
	with open('data/processed/tickers_since_jan.txt', 'r') as f:
		for line in f:
			ticker = line.strip()
			if ticker:
				tickers.append(ticker)
	print(f"Loaded {len(tickers)} tickers with trades since January 2025")
	print("Priority sorting enabled: highest Pipeline 1 probabilities evaluated first")
	sim = Alpaca2025Simulator(starting_cash=30000)
	sim.run_simulation(tickers)

if __name__ == '__main__':
	main()
