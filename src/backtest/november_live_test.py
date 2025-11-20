#!/usr/bin/env python3
"""
November 2025 LIVE SIMULATION - Day-by-day live trading simulation
This simulates exactly how the system would work live:
1. Each morning: Run Pipeline 1 on all tickers (scan for consolidations)
2. During market hours: Monitor for breakouts on watchlist
3. At breakout: Run Pipeline 2 immediately
4. If P2 passes: Enter trade (avg of breakout candle, stop = day low)
5. Track outcome: See if hits 5R target or stop loss
6. Repeat for all November 2025 trading days with $30K capital
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import joblib
import numpy as np
from datetime import datetime, timedelta
import json
import yfinance as yf
import pandas as pd

# Import feature extractors
from ml_pipeline.consolidation_features import ConsolidationFeatureExtractor
from ml_pipeline.breakout_features import BreakoutFeatureExtractor


class NovemberLiveScanner:
    """Scan November 2025 market data with both pipelines"""
    
    def __init__(self, starting_cash=30000):
        # Load models
        print("Loading models...")
        pipeline1_data = joblib.load('models/consolidation_detector.pkl')
        pipeline2_model = joblib.load('models/5R_predictor_5min.pkl')
        
        self.pipeline1_model = pipeline1_data['model']
        self.pipeline1_features = pipeline1_data['feature_names']
        
        # Pipeline 2 model is saved directly
        self.pipeline2_model = pipeline2_model
        with open('models/5R_predictor_5min_features.txt', 'r') as f:
            self.pipeline2_features = [line.strip() for line in f]
        
        # Feature extractors
        self.consol_extractor = ConsolidationFeatureExtractor()
        self.breakout_extractor = BreakoutFeatureExtractor(timeframe='5Min')
        
        # Thresholds
        self.pipeline1_threshold = 0.70  # 70% consolidation confidence
        self.pipeline2_threshold = 0.30  # 30% 5R probability
        
        # Portfolio
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.risk_per_trade_pct = 0.5  # 0.5% risk per trade
        
        print(f"✅ Models loaded")
        print(f"   Pipeline 1: {len(self.pipeline1_features)} features, {self.pipeline1_threshold*100:.0f}% threshold")
        print(f"   Pipeline 2: {len(self.pipeline2_features)} features, {self.pipeline2_threshold*100:.0f}% threshold")
        print(f"   Starting capital: ${self.starting_cash:,.0f}")
    
    def get_november_trading_days(self):
        """Get list of trading days in November 2025"""
        # November 2025 trading days (excluding weekends and holidays)
        start = datetime(2025, 11, 3)  # First trading day
        end = datetime(2025, 11, 28)   # Last trading day (before Thanksgiving)
        
        trading_days = []
        current = start
        while current <= end:
            # Skip weekends
            if current.weekday() < 5:  # Monday = 0, Friday = 4
                trading_days.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)
        
        return trading_days
    
    def scan_for_consolidations(self, tickers, scan_date):
        """Run Pipeline 1 to find good consolidations"""
        print(f"\n🔍 Scanning for consolidations on {scan_date}...")
        
        # Convert date string to datetime object for feature extractor
        scan_date_dt = datetime.strptime(scan_date, '%Y-%m-%d')
        
        watchlist = []
        for i, ticker in enumerate(tickers):
            if (i + 1) % 20 == 0:
                print(f"   Progress: {i+1}/{len(tickers)}")
            
            try:
                # Extract consolidation features (pass datetime object)
                features_dict = self.consol_extractor.extract_features(ticker, scan_date_dt)
                if not features_dict:
                    continue
                
                # Prepare features
                feature_values = [features_dict.get(f, 0) for f in self.pipeline1_features]
                X = np.array([feature_values])
                
                # Get prediction
                prob = self.pipeline1_model.predict_proba(X)[0][1]
                
                if prob >= self.pipeline1_threshold:
                    # Get consolidation range for breakout detection (use price_high/low from features)
                    consol_high = features_dict.get('price_high', 0)
                    consol_low = features_dict.get('price_low', 0)
                    
                    watchlist.append({
                        'ticker': ticker,
                        'prob': prob,
                        'consol_high': consol_high,
                        'consol_low': consol_low,
                        'range_pct': features_dict.get('price_range_pct', 0)
                    })
            
            except Exception as e:
                continue
        
        print(f"✅ Found {len(watchlist)} consolidations (>={self.pipeline1_threshold*100:.0f}%)")
        return watchlist
    
    def check_for_breakout(self, ticker, date, consol_high, consol_low, prev_day_high):
        """
        Check if ticker broke out on this date using proper entry logic:
        - If GAP UP (>2%): Wait for high of day to be taken out
        - If NO GAP: Wait for previous day high to be taken out
        """
        try:
            # Get 5-min intraday data for this date from Alpaca
            from data_processing.download_minute_data import AlpacaMinuteDownloader
            downloader = AlpacaMinuteDownloader()
            
            data = downloader.download_minute_data(ticker, date, '5Min')
            
            if data is None or data.empty:
                return None
            
            # Check if gapping up
            open_price = data['Open'].iloc[0]
            gap_pct = ((open_price - prev_day_high) / prev_day_high) * 100 if prev_day_high > 0 else 0
            is_gap = gap_pct > 2.0  # Gap up more than 2%
            
            # Track running high of day for gap trades
            running_high = 0
            
            # Check each 5-min bar for breakout
            for idx, row in data.iterrows():
                # Update running high
                running_high = max(running_high, row['High'])
                
                # Determine if breakout condition met
                if is_gap:
                    # Gap trade: need to break current high of day
                    if idx == data.index[0]:  # Skip first candle
                        continue
                    trigger_price = data.loc[:idx].shift(1)['High'].max()  # Previous candles' high
                    breakout = row['High'] > trigger_price
                else:
                    # No gap: need to break previous day high
                    breakout = row['High'] > prev_day_high
                
                if breakout:
                    # BREAKOUT DETECTED!
                    # Entry = average of open and close
                    entry_price = (row['Open'] + row['Close']) / 2
                    
                    # Stop = low of day at this moment (real-time day low)
                    day_low_so_far = data.loc[:idx]['Low'].min()
                    stop_price = day_low_so_far
                    
                    # Target = 5R from entry
                    risk = entry_price - stop_price
                    target_price = entry_price + 5 * risk
                    
                    if risk <= 0:  # Invalid setup
                        continue
                    
                    return {
                        'breakout_time': idx,
                        'breakout_bar_index': data.index.get_loc(idx),
                        'entry_price': entry_price,
                        'stop_price': stop_price,
                        'target_price': target_price,
                        'risk_per_share': risk,
                        'is_gap': is_gap,
                        'gap_pct': gap_pct,
                        'data': data  # Full day data for tracking outcome
                    }
            
            # No breakout on this day
            return None
            
        except Exception as e:
            print(f"      ⚠️  Error checking breakout: {e}")
            return None
    
    def check_pipeline2(self, ticker, date, entry_price, stop_price, target_price):
        """Check if trade passes Pipeline 2"""
        try:
            # Extract breakout features
            features_dict = self.breakout_extractor.extract_features(
                ticker, date, entry_price, stop_price, target_price
            )
            if not features_dict:
                print(f"      ⚠️  Feature extraction failed for {ticker}")
                return False, 0.0
            
            # Prepare features in correct order
            feature_values = [features_dict.get(f, 0) for f in self.pipeline2_features]
            X = np.array([feature_values])
            
            # Get prediction
            prob = self.pipeline2_model.predict_proba(X)[0][1]
            
            return prob >= self.pipeline2_threshold, prob
            
        except Exception as e:
            print(f"      ⚠️  Pipeline 2 error: {e}")
            import traceback
            traceback.print_exc()
            return False, 0.0
    
    def simulate_trade_outcome(self, data, entry_bar_index, entry_price, stop_price, target_price, ticker):
        """
        Simulate what happens after entry with advanced exit rules:
        - Take 20% profit at 5R
        - Trail remaining 80% with 10 SMA and day low
        """
        try:
            # Get bars after entry
            remaining_data = data.iloc[entry_bar_index + 1:].copy()
            
            if remaining_data.empty:
                # No more data - exit at close
                exit_price = data.iloc[-1]['Close']
                if exit_price >= target_price:
                    return 'PARTIAL_5R', data.index[-1], exit_price, 0.20  # Only 20% hit
                elif exit_price <= stop_price:
                    return 'STOPPED', data.index[-1], stop_price, 1.0
                else:
                    return 'OPEN', data.index[-1], exit_price, 1.0
            
            # Calculate 10 SMA for trailing stop
            remaining_data['SMA_10'] = data['Close'].rolling(10).mean()
            
            position_size = 1.0  # Start with 100% position
            total_exit_price = 0
            partial_exits = []
            
            # Track the initial stop (day low at entry) - this becomes our trailing stop
            initial_stop = stop_price
            
            # Check each bar after entry
            for idx, row in remaining_data.iterrows():
                # Update trailing stop to current day low (always use the lowest low so far)
                current_day_low = data.loc[:idx, 'Low'].min()
                trailing_stop = current_day_low
                
                # FIRST: Check if hit 5R target for partial profit
                if position_size == 1.0 and row['High'] >= target_price:
                    # Take 20% profit at 5R
                    partial_exits.append({
                        'time': idx,
                        'price': target_price,
                        'size': 0.20,
                        'reason': 'PARTIAL_5R'
                    })
                    position_size = 0.80  # 80% remaining
                    total_exit_price += target_price * 0.20
                    continue
                
                # THEN: Check exit conditions for remaining position
                if position_size > 0:
                    # Exit if close below 10 SMA
                    if pd.notna(row['SMA_10']) and row['Close'] < row['SMA_10']:
                        partial_exits.append({
                            'time': idx,
                            'price': row['Close'],
                            'size': position_size,
                            'reason': 'BELOW_10SMA'
                        })
                        total_exit_price += row['Close'] * position_size
                        position_size = 0
                        break
                    
                    # Exit if hit trailing stop (current day low)
                    if row['Low'] <= trailing_stop:
                        partial_exits.append({
                            'time': idx,
                            'price': trailing_stop,
                            'size': position_size,
                            'reason': 'DAY_LOW_TRAIL'
                        })
                        total_exit_price += trailing_stop * position_size
                        position_size = 0
                        break
            
            # End of day handling
            if position_size > 0:
                final_price = remaining_data.iloc[-1]['Close']
                partial_exits.append({
                    'time': remaining_data.index[-1],
                    'price': final_price,
                    'size': position_size,
                    'reason': 'EOD_CLOSE'
                })
                total_exit_price += final_price * position_size
            
            # Determine overall outcome
            if len(partial_exits) > 0:
                avg_exit_price = total_exit_price
                last_exit = partial_exits[-1]
                
                # Categorize outcome
                if any(e['reason'] == 'PARTIAL_5R' for e in partial_exits):
                    outcome = 'HIT_5R_PARTIAL'
                elif last_exit['reason'] in ['STOPPED', 'DAY_LOW_TRAIL']:
                    outcome = 'TRAIL_STOP'
                elif last_exit['reason'] == 'BELOW_10SMA':
                    outcome = 'TRAIL_STOP'
                else:
                    outcome = 'OPEN'
                
                return outcome, last_exit['time'], avg_exit_price, partial_exits
            else:
                return 'ERROR', None, None, []
                
        except Exception as e:
            print(f"      ⚠️  Error simulating outcome: {e}")
            import traceback
            traceback.print_exc()
            return 'ERROR', None, None, []
    
    def calculate_position_size(self, entry_price, stop_price):
        """Calculate shares based on 1% risk"""
        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share == 0:
            return 0
        
        dollar_risk = self.cash * (self.risk_per_trade_pct / 100.0)
        shares = int(dollar_risk / risk_per_share)
        
        # Make sure we can afford it
        cost = shares * entry_price
        if cost > self.cash:
            shares = int(self.cash / entry_price)
        
        return max(shares, 0)
    
    def run_november_scan(self, tickers):
        """Scan November 2025 day by day"""
        trading_days = self.get_november_trading_days()
        
        print(f"\n{'='*80}")
        print(f"NOVEMBER 2025 LIVE TEST")
        print(f"{'='*80}")
        print(f"Tickers to scan: {len(tickers)}")
        print(f"Trading days: {len(trading_days)} days")
        print(f"Period: {trading_days[0]} to {trading_days[-1]}")
        print(f"{'='*80}\n")
        
        all_trades = []
        
        # Scan each trading day
        for day_idx, date in enumerate(trading_days, 1):
            print(f"\n[Day {day_idx}/{len(trading_days)}] {date}")
            print("-" * 80)
            
            # Step 1: Run Pipeline 1 to get watchlist
            watchlist = self.scan_for_consolidations(tickers, date)
            
            if not watchlist:
                print("   No consolidations found")
                continue
            
            print(f"\n   📋 Watchlist: {', '.join([w['ticker'] for w in watchlist[:10]])}" + 
                  (f" + {len(watchlist)-10} more" if len(watchlist) > 10 else ""))
            
            # Step 2: Get previous day data for breakout triggers
            import yfinance as yf
            prev_day_data = {}
            for stock in watchlist:
                try:
                    ticker = stock['ticker']
                    df = yf.download(ticker, start=date, end=date, progress=False)
                    if not df.empty and len(df) > 0:
                        # Get previous day high from the downloaded data
                        # For this, we need data before 'date'
                        from datetime import datetime, timedelta
                        date_dt = datetime.strptime(date, '%Y-%m-%d')
                        prev_date = (date_dt - timedelta(days=3)).strftime('%Y-%m-%d')  # Go back 3 days to ensure we get previous trading day
                        df_prev = yf.download(ticker, start=prev_date, end=date, progress=False)
                        if len(df_prev) >= 2:
                            prev_day_data[ticker] = df_prev['High'].iloc[-2]  # Previous day high
                except:
                    pass
            
            # Step 3: Check for breakouts during the day
            print(f"\n   🔍 Monitoring for breakouts during market hours...")
            day_trades = 0
            
            for stock in watchlist:
                ticker = stock['ticker']
                
                # Get previous day high (default to consol high if not available)
                prev_day_high = prev_day_data.get(ticker, stock['consol_high'])
                
                # Check if ticker breaks out
                breakout = self.check_for_breakout(
                    ticker, date, stock['consol_high'], stock['consol_low'], prev_day_high
                )
                
                if not breakout:
                    continue
                
                # BREAKOUT DETECTED!
                print(f"\n   🚨 BREAKOUT: {ticker} at {breakout['breakout_time'].strftime('%H:%M')}")
                print(f"      Entry: ${breakout['entry_price']:.2f} (avg of breakout candle)")
                print(f"      Stop: ${breakout['stop_price']:.2f} (candle/consol low)")
                print(f"      Target: ${breakout['target_price']:.2f} (5R)")
                print(f"      Trigger: {'GAP (high of day)' if breakout['is_gap'] else f'PREV DAY HIGH (${prev_day_high:.2f})'}")
                print(f"      P1 Consolidation: {stock['prob']*100:.1f}%")
                
                # Step 4: Run Pipeline 2 immediately at breakout
                p2_pass, p2_prob = self.check_pipeline2(
                    ticker, date,
                    breakout['entry_price'],
                    breakout['stop_price'],
                    breakout['target_price']
                )
                
                print(f"      P2 Probability: {p2_prob*100:.1f}% {'✅ PASS' if p2_pass else '❌ REJECT'}")
                
                if not p2_pass:
                    print(f"      ❌ Pipeline 2 rejected - no trade")
                    continue
                
                # Both pipelines passed - TAKE THE TRADE!
                print(f"\n      ✅✅ TRADE SIGNAL!")
                
                # Calculate position size
                shares = self.calculate_position_size(
                    breakout['entry_price'], breakout['stop_price']
                )
                
                if shares == 0:
                    print(f"      ⚠️  Insufficient capital")
                    continue
                
                cost = shares * breakout['entry_price']
                risk_pct = abs(breakout['entry_price'] - breakout['stop_price']) / breakout['entry_price'] * 100
                
                print(f"      💰 Position: {shares} shares × ${breakout['entry_price']:.2f} = ${cost:,.0f}")
                print(f"      📉 Risk: {risk_pct:.1f}% per share ({self.risk_per_trade_pct}% of capital)")
                print(f"      💵 Cash before: ${self.cash:,.0f}")
                
                # Simulate outcome using rest of day's data
                print(f"      📊 Watching trade play out...")
                outcome, exit_time, avg_exit_price, partial_exits = self.simulate_trade_outcome(
                    breakout['data'],
                    breakout['breakout_bar_index'],
                    breakout['entry_price'],
                    breakout['stop_price'],
                    breakout['target_price'],
                    ticker
                )
                
                # Calculate P&L based on average exit price
                pnl = shares * (avg_exit_price - breakout['entry_price'])
                pnl_pct = ((avg_exit_price - breakout['entry_price']) / breakout['entry_price']) * 100
                
                # Update cash
                self.cash += pnl
                
                # Print outcome with emoji
                outcome_emoji = {
                    'HIT_5R_PARTIAL': '🎯',
                    'STOPPED': '🛑',
                    'TRAIL_STOP': '📉',
                    'DAY_LOW': '⚠️',
                    'OPEN': '⏸️',
                    'ERROR': '❌'
                }.get(outcome, '❓')
                
                print(f"      {outcome_emoji} Result: {outcome}")
                if exit_time:
                    print(f"      Exit: ${avg_exit_price:.2f} at {exit_time.strftime('%H:%M')}")
                    
                # Show partial exits if any
                if len(partial_exits) > 1:
                    print(f"      Exits:")
                    for pexit in partial_exits:
                        print(f"        - {pexit['size']*100:.0f}% at ${pexit['price']:.2f} ({pexit['reason']})")
                        
                print(f"      💵 P&L: ${pnl:,.0f} ({pnl_pct:+.2f}%)")
                print(f"      💰 Cash after: ${self.cash:,.0f}")
                
                # Record trade
                all_trades.append({
                    'date': date,
                    'ticker': ticker,
                    'breakout_time': breakout['breakout_time'].strftime('%H:%M') if breakout['breakout_time'] else None,
                    'entry_price': breakout['entry_price'],
                    'stop_price': breakout['stop_price'],
                    'target_price': breakout['target_price'],
                    'shares': shares,
                    'is_gap': breakout['is_gap'],
                    'gap_pct': breakout['gap_pct'],
                    'p1_prob': stock['prob'],
                    'p2_prob': p2_prob,
                    'outcome': outcome,
                    'exit_time': exit_time.strftime('%H:%M') if exit_time else None,
                    'exit_price': avg_exit_price,
                    'partial_exits': partial_exits,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'cash_after': self.cash
                })
                
                day_trades += 1
            
            if day_trades > 0:
                print(f"\n   📈 {day_trades} trade(s) taken today")
                day_trades += 1
            
            if day_trades > 0:
                print(f"\n   📈 {day_trades} trade(s) taken today")
            else:
                print(f"\n   No trades taken today")
        
        # Final summary
        print(f"\n\n{'='*80}")
        print(f"📊 NOVEMBER 2025 TEST RESULTS")
        print(f"{'='*80}")
        print(f"Starting Capital: ${self.starting_cash:,.0f}")
        print(f"Final Capital: ${self.cash:,.0f}")
        print(f"Total Return: ${self.cash - self.starting_cash:,.0f} ({(self.cash/self.starting_cash - 1)*100:+.2f}%)")
        print(f"\nTrades Taken: {len(all_trades)}")
        
        if all_trades:
            winners = [t for t in all_trades if t['outcome'] == 'HIT_5R']
            losers = [t for t in all_trades if t['outcome'] in ['STOPPED', 'LOSS_1R']]
            
            print(f"Winners: {len(winners)} ({len(winners)/len(all_trades)*100:.1f}%)")
            print(f"Losers: {len(losers)} ({len(losers)/len(all_trades)*100:.1f}%)")
            
            if winners:
                avg_win = sum(t['pnl'] for t in winners) / len(winners)
                print(f"Avg Winner: ${avg_win:,.2f}")
            
            if losers:
                avg_loss = sum(t['pnl'] for t in losers) / len(losers)
                print(f"Avg Loser: ${avg_loss:,.2f}")
            
            print(f"\n📋 All Trades:")
            print("-" * 80)
            for t in all_trades:
                icon = "🎯" if t['outcome'] == 'HIT_5R' else ("🛑" if t['outcome'] == 'STOPPED' else "⏸️")
                time_str = f" {t['breakout_time']}" if 'breakout_time' in t else ""
                exit_str = f" → exit {t['exit_time']}" if t.get('exit_time') else ""
                print(f"{icon} {t['date']}{time_str} {t['ticker']:6s} ${t['entry_price']:7.2f}{exit_str}")
                print(f"   {t['outcome']:20s} P&L: ${t['pnl']:8,.0f} ({t['pnl_pct']:+6.2f}%) Cash: ${t['cash_after']:,.0f}")
        
        print(f"{'='*80}\n")
        
        # Save results
        output_file = Path('data/backtest_results/november_2025_live_test.json')
        output_file.parent.mkdir(exist_ok=True)
        
        results = {
            'config': {
                'starting_cash': self.starting_cash,
                'pipeline1_threshold': self.pipeline1_threshold,
                'pipeline2_threshold': self.pipeline2_threshold,
                'risk_per_trade_pct': self.risk_per_trade_pct
            },
            'summary': {
                'final_cash': self.cash,
                'total_return': self.cash - self.starting_cash,
                'return_pct': (self.cash/self.starting_cash - 1) * 100,
                'trades_taken': len(all_trades),
                'winners': len(winners) if all_trades else 0,
                'losers': len(losers) if all_trades else 0
            },
            'trades': all_trades
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to: {output_file}")


def main():
    # User-specified tickers
    user_tickers = ['LMND', 'CIFR', 'REAL', 'BE', 'NUAI', 'ONDS', 'GRAL']
    
    # Recent 6-month tickers
    recent_tickers = ['QUBT','TMC','PONY','BBAI','QBTS','RZLV','ALAB','ONDS','JOBY','OPEN','SES','OKLO','APLD','NBIS','CRWV','TLRY','RGTI','ARQQ','HIMS','AMPX','PLUG','IREN','MP','GLXY','TEM','RR','PRCH','SOUN','PL','SERV','SMR','NEGG','QS','RKLB','SEDG','SATL','UAMY','EOSE','MBOT','CVNA','WRD','NVTS','ZENA','UAVS','INVZ','LAES','IONQ','CAR','SRFM','CRDO','RCAT','OUST','PGY','AEHR','RDDT','ZEPP','UUUU','AMDL','WULF','HSAI','RUN','BE','DVLT','HWM','GRPN','MSTR','ACHR','NUTX','EVGO','CEP','CGC','LTBR','PTIR','AISP','AEVA','GRYP','CTM','GRAL','NXT','CWRV','SMCI','GRRR','INDO','MODV','USAR','COIN','CONL','UMAC','FFAI','BKSY','PDYN','CRCL','BTCS','WKHS','UPST','ARAI','DPRO','SOFX','FUBO','VNET','DOMO','SYM','RKT','LEU','SLDP','ROBN','KTOS','NIU','BULL','ASTS','SY','VSAT','RERE','ABAT','METC','HOVR','HIVE','NIO','BMNR','BZAI','VERI','SNDK','BTQ','GPRO','PBF','NUAI','W','GSIT']
    
    # Combine and deduplicate
    all_tickers = list(set(user_tickers + recent_tickers))
    all_tickers = [t.upper() for t in all_tickers]
    
    print(f"November 2025 Live Test")
    print(f"Scanning {len(all_tickers)} tickers")
    print(f"This will take some time as it downloads intraday data...\n")
    
    # Run scanner
    scanner = NovemberLiveScanner(starting_cash=30000)
    scanner.run_november_scan(all_tickers)


if __name__ == "__main__":
    main()
