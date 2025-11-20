#!/usr/bin/env python3
"""
Single Ticker Test - Verify simulation logic on one stock for October 2025
Simpler version to debug trading logic without complexity of multiple tickers
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import joblib
from src.ml_pipeline.breakout_features import BreakoutFeatureExtractor

class SingleTickerTest:
    """Test simulation on a single ticker"""
    
    def __init__(self, ticker: str):
        self.ticker = ticker
        
        # Load Pipeline 2
        self.pipeline2_model = joblib.load('models/5R_predictor_5min.pkl')
        with open('models/5R_predictor_5min_features.txt', 'r') as f:
            self.pipeline2_features = [line.strip() for line in f]
        
        self.extractor = BreakoutFeatureExtractor(timeframe='5Min')
        
        # Trading parameters
        self.initial_capital = 30000
        self.capital = 30000
        self.risk_per_trade = 0.005
        self.pipeline2_threshold = 0.45
        
        # Trade tracking
        self.position = None
        self.trades = []
        
        print(f"{'='*80}")
        print(f"Single Ticker Test: {ticker} - October 2025")
        print(f"{'='*80}\n")
        print(f"Starting Capital: ${self.capital:,.2f}")
        print(f"Risk Per Trade: {self.risk_per_trade*100}%")
        print(f"Pipeline 2 Threshold: {self.pipeline2_threshold*100}%\n")
    
    def run_october_2025(self):
        """Run simulation for October 2025"""
        
        # October 2025 trading days
        october_days = pd.date_range('2025-10-01', '2025-10-31', freq='B')  # Business days
        
        for day_num, date in enumerate(october_days, 1):
            date_str = date.strftime('%Y-%m-%d')
            print(f"\n[Day {day_num}/{len(october_days)}] {date_str}")
            
            # Check if we need to exit position based on yesterday's close
            if self.position:
                self._check_daily_exit(date_str)
            
            # Get 5-minute intraday data
            try:
                next_date = (date + timedelta(days=1)).strftime('%Y-%m-%d')
                ticker_obj = yf.Ticker(self.ticker)
                intraday_data = ticker_obj.history(start=date_str, end=next_date, interval='5m')
                
                if intraday_data.empty:
                    print(f"  ⚠️  No data available")
                    continue
                
                print(f"  ✅ Downloaded {len(intraday_data)} 5-min bars")
                
                # Process each 5-minute bar
                for idx, (timestamp, row) in enumerate(intraday_data.iterrows()):
                    time_str = timestamp.strftime('%H:%M')
                    
                    # Check position stops/targets first
                    if self.position:
                        self._check_intraday_exits(timestamp, row, intraday_data.iloc[:idx+1])
                    
                    # If no position, check for entry (only 9:30 AM - 11:30 AM)
                    if not self.position:
                        hour = timestamp.hour
                        minute = timestamp.minute
                        
                        if (hour == 9 and minute >= 30) or (hour == 10) or (hour == 11 and minute <= 30):
                            self._check_entry(timestamp, row, intraday_data.iloc[:idx+1], date_str)
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                continue
        
        # Close any remaining position at end of month
        if self.position:
            print(f"\n⚠️  Closing position at end of month")
            self._exit_position(october_days[-1], self.position['entry'], 'END_OF_MONTH')
        
        # Print final results
        self._print_summary()
    
    def _check_entry(self, timestamp, row, data_up_to_now, date_str):
        """Check if we should enter a position
        
        Workflow:
        1. Every 5 min: Run Pipeline 2 with current high-of-day
        2. If prob >= 30%: Lock in that high-of-day as trigger price
        3. When new candle breaks trigger: Enter at trigger × 1.005
        """
        
        # Calculate high and low of day so far (including current bar)
        high_of_day = data_up_to_now['High'].max()
        low_of_day = data_up_to_now['Low'].min()
        
        # Get high of day BEFORE current bar
        if len(data_up_to_now) > 1:
            high_before = data_up_to_now.iloc[:-1]['High'].max()
        else:
            high_before = 0
        
        # Setup hypothetical trade with current high-of-day as trigger
        trigger_price = high_of_day
        entry_price = trigger_price * 1.005  # Entry with slippage
        stop_price = low_of_day
        risk = entry_price - stop_price
        
        if risk <= 0:
            return
        
        target_price = entry_price + 5 * risk
        
        # Extract features for Pipeline 2 using current high-of-day
        try:
            entry_date = datetime.strptime(date_str, '%Y-%m-%d')
            features_dict = self.extractor.extract_features(
                self.ticker, entry_date, entry_price, stop_price, target_price
            )
            
            if not features_dict:
                return
            
            # Get probability
            feature_values = [features_dict.get(f, 0) for f in self.pipeline2_features]
            X = np.array([feature_values])
            prob = self.pipeline2_model.predict_proba(X)[0][1]
            
            # Check if passes threshold AND current bar made new high
            if prob >= self.pipeline2_threshold:
                # Check if current bar broke above previous high-of-day
                if row['High'] > high_before:
                    # Enter at new high + slippage
                    self._enter_position(timestamp, entry_price, stop_price, target_price, prob)
        
        except Exception as e:
            return
    
    def _enter_position(self, timestamp, entry, stop, target, probability):
        """Enter a position"""
        
        risk_per_share = entry - stop
        risk_dollars = self.capital * self.risk_per_trade
        position_size = int(risk_dollars / risk_per_share)
        
        if position_size <= 0:
            return
        
        self.position = {
            'entry_time': timestamp,
            'entry': entry,
            'stop': stop,
            'target': target,
            'size': position_size,
            'probability': probability,
            'breakeven_triggered': False,
            'partial_exit_done': False
        }
        
        time_str = timestamp.strftime('%H:%M')
        print(f"  ✅ ENTERED at {time_str}")
        print(f"     Entry: ${entry:.2f} | Stop: ${stop:.2f} | Target: ${target:.2f}")
        print(f"     Size: {position_size} shares | Probability: {probability*100:.1f}%")
        print(f"     Risk: ${risk_per_share:.2f}/share (${risk_per_share*position_size:.2f} total)")
    
    def _check_intraday_exits(self, timestamp, row, data_up_to_now):
        """Check for stop loss, profit target, or 3:50 PM exit"""
        
        time_str = timestamp.strftime('%H:%M')
        
        # Check stop loss
        if row['Low'] <= self.position['stop']:
            exit_price = self.position['stop'] * 0.995  # 0.5% slippage
            self._exit_position(timestamp, exit_price, 'STOPPED_OUT')
            return
        
        # Check 5R target (take 20% profit, move stop to breakeven)
        if not self.position['partial_exit_done'] and row['High'] >= self.position['target']:
            # Take 20% profit at target
            partial_size = int(self.position['size'] * 0.20)
            remaining_size = self.position['size'] - partial_size
            
            partial_pnl = partial_size * (self.position['target'] - self.position['entry'])
            self.capital += partial_pnl
            
            print(f"  🎯 5R TARGET HIT at {time_str}")
            print(f"     Sold 20% ({partial_size} shares) at ${self.position['target']:.2f}")
            print(f"     P&L: ${partial_pnl:,.0f}")
            
            # Move stop to breakeven (with 0.5% slippage)
            self.position['stop'] = self.position['entry'] * 0.995
            self.position['size'] = remaining_size
            self.position['partial_exit_done'] = True
            
            print(f"     Remaining: {remaining_size} shares | New stop: ${self.position['stop']:.2f}")
        
        # Check 3:50 PM close below 10 SMA
        if time_str == '15:50':
            # Get daily data to calculate 10 SMA
            try:
                ticker_obj = yf.Ticker(self.ticker)
                entry_date = self.position['entry_time'].date()
                start_date = entry_date - timedelta(days=20)
                daily_data = ticker_obj.history(start=start_date.strftime('%Y-%m-%d'), 
                                                end=(entry_date + timedelta(days=1)).strftime('%Y-%m-%d'))
                
                if len(daily_data) >= 11:
                    daily_data['SMA10'] = daily_data['Close'].rolling(window=10).mean()
                    current_sma = daily_data['SMA10'].iloc[-1]
                    
                    if pd.notna(current_sma) and row['Close'] < current_sma:
                        # Exit at 3:55 PM open (next candle)
                        next_candles = data_up_to_now.iloc[1:]
                        if len(next_candles) > 0:
                            exit_price = next_candles.iloc[0]['Open']
                            print(f"  📉 3:50 PM close below 10 SMA (${current_sma:.2f})")
                            self._exit_position(timestamp, exit_price, 'BELOW_10SMA_350PM')
            except:
                pass
    
    def _check_daily_exit(self, today_str):
        """Check if yesterday's close was below 10 SMA - exit at today's open"""
        
        try:
            ticker_obj = yf.Ticker(self.ticker)
            today = datetime.strptime(today_str, '%Y-%m-%d')
            start_date = today - timedelta(days=20)
            
            # Get daily data up to today
            daily_data = ticker_obj.history(start=start_date.strftime('%Y-%m-%d'),
                                           end=(today + timedelta(days=1)).strftime('%Y-%m-%d'))
            
            if len(daily_data) >= 11:
                daily_data['SMA10'] = daily_data['Close'].rolling(window=10).mean()
                
                # Check yesterday's close vs 10 SMA
                yesterday_close = daily_data['Close'].iloc[-1]
                yesterday_sma = daily_data['SMA10'].iloc[-1]
                
                if pd.notna(yesterday_sma) and yesterday_close < yesterday_sma:
                    # Get today's opening bar
                    next_date = (today + timedelta(days=1)).strftime('%Y-%m-%d')
                    intraday = ticker_obj.history(start=today_str, end=next_date, interval='5m')
                    
                    if not intraday.empty:
                        exit_price = intraday.iloc[0]['Open']
                        print(f"  📉 Yesterday close (${yesterday_close:.2f}) below 10 SMA (${yesterday_sma:.2f})")
                        self._exit_position(today, exit_price, 'BELOW_10SMA_DAILY')
        except:
            pass
    
    def _exit_position(self, timestamp, exit_price, reason):
        """Exit the current position"""
        
        size = self.position['size']
        entry = self.position['entry']
        pnl = size * (exit_price - entry)
        pnl_pct = ((exit_price - entry) / entry) * 100
        
        self.capital += pnl
        
        # Record trade
        self.trades.append({
            'entry_time': self.position['entry_time'],
            'exit_time': timestamp,
            'entry': entry,
            'exit': exit_price,
            'size': size,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason,
            'probability': self.position['probability']
        })
        
        time_str = timestamp.strftime('%H:%M') if hasattr(timestamp, 'strftime') else str(timestamp)
        emoji = "✅" if pnl > 0 else "❌"
        
        print(f"  {emoji} EXITED at {time_str}")
        print(f"     Exit: ${exit_price:.2f} | P&L: ${pnl:,.0f} ({pnl_pct:+.1f}%) | {reason}")
        print(f"     Capital: ${self.capital:,.0f}")
        
        self.position = None
    
    def _print_summary(self):
        """Print final summary"""
        
        print(f"\n{'='*80}")
        print(f"SIMULATION COMPLETE")
        print(f"{'='*80}\n")
        
        print(f"Starting Capital: ${self.initial_capital:,.2f}")
        print(f"Final Capital: ${self.capital:,.2f}")
        
        total_return = self.capital - self.initial_capital
        return_pct = (total_return / self.initial_capital) * 100
        
        print(f"Total Return: ${total_return:,.2f} ({return_pct:+.2f}%)\n")
        
        if self.trades:
            winners = [t for t in self.trades if t['pnl'] > 0]
            losers = [t for t in self.trades if t['pnl'] < 0]
            
            print(f"Total Trades: {len(self.trades)}")
            print(f"Winners: {len(winners)} ({len(winners)/len(self.trades)*100:.1f}%)")
            print(f"Losers: {len(losers)} ({len(losers)/len(self.trades)*100:.1f}%)")
            
            if winners:
                avg_win = sum([t['pnl'] for t in winners]) / len(winners)
                print(f"Average Win: ${avg_win:,.0f}")
            
            if losers:
                avg_loss = sum([t['pnl'] for t in losers]) / len(losers)
                print(f"Average Loss: ${avg_loss:,.0f}")
            
            # Print all trades
            print(f"\n{'='*80}")
            print("ALL TRADES:")
            print(f"{'='*80}\n")
            
            for i, trade in enumerate(self.trades, 1):
                entry_str = trade['entry_time'].strftime('%Y-%m-%d %H:%M')
                exit_str = trade['exit_time'].strftime('%Y-%m-%d %H:%M') if hasattr(trade['exit_time'], 'strftime') else str(trade['exit_time'])
                
                print(f"{i}. {entry_str} → {exit_str}")
                print(f"   Entry: ${trade['entry']:.2f} | Exit: ${trade['exit']:.2f}")
                print(f"   Size: {trade['size']} shares | P&L: ${trade['pnl']:,.0f} ({trade['pnl_pct']:+.1f}%)")
                print(f"   Probability: {trade['probability']*100:.1f}% | Reason: {trade['reason']}\n")
        else:
            print("No trades executed")


def main():
    """Main entry point"""
    
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
    else:
        ticker = input("Enter ticker to test (e.g., RGTI): ").strip().upper()
    
    if not ticker:
        print("❌ No ticker provided")
        return
    
    sim = SingleTickerTest(ticker)
    sim.run_october_2025()


if __name__ == '__main__':
    main()
