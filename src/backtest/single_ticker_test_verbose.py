#!/usr/bin/env python3
"""
Single ticker simulation with VERBOSE debug output
"""

import sys
import os
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from src.ml_pipeline.breakout_features import BreakoutFeatureExtractor


class SingleTickerTestVerbose:
    def __init__(self, ticker, verbose_date=None):
        self.ticker = ticker
        self.verbose_date = verbose_date  # e.g. '2025-10-02'
        
        # Load Pipeline 2
        self.pipeline2_model = joblib.load('models/5R_predictor_5min.pkl')
        self.extractor = BreakoutFeatureExtractor(timeframe='5Min')
        
        # Load feature names
        with open('models/5R_predictor_5min_features.txt', 'r') as f:
            self.pipeline2_features = [line.strip() for line in f.readlines()]
        
        # Trading parameters
        self.initial_capital = 30000
        self.capital = 30000
        self.risk_per_trade = 0.005
        self.pipeline2_threshold = 0.30
        
        # Position tracking
        self.position = None
        self.trades = []
    
    def run_october_2025(self):
        """Run simulation for October 2025"""
        
        print("="*80)
        print(f"Single Ticker Test (VERBOSE): {self.ticker} - October 2025")
        print(f"Verbose debugging for: {self.verbose_date}")
        print("="*80)
        print(f"\nStarting Capital: ${self.initial_capital:,.2f}")
        print(f"Risk Per Trade: {self.risk_per_trade*100}%")
        print(f"Pipeline 2 Threshold: {self.pipeline2_threshold*100}%\n")
        
        # October 2025 trading days
        october_days = [
            '2025-10-01', '2025-10-02', '2025-10-03', '2025-10-06', '2025-10-07',
            '2025-10-08', '2025-10-09', '2025-10-10', '2025-10-13', '2025-10-14',
            '2025-10-15', '2025-10-16', '2025-10-17', '2025-10-20', '2025-10-21',
            '2025-10-22', '2025-10-23', '2025-10-24', '2025-10-27', '2025-10-28',
            '2025-10-29', '2025-10-30', '2025-10-31'
        ]
        
        for day_num, date_str in enumerate(october_days, 1):
            is_verbose_day = (date_str == self.verbose_date)
            
            if not is_verbose_day:
                continue  # Skip non-verbose days
            
            print(f"[VERBOSE DAY] {date_str}")
            
            # Check if we need to exit based on daily 10 SMA
            if self.position:
                self._check_daily_exit(date_str)
                if not self.position:
                    continue
            
            # Download 5-min data for the day
            try:
                ticker_obj = yf.Ticker(self.ticker)
                next_date = (datetime.strptime(date_str, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
                intraday_data = ticker_obj.history(start=date_str, end=next_date, interval='5m')
                
                if intraday_data.empty:
                    print(f"  ❌ No data for {date_str}")
                    continue
                
                print(f"  ✅ Downloaded {len(intraday_data)} 5-min bars\n")
                
                # Process each 5-minute bar
                for idx, (timestamp, row) in enumerate(intraday_data.iterrows()):
                    time_str = timestamp.strftime('%H:%M')
                    
                    # If we have a position, check exits
                    if self.position:
                        self._check_intraday_exits(timestamp, row, intraday_data.iloc[:idx+1])
                        if not self.position:
                            continue
                    
                    # If no position, check for entry (only 9:30 AM - 11:30 AM)
                    if not self.position:
                        hour = timestamp.hour
                        minute = timestamp.minute
                        
                        if (hour == 9 and minute >= 30) or (hour == 10) or (hour == 11 and minute <= 30):
                            print(f"\n  [{time_str}] Checking entry signal...")
                            self._check_entry(timestamp, row, intraday_data.iloc[:idx+1], date_str)
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                continue
        
        print(f"\n{'='*80}")
        print("VERBOSE DAY COMPLETE")
        print(f"{'='*80}\n")
    
    def _check_entry(self, timestamp, row, data_up_to_now, date_str):
        """Check if we should enter a position (VERBOSE)
        
        Workflow:
        1. Every 5 min: Run Pipeline 2 with current high-of-day as trigger
        2. If prob >= 30%: That high-of-day becomes the trigger price
        3. If current candle breaks trigger: Enter at trigger × 1.005
        """
        
        # Calculate high and low of day so far
        high_of_day = data_up_to_now['High'].max()
        low_of_day = data_up_to_now['Low'].min()
        
        # Trigger = current high-of-day at check time
        trigger_price = high_of_day
        entry_price = trigger_price * 1.005  # Entry with slippage
        stop_price = low_of_day
        risk = entry_price - stop_price
        
        print(f"     High of day: ${high_of_day:.2f} (this is the trigger)")
        print(f"     Low of day: ${low_of_day:.2f}")
        print(f"     Entry price (if triggered): ${entry_price:.2f} (trigger × 1.005)")
        print(f"     Stop: ${stop_price:.2f}")
        print(f"     Risk: ${risk:.2f} ({(risk/entry_price)*100:.2f}%)")
        print(f"     Current bar high: ${row['High']:.2f}")
        
        if risk <= 0:
            print(f"     ❌ SKIP: Risk <= 0")
            return
        
        target_price = entry_price + 5 * risk
        print(f"     5R Target: ${target_price:.2f}")
        
        # Extract features for Pipeline 2
        try:
            entry_date = datetime.strptime(date_str, '%Y-%m-%d')
            features_dict = self.extractor.extract_features(
                self.ticker, entry_date, entry_price, stop_price, target_price
            )
            
            if not features_dict:
                print(f"     ❌ SKIP: Feature extraction failed")
                return
            
            # Get probability
            feature_values = [features_dict.get(f, 0) for f in self.pipeline2_features]
            X = np.array([feature_values])
            prob = self.pipeline2_model.predict_proba(X)[0][1]
            
            print(f"     🤖 Pipeline 2 Probability: {prob*100:.1f}%")
            
            # Print key features
            print(f"     📊 Key Features:")
            print(f"        - Risk %: {features_dict.get('risk_pct', 0):.2f}%")
            print(f"        - Gap %: {features_dict.get('gap_pct', 0):.2f}%")
            print(f"        - Consol range: {features_dict.get('consol_range_pct', 0):.2f}%")
            print(f"        - Consol position: {features_dict.get('consol_position_in_range', 0):.2f}")
            
            # Check if passes threshold
            if prob >= self.pipeline2_threshold:
                print(f"     ✅ Probability >= {self.pipeline2_threshold*100}% threshold")
                
                # Check if current bar broke trigger (high-of-day at check time)
                if row['High'] > trigger_price:
                    print(f"     ✅ Current bar broke trigger! ${row['High']:.2f} > ${trigger_price:.2f}")
                    print(f"     🚀 ENTERING at ${entry_price:.2f} (trigger + 0.5% slippage)")
                    self._enter_position(timestamp, entry_price, stop_price, target_price, prob)
                else:
                    print(f"     ⏳ Waiting for breakout. Current: ${row['High']:.2f} ≤ Trigger: ${trigger_price:.2f}")
            else:
                print(f"     ❌ Probability {prob*100:.1f}% < {self.pipeline2_threshold*100}% threshold")
        
        except Exception as e:
            print(f"     ❌ SKIP: Exception: {e}")
            import traceback
            traceback.print_exc()
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
        print(f"\n✅ ENTERED at {time_str}")
        print(f"   Entry: ${entry:.2f} | Stop: ${stop:.2f} | Target: ${target:.2f}")
        print(f"   Size: {position_size} shares | Probability: {probability*100:.1f}%\n")
    
    def _check_intraday_exits(self, timestamp, row, data_up_to_now):
        """Check for intraday exits (stops, targets, 3:50 PM)"""
        pass
    
    def _check_daily_exit(self, today_str):
        """Check if yesterday's close was below 10 SMA"""
        pass


if __name__ == '__main__':
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
        verbose_date = sys.argv[2] if len(sys.argv) > 2 else '2025-10-02'
    else:
        ticker = input("Enter ticker symbol: ").upper()
        verbose_date = input("Enter date for verbose output (YYYY-MM-DD): ")
    
    sim = SingleTickerTestVerbose(ticker, verbose_date=verbose_date)
    sim.run_october_2025()
