#!/usr/bin/env python3
"""
Live Trading Bot - CLI Application

Phase 1: Simulated scanning with cached historical data
- Add/remove tickers to hotlist via Pipeline 1
- Monitor hotlist stocks during market hours (simulated)
- Generate BUY signals when Pipeline 2 conditions met
- Daily revalidation of hotlist

Usage:
    python3 src/trading_bot/live_scanner.py --date 2025-10-01
    
Commands:
    add TICKER      - Add ticker to hotlist (runs Pipeline 1 check)
    remove TICKER   - Remove ticker from hotlist
    hotlist         - View current hotlist with probabilities
    scan            - Run intraday scan on hotlist (simulated)
    revalidate      - Re-check all hotlist tickers with Pipeline 1
    signals         - Show signals generated today
    help            - Show available commands
    quit/exit       - Exit the bot
"""

import sys
import json
import joblib
import numpy as np
import pandas as pd
import yaml
import warnings
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

# Suppress ALL warnings (sklearn feature name warnings are verbose)
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Also suppress warnings from sklearn at import time
import logging
logging.getLogger('sklearn').setLevel(logging.ERROR)

# Context manager to suppress stderr during predictions
from contextlib import contextmanager

@contextmanager
def suppress_stderr():
    """Temporarily suppress stderr to hide sklearn warnings"""
    stderr_fd = sys.stderr.fileno()
    old_stderr = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stderr_fd)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_stderr, stderr_fd)
        os.close(old_stderr)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_pipeline.consolidation_features import ConsolidationFeatureExtractor
from ml_pipeline.breakout_features import BreakoutFeatureExtractor
from data_processing.download_minute_data import TwelveDataMinuteDownloader


class HotlistManager:
    """Manages the watchlist of stocks that passed Pipeline 1"""
    
    def __init__(self, config: dict):
        self.config = config
        self.hotlist_file = Path(config['hotlist_file'])
        self.pipeline1_threshold = config['pipeline1_threshold']
        
        # Load Pipeline 1 model
        pipeline1_data = joblib.load(config['pipeline1_model'])
        self.pipeline1_model = pipeline1_data['model']
        self.pipeline1_features = pipeline1_data['feature_names']
        
        # Feature extractor
        self.consol_extractor = ConsolidationFeatureExtractor(lookback_days=10)
        
        # Load existing hotlist
        self.hotlist = self._load_hotlist()
        
    def _load_hotlist(self) -> List[Dict]:
        """Load hotlist from JSON file"""
        if self.hotlist_file.exists():
            with open(self.hotlist_file, 'r') as f:
                data = json.load(f)
                return data.get('hotlist', [])
        return []
    
    def _save_hotlist(self):
        """Save hotlist to JSON file"""
        self.hotlist_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.hotlist_file, 'w') as f:
            json.dump({
                'hotlist': self.hotlist,
                'last_updated': datetime.now().isoformat()
            }, f, indent=2)
    
    def check_pipeline1(self, ticker: str, check_date: datetime) -> Optional[Tuple[float, dict]]:
        """
        Run Pipeline 1 check on ticker
        Returns (probability, features_dict) or None if failed
        """
        try:
            features = self.consol_extractor.extract_features(ticker, check_date)
            if not features:
                return None
            
            # Build feature vector
            feature_values = [features.get(f, 0) for f in self.pipeline1_features]
            X = pd.DataFrame([feature_values], columns=self.pipeline1_features)
            
            # Predict (suppress warnings)
            with suppress_stderr():
                prob = self.pipeline1_model.predict_proba(X)[0][1]
            
            return prob, features
            
        except Exception as e:
            print(f"   ⚠️  Error checking Pipeline 1 for {ticker}: {e}")
            return None
    
    def add_ticker(self, ticker: str, check_date: datetime) -> Tuple[bool, Optional[float]]:
        """
        Add ticker to hotlist if passes Pipeline 1
        Returns (success, probability)
        """
        # Check if already in hotlist
        if any(stock['ticker'] == ticker for stock in self.hotlist):
            print(f"   ℹ️  {ticker} already in hotlist")
            return False, None
        
        # Run Pipeline 1 check
        result = self.check_pipeline1(ticker, check_date)
        if not result:
            print(f"   ❌ Failed to extract features for {ticker}")
            return False, None
        
        prob, features = result
        
        # Check threshold
        if prob >= self.pipeline1_threshold:
            entry = {
                'ticker': ticker,
                'probability': prob,
                'added_date': check_date.strftime('%Y-%m-%d'),
                'consol_high': features.get('price_high', 0),
                'consol_low': features.get('price_low', 0),
                'consol_range_pct': features.get('price_range_pct', 0)
            }
            self.hotlist.append(entry)
            self._save_hotlist()
            return True, prob
        else:
            return False, prob
    
    def remove_ticker(self, ticker: str) -> bool:
        """Remove ticker from hotlist"""
        initial_len = len(self.hotlist)
        self.hotlist = [s for s in self.hotlist if s['ticker'] != ticker]
        
        if len(self.hotlist) < initial_len:
            self._save_hotlist()
            return True
        return False
    
    def revalidate_all(self, check_date: datetime) -> List[str]:
        """
        Re-check all hotlist tickers with Pipeline 1
        Returns list of removed tickers
        """
        removed = []
        updated_hotlist = []
        
        for stock in self.hotlist:
            ticker = stock['ticker']
            result = self.check_pipeline1(ticker, check_date)
            
            if not result:
                print(f"   ⚠️  {ticker}: Failed to extract features - keeping in hotlist")
                updated_hotlist.append(stock)
                continue
            
            prob, features = result
            
            if prob >= self.pipeline1_threshold:
                # Update probability
                stock['probability'] = prob
                stock['consol_high'] = features.get('price_high', 0)
                stock['consol_low'] = features.get('price_low', 0)
                stock['consol_range_pct'] = features.get('price_range_pct', 0)
                updated_hotlist.append(stock)
                print(f"   ✅ {ticker}: {prob*100:.1f}% - Still valid")
            else:
                removed.append(ticker)
                print(f"   ❌ {ticker}: {prob*100:.1f}% - Removed (below threshold)")
        
        self.hotlist = updated_hotlist
        self._save_hotlist()
        return removed
    
    def get_hotlist(self) -> List[Dict]:
        """Return current hotlist"""
        return self.hotlist
    
    def clear_hotlist(self):
        """Clear entire hotlist"""
        self.hotlist = []
        self._save_hotlist()


class LiveScanner:
    """Main trading bot that monitors hotlist and generates signals"""
    
    def __init__(self, config: dict, simulation_date: str = None):
        self.config = config
        self.simulation_date = simulation_date
        
        print("\n" + "="*80)
        print("🤖 LIVE TRADING BOT - CLI")
        print("="*80)
        print(f"\nMode: {'SIMULATION' if simulation_date else 'LIVE'}")
        if simulation_date:
            print(f"Date: {simulation_date}")
        print()
        
        # Initialize managers
        print("Loading models...")
        self.hotlist_manager = HotlistManager(config)
        
        # Load Pipeline 2 model
        self.pipeline2_model = joblib.load(config['pipeline2_model'])
        with open(config['pipeline2_features'], 'r') as f:
            self.pipeline2_features = [line.strip() for line in f]
        
        # Feature extractors
        self.td_downloader = TwelveDataMinuteDownloader()
        self.breakout_extractor = BreakoutFeatureExtractor(
            timeframe='5min',
            downloader=self.td_downloader
        )
        
        print("✅ Models loaded")
        print(f"   Pipeline 1 threshold: {config['pipeline1_threshold']:.0%}")
        print(f"   Pipeline 2 threshold: {config['pipeline2_threshold']:.0%}")
        
        # Tracking
        self.signals_today = []
        self.signaled_tickers = set()  # Track what we've already signaled
        
    def check_pipeline2(self, ticker: str, entry_date: datetime, 
                       entry_price: float, stop_price: float, 
                       target_price: float) -> Optional[float]:
        """
        Check if trade passes Pipeline 2
        Returns probability or None if failed
        """
        try:
            features = self.breakout_extractor.extract_features(
                ticker=ticker,
                entry_date=entry_date,
                entry_price=entry_price,
                stop_price=stop_price,
                target_price=target_price
            )
            
            if not features:
                return None
            
            # Predict (suppress warnings)
            X = np.array([[features.get(f, 0) for f in self.pipeline2_features]])
            X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
            with suppress_stderr():
                prob = self.pipeline2_model.predict_proba(X)[0][1]
            
            return prob
            
        except Exception as e:
            return None
    
    def generate_signal(self, ticker: str, time_str: str, entry: float, 
                       stop: float, target: float, prob: float, 
                       risk_pct: float):
        """Generate and display BUY signal"""
        risk_per_share = entry - stop
        
        signal = {
            'timestamp': f"{self.simulation_date} {time_str}",
            'ticker': ticker,
            'entry_price': entry,
            'stop_price': stop,
            'target_price': target,
            'risk_per_share': risk_per_share,
            'risk_pct': risk_pct,
            'probability': prob,
            'rr_ratio': 5.0
        }
        
        self.signals_today.append(signal)
        self.signaled_tickers.add(ticker)
        
        # Display signal
        print("\n" + "="*80)
        print("🚨 BUY SIGNAL GENERATED 🚨")
        print("="*80)
        print(f"\n📅 Time: {signal['timestamp']}")
        print(f"📈 Ticker: {ticker}")
        print(f"\n💰 Trade Setup:")
        print(f"   Entry:  ${entry:.2f}")
        print(f"   Stop:   ${stop:.2f}")
        print(f"   Target: ${target:.2f} (5R)")
        print(f"   Risk:   ${risk_per_share:.2f}/share ({risk_pct:.2f}%)")
        print(f"\n🎯 Confidence: {prob*100:.1f}%")
        print(f"📊 Risk/Reward: 1:5")
        print("="*80 + "\n")
        
    def scan_intraday_simulated(self, date_str: str):
        """
        Simulate intraday scanning using cached data
        Monitors hotlist stocks throughout the day
        """
        hotlist = self.hotlist_manager.get_hotlist()
        
        if not hotlist:
            print("❌ Hotlist is empty. Add tickers first with 'add TICKER'")
            return
        
        print(f"\n{'='*80}")
        print(f"📊 INTRADAY SCAN - {date_str}")
        print(f"{'='*80}")
        print(f"\n🔍 Monitoring {len(hotlist)} stocks: {', '.join([s['ticker'] for s in hotlist])}")
        print(f"   Simulation mode: Using cached 5-min data\n")
        
        # Reset daily tracking
        self.signaled_tickers.clear()
        self.signals_today.clear()
        
        scan_date = datetime.strptime(date_str, '%Y-%m-%d')
        
        for stock in hotlist:
            ticker = stock['ticker']
            
            print(f"\n{'─'*80}")
            print(f"📈 {ticker} - Consolidation prob: {stock['probability']*100:.1f}%")
            print(f"{'─'*80}")
            
            # Get intraday data
            try:
                intraday_data = self.td_downloader.download_minute_data(
                    ticker, date_str, timeframe='5min'
                )
                
                if intraday_data is None or intraday_data.empty:
                    print(f"   ⚠️  No intraday data available")
                    continue
                
                print(f"   ✅ Loaded {len(intraday_data)} 5-min bars")
                
            except Exception as e:
                print(f"   ❌ Error loading data: {e}")
                continue
            
            # Scan each bar
            # Suppress sklearn warnings for all predictions in this loop
            import io
            from contextlib import redirect_stderr
            
            f = io.StringIO()
            with redirect_stderr(f):
                for i, (bar_time, bar) in enumerate(intraday_data.iterrows()):
                    # Calculate high/low of day so far
                    data_so_far = intraday_data.iloc[:i+1]
                    high_of_day = data_so_far['High'].max()
                    low_of_day = data_so_far['Low'].min()
                    
                    # Calculate hypothetical entry
                    entry_price = high_of_day * (1 + self.config['slippage_pct']/100)
                    stop_price = low_of_day
                    risk = entry_price - stop_price
                    
                    if risk <= 0:
                        continue
                    
                    target_price = entry_price + (5 * risk)
                    risk_pct = (risk / entry_price) * 100
                    
                    # Check Pipeline 2
                    prob = self.check_pipeline2(
                        ticker, scan_date, entry_price, stop_price, target_price
                    )
                    
                    if prob is None:
                        continue
                    
                    time_str = bar_time.strftime('%H:%M') if hasattr(bar_time, 'strftime') else str(bar_time)[11:16]
                    
                    # Check if passes threshold
                    if prob >= self.config['pipeline2_threshold']:
                        # Check if not already signaled
                        if ticker not in self.signaled_tickers:
                            # Clear the progress line first
                            print(f"\n   [{time_str}] 🎯 Pipeline 2: {prob*100:.1f}% ✅ PASS\n")
                            self.generate_signal(
                                ticker, time_str, entry_price, stop_price, 
                                target_price, prob, risk_pct
                            )
                            break  # Only signal once per ticker per day
                    else:
                        # Print every 6th bar to avoid spam
                        if i % 6 == 0:
                            print(f"   [{time_str}] Pipeline 2: {prob*100:.1f}%        ", end='\r')
        
        # Summary
        print(f"\n\n{'='*80}")
        print(f"📊 SCAN COMPLETE")
        print(f"{'='*80}")
        print(f"Signals generated: {len(self.signals_today)}")
        if self.signals_today:
            print(f"Tickers: {', '.join([s['ticker'] for s in self.signals_today])}")
        print(f"{'='*80}\n")
    
    def show_hotlist(self):
        """Display current hotlist"""
        hotlist = self.hotlist_manager.get_hotlist()
        
        if not hotlist:
            print("\n📋 Hotlist is empty\n")
            return
        
        print(f"\n{'='*80}")
        print(f"📋 CURRENT HOTLIST ({len(hotlist)} stocks)")
        print(f"{'='*80}\n")
        print(f"{'Ticker':<8} {'Probability':<12} {'Added':<12} {'Range %':<10}")
        print("─"*80)
        
        for stock in hotlist:
            print(f"{stock['ticker']:<8} {stock['probability']*100:>6.1f}%     "
                  f"{stock['added_date']:<12} {stock['consol_range_pct']:>7.2f}%")
        
        print("="*80 + "\n")
    
    def show_signals(self):
        """Display signals generated today"""
        if not self.signals_today:
            print("\n📊 No signals generated today\n")
            return
        
        print(f"\n{'='*80}")
        print(f"📊 SIGNALS TODAY ({len(self.signals_today)})")
        print(f"{'='*80}\n")
        
        for i, signal in enumerate(self.signals_today, 1):
            print(f"{i}. {signal['ticker']} at {signal['timestamp']}")
            print(f"   Entry: ${signal['entry_price']:.2f} | Stop: ${signal['stop_price']:.2f} | "
                  f"Target: ${signal['target_price']:.2f}")
            print(f"   Confidence: {signal['probability']*100:.1f}%\n")
        
        print("="*80 + "\n")
    
    def show_help(self):
        """Show available commands"""
        print("\n" + "="*80)
        print("📖 AVAILABLE COMMANDS")
        print("="*80)
        print("\nHotlist Management:")
        print("  add TICKER       - Add ticker to hotlist (runs Pipeline 1 check)")
        print("  remove TICKER    - Remove ticker from hotlist")
        print("  hotlist          - View current hotlist")
        print("  clear            - Clear entire hotlist")
        print("\nScanning:")
        print("  scan             - Run intraday scan on hotlist")
        print("  revalidate       - Re-check all hotlist tickers with Pipeline 1")
        print("  signals          - Show signals generated today")
        print("\nOther:")
        print("  help             - Show this help message")
        print("  quit / exit      - Exit the bot")
        print("="*80 + "\n")
    
    def run_cli(self):
        """Run interactive CLI"""
        check_date = datetime.strptime(self.simulation_date, '%Y-%m-%d') if self.simulation_date else datetime.now()
        
        print("Type 'help' for available commands\n")
        
        while True:
            try:
                cmd = input("bot> ").strip().lower()
                
                if not cmd:
                    continue
                
                parts = cmd.split()
                command = parts[0]
                
                if command in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!\n")
                    break
                
                elif command == 'help':
                    self.show_help()
                
                elif command == 'add' and len(parts) >= 2:
                    ticker = parts[1].upper()
                    print(f"\n🔍 Checking {ticker} with Pipeline 1...")
                    success, prob = self.hotlist_manager.add_ticker(ticker, check_date)
                    
                    if success:
                        print(f"   ✅ {ticker} added to hotlist ({prob*100:.1f}%)\n")
                    elif prob is not None:
                        print(f"   ❌ {ticker} failed Pipeline 1 ({prob*100:.1f}% < {self.config['pipeline1_threshold']*100:.0f}%)\n")
                
                elif command == 'remove' and len(parts) >= 2:
                    ticker = parts[1].upper()
                    if self.hotlist_manager.remove_ticker(ticker):
                        print(f"\n✅ {ticker} removed from hotlist\n")
                    else:
                        print(f"\n❌ {ticker} not in hotlist\n")
                
                elif command == 'hotlist':
                    self.show_hotlist()
                
                elif command == 'clear':
                    confirm = input("\n⚠️  Clear entire hotlist? (yes/no): ").strip().lower()
                    if confirm == 'yes':
                        self.hotlist_manager.clear_hotlist()
                        print("✅ Hotlist cleared\n")
                    else:
                        print("❌ Cancelled\n")
                
                elif command == 'scan':
                    if not self.simulation_date:
                        print("\n❌ Scanning requires --date parameter in simulation mode\n")
                    else:
                        self.scan_intraday_simulated(self.simulation_date)
                
                elif command == 'revalidate':
                    print(f"\n🔍 Re-validating hotlist with Pipeline 1...\n")
                    removed = self.hotlist_manager.revalidate_all(check_date)
                    
                    if removed:
                        print(f"\n📊 Removed {len(removed)} ticker(s): {', '.join(removed)}\n")
                    else:
                        print(f"\n✅ All tickers still valid\n")
                
                elif command == 'signals':
                    self.show_signals()
                
                else:
                    print(f"\n❌ Unknown command: {cmd}")
                    print("Type 'help' for available commands\n")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!\n")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Live Trading Bot - CLI')
    parser.add_argument('--date', type=str, help='Simulation date (YYYY-MM-DD)')
    parser.add_argument('--config', type=str, default='config/trading_bot.yaml',
                       help='Config file path')
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate simulation date
    if args.date:
        try:
            datetime.strptime(args.date, '%Y-%m-%d')
        except ValueError:
            print(f"❌ Invalid date format: {args.date}. Use YYYY-MM-DD")
            sys.exit(1)
    
    # Run bot
    bot = LiveScanner(config, simulation_date=args.date)
    bot.run_cli()


if __name__ == '__main__':
    main()
