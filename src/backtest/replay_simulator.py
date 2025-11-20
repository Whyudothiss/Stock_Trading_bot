#!/usr/bin/env python3
"""
Trade Replay Simulator - Tests Both Pipelines Against Historical Trades

Strategy:
1. Load your actual historical trades (Dec 2023 - Oct 2025)
2. For each trade, check if both pipelines would've approved it:
   - Pipeline 1: Extract consolidation features, check if prob > 75%
   - Pipeline 2: Extract breakout features, check if prob > threshold
3. Simulate portfolio with $30K starting capital
4. Calculate performance and compare to actual results

This answers: "If I had used both ML pipelines, how would I have performed?"
"""

import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import Dict, List, Optional
import pytz

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from ml_pipeline.breakout_features import BreakoutFeatureExtractor
    from ml_pipeline.consolidation_features import ConsolidationFeatureExtractor
    from data_processing.download_minute_data import AlpacaMinuteDownloader
except ImportError as e:
    print(f"⚠️  Warning: Could not import ML pipeline modules: {e}")
    print("   Make sure you're running from the project root")
    sys.exit(1)


class TradeReplaySimulator:
    """Replay historical trades through both ML pipelines"""
    
    def __init__(self,
                 pipeline1_model_path: str,
                 pipeline2_model_path: str,
                 starting_cash: float = 30000.0,
                 risk_per_trade_pct: float = 1.0,
                 pipeline1_threshold: float = 0.75,
                 pipeline2_threshold: float = 0.37):
        
        print(f"\n{'='*80}")
        print(f"🔬 TRADE REPLAY SIMULATOR")
        print(f"{'='*80}\n")
        
        # Load models
        print("📦 Loading models...")
        pipeline1_data = joblib.load(pipeline1_model_path)
        pipeline2_data = joblib.load(pipeline2_model_path)
        
        # Extract actual models from saved dictionaries
        if isinstance(pipeline1_data, dict):
            self.pipeline1_model = pipeline1_data['model']
            self.pipeline1_features = pipeline1_data['feature_names']
        else:
            self.pipeline1_model = pipeline1_data
            self.pipeline1_features = None
        
        if isinstance(pipeline2_data, dict):
            self.pipeline2_model = pipeline2_data['model']
        else:
            self.pipeline2_model = pipeline2_data
        
        # Load Pipeline 2 feature names
        feature_file = pipeline2_model_path.replace('.pkl', '_features.txt')
        with open(feature_file, 'r') as f:
            self.pipeline2_features = [line.strip() for line in f]
        
        print(f"   ✓ Pipeline 1 (Consolidation): {pipeline1_model_path}")
        print(f"     Features: {len(self.pipeline1_features) if self.pipeline1_features else 'unknown'}")
        print(f"   ✓ Pipeline 2 (5R Predictor): {pipeline2_model_path}")
        print(f"     Features: {len(self.pipeline2_features)}")
        
        # Initialize feature extractors
        print("\n🔧 Initializing feature extractors...")
        self.consol_extractor = ConsolidationFeatureExtractor(lookback_days=10)
        self.breakout_extractor = BreakoutFeatureExtractor(timeframe='5min')
        
        # Configuration
        self.starting_cash = starting_cash
        self.risk_per_trade_pct = risk_per_trade_pct
        self.pipeline1_threshold = pipeline1_threshold
        self.pipeline2_threshold = pipeline2_threshold
        
        # Portfolio state
        self.cash = starting_cash
        self.positions = {}
        self.trades_taken = []
        self.trades_skipped = []
        
        print(f"\n⚙️  Configuration:")
        print(f"   Starting capital: ${starting_cash:,.0f}")
        print(f"   Risk per trade: {risk_per_trade_pct}%")
        print(f"   Pipeline 1 threshold: {pipeline1_threshold:.0%}")
        print(f"   Pipeline 2 threshold: {pipeline2_threshold:.0%}")
    
    def convert_sgt_to_et(self, sgt_timestamp: str) -> str:
        """Convert SGT timestamp to ET date string"""
        # Normalize microseconds
        if '.' in sgt_timestamp:
            parts = sgt_timestamp.split('.')
            micro = parts[1].split('+')[0]
            micro = micro.ljust(6, '0')[:6]
            sgt_timestamp = f"{parts[0]}.{micro}+08:00"
        
        # Parse and convert
        dt_sgt = datetime.strptime(sgt_timestamp, '%Y-%m-%dT%H:%M:%S.%f%z')
        dt_et = dt_sgt.astimezone(pytz.timezone('US/Eastern'))
        return dt_et.strftime('%Y-%m-%d')
    
    def check_pipeline1(self, ticker: str, entry_date: str) -> Optional[float]:
        """
        Check if Pipeline 1 would approve this trade
        Returns consolidation probability or None if failed
        """
        try:
            # Extract consolidation features
            # Note: entry_date is the breakout day, so we check consolidation on day before
            entry_dt = datetime.strptime(entry_date, '%Y-%m-%d')
            check_dt = entry_dt - timedelta(days=1)
            
            features = self.consol_extractor.extract_features(ticker, check_dt)
            if features is None:
                return None
            
            # Use saved feature names if available, otherwise extract from features dict
            if self.pipeline1_features:
                feature_values = [features.get(f, 0) for f in self.pipeline1_features]
            else:
                # Remove metadata fields
                feature_keys = [k for k in features.keys() if k not in ['ticker', 'date']]
                feature_values = [features[k] for k in feature_keys]
            
            # Predict probability
            X = np.array([feature_values])
            X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
            
            prob = self.pipeline1_model.predict_proba(X)[0][1]
            
            return prob if prob >= self.pipeline1_threshold else None
            
        except Exception as e:
            print(f"      ⚠️  Pipeline 1 error for {ticker}: {e}")
            return None
    
    def check_pipeline2(self, ticker: str, entry_date: str, 
                       entry_price: float, stop_price: float, target_price: float) -> Optional[float]:
        """
        Check if Pipeline 2 would approve this trade
        Returns 5R probability or None if failed
        """
        try:
            # Extract breakout features
            features = self.breakout_extractor.extract_features(
                ticker=ticker,
                entry_date=entry_date,
                entry_price=entry_price,
                stop_price=stop_price,
                target_price=target_price
            )
            
            if features is None:
                return None
            
            # Predict probability
            X = np.array([[features.get(f, 0) for f in self.pipeline2_features]])
            X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
            
            prob = self.pipeline2_model.predict_proba(X)[0][1]
            
            return prob if prob >= self.pipeline2_threshold else None
            
        except Exception as e:
            print(f"      ⚠️  Pipeline 2 error for {ticker}: {e}")
            return None
    
    def calculate_position_size(self, entry_price: float, stop_price: float) -> int:
        """Calculate shares based on risk percentage"""
        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share == 0:
            return 0
        
        dollar_risk = self.cash * (self.risk_per_trade_pct / 100.0)
        shares = int(dollar_risk / risk_per_share)
        
        # Don't exceed available cash
        max_shares = int(self.cash / entry_price) if entry_price > 0 else 0
        return min(shares, max_shares)
    
    def simulate_trade(self, trade_data: dict, trade_num: int, total_trades: int) -> dict:
        """
        Simulate one trade through both pipelines
        Returns trade result dict
        """
        ticker = trade_data['ticker']
        entry_sgt = trade_data['entry_date']
        entry_price = trade_data['entry_price']
        stop_price = trade_data['initial_stop_loss_price']
        target_price = trade_data['ml_target']
        actual_label = trade_data['ml_label']
        
        # Convert timestamp
        entry_date = self.convert_sgt_to_et(entry_sgt)
        
        print(f"\n[{trade_num}/{total_trades}] {ticker} on {entry_date}")
        print(f"   Entry: ${entry_price:.2f} | Stop: ${stop_price:.2f} | Target: ${target_price:.2f}")
        print(f"   Actual outcome: {actual_label}")
        
        result = {
            'ticker': ticker,
            'entry_date': entry_date,
            'entry_price': entry_price,
            'stop_price': stop_price,
            'target_price': target_price,
            'actual_label': actual_label,
            'decision': 'SKIP',
            'reason': '',
            'pipeline1_prob': None,
            'pipeline2_prob': None,
            'shares': 0,
            'cost': 0,
            'pnl': 0,
            'cash_before': self.cash
        }
        
        # Check Pipeline 1
        print(f"   🔍 Checking Pipeline 1 (Consolidation)...", end='')
        p1_prob = self.check_pipeline1(ticker, entry_date)
        
        if p1_prob is None:
            print(f" ❌ REJECT (below {self.pipeline1_threshold:.0%})")
            result['reason'] = 'Failed Pipeline 1'
            return result
        
        result['pipeline1_prob'] = p1_prob
        print(f" ✅ PASS ({p1_prob:.1%})")
        
        # Check Pipeline 2
        print(f"   🔍 Checking Pipeline 2 (5R Predictor)...", end='')
        p2_prob = self.check_pipeline2(ticker, entry_date, entry_price, stop_price, target_price)
        
        if p2_prob is None:
            print(f" ❌ REJECT (below {self.pipeline2_threshold:.0%})")
            result['reason'] = 'Failed Pipeline 2'
            return result
        
        result['pipeline2_prob'] = p2_prob
        print(f" ✅ PASS ({p2_prob:.1%})")
        
        # Calculate position size
        shares = self.calculate_position_size(entry_price, stop_price)
        
        if shares == 0:
            print(f"   💰 Cannot size position (insufficient capital)")
            result['reason'] = 'Insufficient capital'
            return result
        
        cost = shares * entry_price
        
        if cost > self.cash:
            print(f"   💰 Cannot afford trade (need ${cost:,.0f}, have ${self.cash:,.0f})")
            result['reason'] = 'Insufficient cash'
            return result
        
        # Take the trade!
        self.cash -= cost
        
        # Simulate outcome based on actual label
        if actual_label == 'HIT_5R':
            # Hit target
            exit_price = target_price
            pnl = shares * (exit_price - entry_price)
            self.cash += shares * exit_price
            outcome = "🎯 HIT 5R TARGET"
        else:
            # Stopped out
            exit_price = stop_price
            pnl = shares * (exit_price - entry_price)
            self.cash += shares * exit_price
            outcome = "🛑 STOPPED OUT"
        
        result.update({
            'decision': 'TAKE',
            'shares': shares,
            'cost': cost,
            'exit_price': exit_price,
            'pnl': pnl,
            'pnl_pct': (pnl / cost) * 100,
            'cash_after': self.cash
        })
        
        print(f"   ✅ TRADE TAKEN: {shares} shares @ ${entry_price:.2f} = ${cost:,.0f}")
        print(f"   📊 Outcome: {outcome}")
        print(f"   💵 P&L: ${pnl:,.2f} ({result['pnl_pct']:+.2f}%)")
        print(f"   💰 Cash: ${result['cash_before']:,.0f} → ${self.cash:,.0f}")
        
        return result
    
    def run_simulation(self, trades: List[dict]):
        """Run simulation on all trades"""
        
        print(f"\n{'='*80}")
        print(f"🚀 STARTING SIMULATION")
        print(f"{'='*80}")
        print(f"\nTotal trades to replay: {len(trades)}")
        
        for i, trade in enumerate(trades, 1):
            result = self.simulate_trade(trade, i, len(trades))
            
            if result['decision'] == 'TAKE':
                self.trades_taken.append(result)
            else:
                self.trades_skipped.append(result)
        
        self._generate_report(trades)
    
    def _generate_report(self, original_trades: List[dict]):
        """Generate comprehensive performance report"""
        
        print(f"\n\n{'='*80}")
        print(f"📊 FINAL RESULTS")
        print(f"{'='*80}")
        
        # Portfolio performance
        final_equity = self.cash
        total_return = final_equity - self.starting_cash
        return_pct = (total_return / self.starting_cash) * 100
        
        print(f"\n💰 Portfolio Performance:")
        print(f"   Starting capital: ${self.starting_cash:,.2f}")
        print(f"   Final equity: ${final_equity:,.2f}")
        print(f"   Total return: ${total_return:,.2f} ({return_pct:+.2f}%)")
        
        # Trade statistics
        total_evaluated = len(original_trades)
        num_taken = len(self.trades_taken)
        num_skipped = len(self.trades_skipped)
        
        print(f"\n📈 Trade Statistics:")
        print(f"   Trades evaluated: {total_evaluated}")
        print(f"   Trades TAKEN: {num_taken} ({num_taken/total_evaluated*100:.1f}%)")
        print(f"   Trades SKIPPED: {num_skipped} ({num_skipped/total_evaluated*100:.1f}%)")
        
        if self.trades_taken:
            taken = self.trades_taken
            winners = [t for t in taken if t['pnl'] > 0]
            losers = [t for t in taken if t['pnl'] <= 0]
            hit_5r = [t for t in taken if t['actual_label'] == 'HIT_5R']
            
            print(f"\n   Of trades taken:")
            print(f"      Winners: {len(winners)} ({len(winners)/len(taken)*100:.1f}%)")
            print(f"      Losers: {len(losers)} ({len(losers)/len(taken)*100:.1f}%)")
            print(f"      Hit 5R: {len(hit_5r)} ({len(hit_5r)/len(taken)*100:.1f}%)")
            
            if winners:
                avg_win = np.mean([t['pnl_pct'] for t in winners])
                print(f"      Avg winner: {avg_win:+.2f}%")
            
            if losers:
                avg_loss = np.mean([t['pnl_pct'] for t in losers])
                print(f"      Avg loser: {avg_loss:+.2f}%")
            
            if losers and winners:
                profit_factor = abs(sum(t['pnl'] for t in winners) / sum(t['pnl'] for t in losers))
                print(f"      Profit factor: {profit_factor:.2f}")
        
        # Pipeline effectiveness
        print(f"\n🔍 Pipeline Performance:")
        
        # Pipeline 1 rejections
        p1_rejects = [t for t in self.trades_skipped if t['reason'] == 'Failed Pipeline 1']
        print(f"   Pipeline 1 rejections: {len(p1_rejects)}")
        
        # Pipeline 2 rejections  
        p2_rejects = [t for t in self.trades_skipped if t['reason'] == 'Failed Pipeline 2']
        print(f"   Pipeline 2 rejections: {len(p2_rejects)}")
        
        # Check how many 5R winners we caught vs missed
        original_5r_winners = [t for t in original_trades if t['ml_label'] == 'HIT_5R']
        caught_5r = [t for t in self.trades_taken if t['actual_label'] == 'HIT_5R']
        missed_5r = len(original_5r_winners) - len(caught_5r)
        
        print(f"\n   5R Winner Analysis:")
        print(f"      Total 5R winners available: {len(original_5r_winners)}")
        print(f"      5R winners CAUGHT: {len(caught_5r)} ({len(caught_5r)/len(original_5r_winners)*100:.1f}%)")
        print(f"      5R winners MISSED: {missed_5r} ({missed_5r/len(original_5r_winners)*100:.1f}%)")
        
        # Compare to if we took ALL trades
        print(f"\n📊 Comparison to Taking ALL Trades:")
        all_trades_pnl = self._calculate_all_trades_pnl(original_trades)
        print(f"   ML-filtered strategy: ${total_return:,.2f} ({return_pct:+.2f}%)")
        print(f"   Take-all strategy: ${all_trades_pnl:,.2f} ({all_trades_pnl/self.starting_cash*100:+.2f}%)")
        
        if all_trades_pnl != 0:
            improvement = ((total_return - all_trades_pnl) / abs(all_trades_pnl)) * 100
            print(f"   Improvement: {improvement:+.1f}%")
        
        # Save results
        self._save_results()
    
    def _calculate_all_trades_pnl(self, trades: List[dict]) -> float:
        """Calculate what would happen if we took all trades blindly"""
        sim_cash = self.starting_cash
        
        for trade in trades:
            entry = trade['entry_price']
            stop = trade['initial_stop_loss_price']
            target = trade['ml_target']
            
            shares = self.calculate_position_size(entry, stop)
            if shares == 0:
                continue
            
            cost = shares * entry
            if cost > sim_cash:
                continue
            
            sim_cash -= cost
            
            if trade['ml_label'] == 'HIT_5R':
                sim_cash += shares * target
            else:
                sim_cash += shares * stop
        
        return sim_cash - self.starting_cash
    
    def _save_results(self):
        """Save detailed results to JSON"""
        output_dir = Path('data/backtest_results')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        results = {
            'config': {
                'starting_cash': self.starting_cash,
                'risk_per_trade_pct': self.risk_per_trade_pct,
                'pipeline1_threshold': self.pipeline1_threshold,
                'pipeline2_threshold': self.pipeline2_threshold
            },
            'summary': {
                'final_equity': self.cash,
                'total_return': self.cash - self.starting_cash,
                'return_pct': (self.cash - self.starting_cash) / self.starting_cash * 100,
                'trades_taken': len(self.trades_taken),
                'trades_skipped': len(self.trades_skipped)
            },
            'trades_taken': self.trades_taken,
            'trades_skipped': self.trades_skipped
        }
        
        results_file = output_dir / f'replay_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")


def main():
    """Main entry point"""
    
    # Load historical trades
    trades_file = 'data/processed/ml_5R_labels_filtered.jsonl'
    
    with open(trades_file, 'r') as f:
        all_trades = [json.loads(line) for line in f]
    
    # Filter to date range: Dec 2023 - Oct 2025
    start_date = datetime(2023, 12, 1)
    end_date = datetime(2025, 10, 31)
    
    filtered_trades = []
    for trade in all_trades:
        trade_date = datetime.strptime(trade['entry_date'][:10], '%Y-%m-%d')
        if start_date <= trade_date <= end_date:
            filtered_trades.append(trade)
    
    print(f"\n📅 Date range: {start_date.date()} to {end_date.date()}")
    print(f"📊 Trades in range: {len(filtered_trades)} / {len(all_trades)}")
    
    # Run simulation with different thresholds
    thresholds_to_test = [0.30, 0.37, 0.45]
    
    print(f"\n🧪 Testing {len(thresholds_to_test)} Pipeline 2 thresholds: {thresholds_to_test}")
    print(f"   (This helps find optimal trade-off between quantity and quality)\n")
    
    for threshold in thresholds_to_test:
        print(f"\n{'#'*80}")
        print(f"# TESTING PIPELINE 2 THRESHOLD: {threshold:.0%}")
        print(f"{'#'*80}")
        
        simulator = TradeReplaySimulator(
            pipeline1_model_path='models/consolidation_detector.pkl',
            pipeline2_model_path='models/5R_predictor_5min.pkl',
            starting_cash=30000.0,
            risk_per_trade_pct=1.0,
            pipeline1_threshold=0.70,
            pipeline2_threshold=threshold
        )
        
        simulator.run_simulation(filtered_trades)
        
        print(f"\n{'#'*80}\n")
        input("Press Enter to continue to next threshold...")


if __name__ == '__main__':
    main()
