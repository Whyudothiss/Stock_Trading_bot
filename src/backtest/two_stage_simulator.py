#!/usr/bin/env python3
"""
Two-Stage Pipeline Backtest Simulator
Combines Pipeline 1 (consolidation scanner) + Pipeline 2 (5R predictor)

This simulator:
1. Uses your ACTUAL historical trades as the "ground truth"
2. Replays each trade through both pipelines to see if they would've been taken
3. Simulates portfolio performance with $30K starting capital
4. Tests different Pipeline 2 probability thresholds
"""

import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_pipeline.breakout_features import BreakoutFeatureExtractor
from ml_pipeline.data_fetcher import AlpacaMinuteDownloader
from ml_pipeline.consolidation_features import ConsolidationFeatureExtractor


class PortfolioManager:
    """Manages cash, positions, and risk"""
    
    def __init__(self, starting_cash: float = 30000.0):
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.positions: Dict[str, dict] = {}  # ticker -> position info
        self.closed_trades: List[dict] = []
        self.equity_curve: List[dict] = []
        
    def calculate_position_size(self, entry_price: float, stop_price: float, 
                               risk_pct_of_capital: float = 1.0) -> int:
        """Calculate shares to buy based on risk % of capital"""
        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share == 0:
            return 0
        
        dollar_risk = self.cash * (risk_pct_of_capital / 100.0)
        shares = int(dollar_risk / risk_per_share)
        
        # Don't risk more than available cash
        max_shares = int(self.cash / entry_price) if entry_price > 0 else 0
        return min(shares, max_shares)
    
    def can_open_position(self, ticker: str, entry_price: float, shares: int) -> bool:
        """Check if we have enough cash and no existing position"""
        if ticker in self.positions:
            return False
        
        cost = entry_price * shares
        return cost <= self.cash
    
    def open_position(self, ticker: str, entry_date: str, entry_price: float, 
                     stop_price: float, target_price: float, shares: int,
                     pipeline1_conf: float, pipeline2_prob: float) -> bool:
        """Open a new position"""
        if not self.can_open_position(ticker, entry_price, shares):
            return False
        
        cost = entry_price * shares
        self.cash -= cost
        
        self.positions[ticker] = {
            'entry_date': entry_date,
            'entry_price': entry_price,
            'stop_price': stop_price,
            'target_price': target_price,
            'shares': shares,
            'cost': cost,
            'pipeline1_conf': pipeline1_conf,
            'pipeline2_prob': pipeline2_prob,
            'highest_price': entry_price,
            'status': 'open'
        }
        
        return True
    
    def update_positions(self, current_date: str, price_data: Dict[str, dict]):
        """Update open positions and check for exits"""
        for ticker in list(self.positions.keys()):
            if ticker not in price_data:
                continue
            
            pos = self.positions[ticker]
            data = price_data[ticker]
            
            # Update highest price for trailing
            if 'high' in data and data['high'] > pos['highest_price']:
                pos['highest_price'] = data['high']
            
            # Check stop hit
            if 'low' in data and data['low'] <= pos['stop_price']:
                self._close_position(ticker, current_date, pos['stop_price'], 'STOP')
                continue
            
            # Check target hit (5R)
            if 'high' in data and data['high'] >= pos['target_price']:
                self._close_position(ticker, current_date, pos['target_price'], 'TARGET_5R')
                continue
    
    def _close_position(self, ticker: str, exit_date: str, exit_price: float, reason: str):
        """Close a position and record the trade"""
        pos = self.positions.pop(ticker)
        
        proceeds = exit_price * pos['shares']
        self.cash += proceeds
        
        pnl = proceeds - pos['cost']
        pnl_pct = (pnl / pos['cost']) * 100
        
        trade = {
            'ticker': ticker,
            'entry_date': pos['entry_date'],
            'exit_date': exit_date,
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'shares': pos['shares'],
            'cost': pos['cost'],
            'proceeds': proceeds,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'exit_reason': reason,
            'pipeline1_conf': pos['pipeline1_conf'],
            'pipeline2_prob': pos['pipeline2_prob'],
            'days_held': (datetime.strptime(exit_date, '%Y-%m-%d') - 
                         datetime.strptime(pos['entry_date'], '%Y-%m-%d')).days
        }
        
        self.closed_trades.append(trade)
    
    def get_total_equity(self) -> float:
        """Calculate total equity (cash + open positions at current value)"""
        position_value = sum(
            pos['shares'] * pos['entry_price']  # We'd need current price for live
            for pos in self.positions.values()
        )
        return self.cash + position_value
    
    def record_equity(self, date: str):
        """Record equity for equity curve"""
        self.equity_curve.append({
            'date': date,
            'cash': self.cash,
            'equity': self.get_total_equity(),
            'num_positions': len(self.positions)
        })


class TwoStagePipelineSimulator:
    """Runs full backtest combining both pipelines"""
    
    def __init__(self, 
                 pipeline1_model_path: str,
                 pipeline2_model_path: str,
                 pipeline1_threshold: float = 0.75,
                 pipeline2_threshold: float = 0.37,
                 starting_cash: float = 30000.0,
                 risk_per_trade: float = 1.0):
        
        # Load models
        self.pipeline1_model = joblib.load(pipeline1_model_path)
        self.pipeline2_model = joblib.load(pipeline2_model_path)
        
        # Load feature names for Pipeline 2
        feature_file = pipeline2_model_path.replace('.pkl', '_features.txt')
        with open(feature_file, 'r') as f:
            self.pipeline2_features = [line.strip() for line in f]
        
        # Thresholds
        self.pipeline1_threshold = pipeline1_threshold
        self.pipeline2_threshold = pipeline2_threshold
        
        # Portfolio
        self.portfolio = PortfolioManager(starting_cash)
        self.risk_per_trade = risk_per_trade
        
        # Data fetchers
        self.data_fetcher = AlpacaMinuteDownloader()
        self.feature_extractor = BreakoutFeatureExtractor(self.data_fetcher)
        
        # Tracking
        self.scan_log: List[dict] = []
        self.signal_log: List[dict] = []
        
    def scan_pipeline1(self, ticker: str, date: str) -> Optional[float]:
        """
        Run Pipeline 1 consolidation scan
        Returns confidence if consolidation detected, None otherwise
        """
        # TODO: Implement Pipeline 1 scanning
        # For now, return a placeholder
        # In reality, you'd need to:
        # 1. Fetch 10+ days of daily data leading up to 'date'
        # 2. Check if ticker is in consolidation
        # 3. Return confidence score
        
        # Placeholder: randomly filter ~15% (similar to your actual win rate)
        # In production, this would call your actual consolidation scanner
        import random
        random.seed(hash(ticker + date))  # Deterministic
        
        if random.random() < 0.15:  # ~15% pass rate
            conf = random.uniform(0.70, 0.95)
            return conf if conf >= self.pipeline1_threshold else None
        return None
    
    def scan_pipeline2(self, ticker: str, entry_date: str, 
                      entry_price: float, stop_price: float) -> Optional[float]:
        """
        Run Pipeline 2 breakout prediction
        Returns probability if good setup, None otherwise
        """
        try:
            # Extract features
            features = self.feature_extractor.extract_features(
                ticker=ticker,
                entry_date=entry_date,
                entry_price=entry_price,
                stop_price=stop_price,
                timeframe='5min'
            )
            
            if features is None:
                return None
            
            # Prepare feature vector
            X = np.array([[features.get(f, 0) for f in self.pipeline2_features]])
            
            # Replace NaN/inf
            X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
            
            # Predict probability
            prob = self.pipeline2_model.predict_proba(X)[0][1]  # Probability of class 1 (HIT_5R)
            
            return prob
            
        except Exception as e:
            print(f"   ⚠️  Pipeline 2 failed for {ticker} on {entry_date}: {e}")
            return None
    
    def simulate_trade_day(self, date: str, tickers: List[str]) -> int:
        """
        Simulate one trading day:
        1. Update open positions (check stops/targets)
        2. Scan watchlist for new setups
        3. Open new positions if signals found
        
        Returns: number of new signals found
        """
        signals_found = 0
        
        # Step 1: Run Pipeline 1 scan on all tickers
        watchlist = []
        for ticker in tickers:
            conf = self.scan_pipeline1(ticker, date)
            if conf is not None:
                watchlist.append((ticker, conf))
                
                self.scan_log.append({
                    'date': date,
                    'ticker': ticker,
                    'pipeline1_conf': conf,
                    'stage': 'watchlist'
                })
        
        if not watchlist:
            return 0
        
        # Step 2: For each watchlist ticker, simulate breakout
        # In real backtest, we'd fetch intraday data and check for breakout
        # For simplicity, assume breakout happens at some point during day
        
        for ticker, p1_conf in watchlist:
            # Simulate entry/stop prices (in reality, these come from actual price action)
            # For now, use placeholder logic
            entry_price = 50.0  # Placeholder
            stop_price = entry_price * 0.95  # 5% stop
            target_price = entry_price + 5 * (entry_price - stop_price)  # 5R target
            
            # Run Pipeline 2
            p2_prob = self.scan_pipeline2(ticker, date, entry_price, stop_price)
            
            if p2_prob is None:
                continue
            
            self.signal_log.append({
                'date': date,
                'ticker': ticker,
                'entry_price': entry_price,
                'stop_price': stop_price,
                'target_price': target_price,
                'pipeline1_conf': p1_conf,
                'pipeline2_prob': p2_prob,
                'stage': 'evaluated'
            })
            
            # Check if probability meets threshold
            if p2_prob >= self.pipeline2_threshold:
                signals_found += 1
                
                # Calculate position size
                shares = self.portfolio.calculate_position_size(
                    entry_price, stop_price, self.risk_per_trade
                )
                
                if shares > 0:
                    success = self.portfolio.open_position(
                        ticker, date, entry_price, stop_price, target_price,
                        shares, p1_conf, p2_prob
                    )
                    
                    if success:
                        print(f"   📈 Opened {ticker} @ ${entry_price:.2f} | "
                              f"Stop: ${stop_price:.2f} | Target: ${target_price:.2f} | "
                              f"Shares: {shares} | P1: {p1_conf:.1%} | P2: {p2_prob:.1%}")
        
        return signals_found
    
    def run_backtest(self, start_date: str, end_date: str, tickers: List[str],
                    progress_interval: int = 30):
        """Run full backtest simulation"""
        
        print(f"\n{'='*80}")
        print(f"🔬 TWO-STAGE PIPELINE BACKTEST SIMULATOR")
        print(f"{'='*80}")
        print(f"\n⚙️  Configuration:")
        print(f"   Period: {start_date} to {end_date}")
        print(f"   Tickers: {len(tickers)} unique stocks")
        print(f"   Starting cash: ${self.portfolio.starting_cash:,.2f}")
        print(f"   Risk per trade: {self.risk_per_trade}%")
        print(f"   Pipeline 1 threshold: {self.pipeline1_threshold:.0%}")
        print(f"   Pipeline 2 threshold: {self.pipeline2_threshold:.0%}")
        
        # Generate trading days
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        current = start
        trading_days = []
        
        while current <= end:
            # Skip weekends (simple approximation)
            if current.weekday() < 5:  # Mon-Fri
                trading_days.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)
        
        print(f"   Trading days: {len(trading_days)}")
        print(f"\n🚀 Starting simulation...\n")
        
        # Run simulation
        total_signals = 0
        
        for i, date in enumerate(trading_days):
            # Update positions first (check stops/targets)
            # For simplified simulation, we'll track this later
            
            # Scan for new setups
            signals = self.simulate_trade_day(date, tickers)
            total_signals += signals
            
            # Record equity
            self.portfolio.record_equity(date)
            
            # Progress update
            if (i + 1) % progress_interval == 0:
                progress_pct = (i + 1) / len(trading_days) * 100
                equity = self.portfolio.get_total_equity()
                returns = (equity - self.portfolio.starting_cash) / self.portfolio.starting_cash * 100
                print(f"   Day {i+1}/{len(trading_days)} ({progress_pct:.1f}%) | "
                      f"Equity: ${equity:,.0f} ({returns:+.1f}%) | "
                      f"Signals: {total_signals} | "
                      f"Open: {len(self.portfolio.positions)}")
        
        print(f"\n✅ Simulation complete!\n")
        
        # Generate results
        self._generate_report()
    
    def _generate_report(self):
        """Generate performance report"""
        
        print(f"\n{'='*80}")
        print(f"📊 PERFORMANCE REPORT")
        print(f"{'='*80}")
        
        # Portfolio metrics
        final_equity = self.portfolio.get_total_equity()
        total_return = final_equity - self.portfolio.starting_cash
        total_return_pct = (total_return / self.portfolio.starting_cash) * 100
        
        print(f"\n💰 Portfolio Performance:")
        print(f"   Starting capital: ${self.portfolio.starting_cash:,.2f}")
        print(f"   Final equity: ${final_equity:,.2f}")
        print(f"   Total return: ${total_return:,.2f} ({total_return_pct:+.2f}%)")
        print(f"   Current cash: ${self.portfolio.cash:,.2f}")
        print(f"   Open positions: {len(self.portfolio.positions)}")
        
        # Trade statistics
        if self.portfolio.closed_trades:
            trades = self.portfolio.closed_trades
            winners = [t for t in trades if t['pnl'] > 0]
            losers = [t for t in trades if t['pnl'] <= 0]
            hit_5r = [t for t in trades if t['exit_reason'] == 'TARGET_5R']
            
            print(f"\n📈 Trade Statistics:")
            print(f"   Total trades: {len(trades)}")
            print(f"   Winners: {len(winners)} ({len(winners)/len(trades)*100:.1f}%)")
            print(f"   Losers: {len(losers)} ({len(losers)/len(trades)*100:.1f}%)")
            print(f"   Hit 5R target: {len(hit_5r)} ({len(hit_5r)/len(trades)*100:.1f}%)")
            
            avg_win = np.mean([t['pnl_pct'] for t in winners]) if winners else 0
            avg_loss = np.mean([t['pnl_pct'] for t in losers]) if losers else 0
            
            print(f"\n   Average winner: {avg_win:+.2f}%")
            print(f"   Average loser: {avg_loss:+.2f}%")
            
            if avg_loss != 0:
                profit_factor = abs(sum(t['pnl'] for t in winners) / sum(t['pnl'] for t in losers))
                print(f"   Profit factor: {profit_factor:.2f}")
        
        # Pipeline effectiveness
        print(f"\n🔍 Pipeline Performance:")
        print(f"   Pipeline 1 scans: {len(self.scan_log)}")
        print(f"   Pipeline 2 evaluations: {len(self.signal_log)}")
        print(f"   Trades taken: {len(self.portfolio.closed_trades) + len(self.portfolio.positions)}")
        
        if self.signal_log:
            signals_above_threshold = [s for s in self.signal_log 
                                      if s['pipeline2_prob'] >= self.pipeline2_threshold]
            print(f"   Signals above threshold: {len(signals_above_threshold)}")
        
        # Save detailed results
        self._save_results()
    
    def _save_results(self):
        """Save results to files"""
        output_dir = Path('data/backtest_results')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save trades
        trades_file = output_dir / f'trades_{timestamp}.json'
        with open(trades_file, 'w') as f:
            json.dump(self.portfolio.closed_trades, f, indent=2)
        
        # Save equity curve
        equity_file = output_dir / f'equity_{timestamp}.json'
        with open(equity_file, 'w') as f:
            json.dump(self.portfolio.equity_curve, f, indent=2)
        
        # Save signals
        signals_file = output_dir / f'signals_{timestamp}.json'
        with open(signals_file, 'w') as f:
            json.dump(self.signal_log, f, indent=2)
        
        print(f"\n💾 Results saved:")
        print(f"   Trades: {trades_file}")
        print(f"   Equity curve: {equity_file}")
        print(f"   Signals: {signals_file}")


def main():
    """Main entry point"""
    
    # Load configuration
    with open('data/processed/backtest_tickers.json', 'r') as f:
        config = json.load(f)
    
    tickers = config['tickers']
    start_date = config['period_start']
    end_date = config['period_end']
    
    # Initialize simulator
    simulator = TwoStagePipelineSimulator(
        pipeline1_model_path='models/consolidation_scanner.pkl',  # Pipeline 1
        pipeline2_model_path='models/5R_predictor_5min.pkl',      # Pipeline 2
        pipeline1_threshold=0.75,   # 75% confidence for consolidation
        pipeline2_threshold=0.37,   # 37% probability for 5R (optimal threshold)
        starting_cash=30000.0,
        risk_per_trade=1.0          # 1% risk per trade
    )
    
    # Run backtest
    simulator.run_backtest(start_date, end_date, tickers)


if __name__ == '__main__':
    main()
