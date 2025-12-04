#!/usr/bin/env python3
"""
Threshold Optimization Script for October 2025

Tests multiple combinations of Pipeline 1 and Pipeline 2 thresholds
to find the optimal settings that maximize returns.

Strategy:
1. Grid search across threshold combinations
2. Run full simulation for each combination
3. Track key metrics (return, win rate, profit factor)
4. Output best thresholds
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
import json
import time

# Import the simulator
from backtest.simulation_twelve_fresh import FreshSimulator


def test_threshold_combination(tickers, p1_threshold, p2_threshold, starting_cash=30000):
    """
    Run simulation with specific threshold combination
    
    Returns:
        dict with performance metrics
    """
    print(f"\n{'='*80}")
    print(f"Testing P1={p1_threshold:.2f}, P2={p2_threshold:.2f}")
    print(f"{'='*80}")
    
    # Create simulator with custom thresholds
    sim = FreshSimulator(starting_cash=starting_cash, random_seed=42)
    sim.pipeline1_threshold = p1_threshold
    sim.pipeline2_threshold = p2_threshold
    
    # Run simulation (suppress verbose output)
    import io
    import contextlib
    
    # Capture output to reduce clutter
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        sim.run_simulation(tickers)
    
    # Calculate metrics
    if len(sim.trades) == 0:
        return {
            'p1_threshold': p1_threshold,
            'p2_threshold': p2_threshold,
            'total_return': 0,
            'return_pct': 0,
            'num_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'max_drawdown': 0
        }
    
    winners = [t for t in sim.trades if t['pnl'] > 0]
    losers = [t for t in sim.trades if t['pnl'] < 0]
    
    total_return = sim.cash - starting_cash
    return_pct = (total_return / starting_cash) * 100
    win_rate = len(winners) / len(sim.trades) * 100 if len(sim.trades) > 0 else 0
    
    avg_win = np.mean([t['pnl'] for t in winners]) if winners else 0
    avg_loss = np.mean([abs(t['pnl']) for t in losers]) if losers else 0
    
    total_gains = sum([t['pnl'] for t in winners]) if winners else 0
    total_losses = abs(sum([t['pnl'] for t in losers])) if losers else 0
    profit_factor = total_gains / total_losses if total_losses > 0 else 0
    
    # Calculate max drawdown
    equity_curve = [starting_cash]
    for trade in sim.trades:
        equity_curve.append(equity_curve[-1] + trade['pnl'])
    
    peak = equity_curve[0]
    max_dd = 0
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd:
            max_dd = dd
    
    print(f"✅ Return: ${total_return:,.0f} ({return_pct:+.2f}%) | "
          f"Trades: {len(sim.trades)} | Win Rate: {win_rate:.1f}% | "
          f"Profit Factor: {profit_factor:.2f}")
    
    return {
        'p1_threshold': p1_threshold,
        'p2_threshold': p2_threshold,
        'total_return': total_return,
        'return_pct': return_pct,
        'num_trades': len(sim.trades),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'max_drawdown': max_dd,
        'final_cash': sim.cash
    }


def optimize_thresholds(tickers, starting_cash=30000):
    """
    Run grid search to find optimal thresholds
    
    Tests combinations of:
    - Pipeline 1: 0.60 to 0.85 (step 0.05)
    - Pipeline 2: 0.15 to 0.50 (step 0.05)
    """
    print("="*80)
    print("THRESHOLD OPTIMIZATION - OCTOBER 2025")
    print("="*80)
    print(f"\nStarting Capital: ${starting_cash:,}")
    print(f"Tickers: {len(tickers)}")
    
    # Define threshold ranges
    p1_thresholds = np.arange(0.60, 0.86, 0.05)  # 0.60, 0.65, 0.70, 0.75, 0.80, 0.85
    p2_thresholds = np.arange(0.15, 0.51, 0.05)  # 0.15, 0.20, ..., 0.50
    
    total_combinations = len(p1_thresholds) * len(p2_thresholds)
    
    print(f"\nPipeline 1 thresholds: {list(p1_thresholds)}")
    print(f"Pipeline 2 thresholds: {list(p2_thresholds)}")
    print(f"\nTotal combinations to test: {total_combinations}")
    print(f"Estimated time: ~{total_combinations * 2} minutes")
    
    input("\nPress Enter to start optimization (or Ctrl+C to cancel)...")
    
    results = []
    start_time = time.time()
    
    for i, p1 in enumerate(p1_thresholds):
        for j, p2 in enumerate(p2_thresholds):
            idx = i * len(p2_thresholds) + j + 1
            
            # Calculate time estimates
            if idx > 1:
                elapsed = time.time() - start_time
                avg_time_per_test = elapsed / (idx - 1)
                remaining_tests = total_combinations - idx + 1
                estimated_remaining = avg_time_per_test * remaining_tests
                
                elapsed_min = int(elapsed // 60)
                elapsed_sec = int(elapsed % 60)
                remaining_min = int(estimated_remaining // 60)
                remaining_sec = int(estimated_remaining % 60)
                
                print(f"\n[{idx}/{total_combinations}] Testing combination... | "
                      f"Elapsed: {elapsed_min}m {elapsed_sec}s | "
                      f"Remaining: ~{remaining_min}m {remaining_sec}s")
            else:
                print(f"\n[{idx}/{total_combinations}] Testing combination...")
            
            test_start = time.time()
            result = test_threshold_combination(tickers, p1, p2, starting_cash)
            test_duration = time.time() - test_start
            result['test_duration_sec'] = test_duration
            results.append(result)
    
    # Convert to DataFrame for analysis
    df_results = pd.DataFrame(results)
    
    # Calculate total time
    total_time = time.time() - start_time
    total_min = int(total_time // 60)
    total_sec = int(total_time % 60)
    avg_time = total_time / len(results)
    
    # Sort by return percentage
    df_results = df_results.sort_values('return_pct', ascending=False)
    
    # Save results
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_file = output_dir / f'threshold_optimization_{timestamp}.csv'
    df_results.to_csv(csv_file, index=False)
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    
    print(f"\n⏱️  Total Time: {total_min}m {total_sec}s")
    print(f"    Average per test: {avg_time:.1f}s")
    
    print(f"\n💾 Results saved to: {csv_file}")
    
    # Print top 10 combinations
    print("\n" + "="*80)
    print("TOP 10 THRESHOLD COMBINATIONS")
    print("="*80)
    
    print(f"\n{'Rank':<6} {'P1':<7} {'P2':<7} {'Return %':<12} {'Trades':<8} {'Win %':<8} {'PF':<8} {'Max DD %':<10}")
    print("-"*80)
    
    for idx, row in df_results.head(10).iterrows():
        print(f"{df_results.index.get_loc(idx)+1:<6} "
              f"{row['p1_threshold']:<7.2f} "
              f"{row['p2_threshold']:<7.2f} "
              f"{row['return_pct']:<12.2f} "
              f"{int(row['num_trades']):<8} "
              f"{row['win_rate']:<8.1f} "
              f"{row['profit_factor']:<8.2f} "
              f"{row['max_drawdown']:<10.2f}")
    
    # Print best by different metrics
    print("\n" + "="*80)
    print("BEST THRESHOLDS BY METRIC")
    print("="*80)
    
    best_return = df_results.iloc[0]
    print(f"\n🏆 Best Return: P1={best_return['p1_threshold']:.2f}, P2={best_return['p2_threshold']:.2f}")
    print(f"   Return: {best_return['return_pct']:.2f}% | Trades: {int(best_return['num_trades'])} | Win Rate: {best_return['win_rate']:.1f}%")
    
    best_pf = df_results.loc[df_results['profit_factor'].idxmax()]
    print(f"\n💰 Best Profit Factor: P1={best_pf['p1_threshold']:.2f}, P2={best_pf['p2_threshold']:.2f}")
    print(f"   Profit Factor: {best_pf['profit_factor']:.2f} | Return: {best_pf['return_pct']:.2f}% | Trades: {int(best_pf['num_trades'])}")
    
    best_wr = df_results.loc[df_results['win_rate'].idxmax()]
    print(f"\n🎯 Best Win Rate: P1={best_wr['p1_threshold']:.2f}, P2={best_wr['p2_threshold']:.2f}")
    print(f"   Win Rate: {best_wr['win_rate']:.1f}% | Return: {best_wr['return_pct']:.2f}% | Trades: {int(best_wr['num_trades'])}")
    
    # Filter for reasonable trade count (at least 5 trades)
    df_filtered = df_results[df_results['num_trades'] >= 5]
    
    if len(df_filtered) > 0:
        best_filtered = df_filtered.iloc[0]
        print(f"\n⭐ Best with ≥5 trades: P1={best_filtered['p1_threshold']:.2f}, P2={best_filtered['p2_threshold']:.2f}")
        print(f"   Return: {best_filtered['return_pct']:.2f}% | Trades: {int(best_filtered['num_trades'])} | Win Rate: {best_filtered['win_rate']:.1f}%")
    
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    print(f"\n💡 Recommended thresholds for production:")
    print(f"   Pipeline 1: {best_return['p1_threshold']:.2f}")
    print(f"   Pipeline 2: {best_return['p2_threshold']:.2f}")
    print(f"\n   Expected Performance:")
    print(f"   - Monthly Return: {best_return['return_pct']:.2f}%")
    print(f"   - Win Rate: {best_return['win_rate']:.1f}%")
    print(f"   - Profit Factor: {best_return['profit_factor']:.2f}")
    print(f"   - Max Drawdown: {best_return['max_drawdown']:.2f}%")
    print(f"   - Trades per Month: {int(best_return['num_trades'])}")
    
    print("\n" + "="*80)
    
    return df_results


def quick_test(tickers, starting_cash=30000):
    """
    Quick test with fewer combinations (faster)
    
    Tests:
    - Pipeline 1: 0.65, 0.70, 0.75, 0.80
    - Pipeline 2: 0.20, 0.30, 0.40, 0.50
    """
    print("="*80)
    print("QUICK THRESHOLD TEST - OCTOBER 2025")
    print("="*80)
    
    p1_thresholds = [0.65, 0.70, 0.75, 0.80]
    p2_thresholds = [0.20, 0.30, 0.40, 0.50]
    
    total_combinations = len(p1_thresholds) * len(p2_thresholds)
    print(f"\nTesting {total_combinations} combinations (quick mode)")
    
    results = []
    start_time = time.time()
    
    for i, p1 in enumerate(p1_thresholds):
        for j, p2 in enumerate(p2_thresholds):
            idx = i * len(p2_thresholds) + j + 1
            
            # Time tracking
            if idx > 1:
                elapsed = time.time() - start_time
                avg_time = elapsed / (idx - 1)
                remaining = avg_time * (total_combinations - idx + 1)
                print(f"\n[{idx}/{total_combinations}] | "
                      f"Elapsed: {int(elapsed//60)}m {int(elapsed%60)}s | "
                      f"Remaining: ~{int(remaining//60)}m {int(remaining%60)}s")
            else:
                print(f"\n[{idx}/{total_combinations}]")
            
            result = test_threshold_combination(tickers, p1, p2, starting_cash)
            results.append(result)
    
    total_time = time.time() - start_time
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('return_pct', ascending=False)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\n⏱️  Total Time: {int(total_time//60)}m {int(total_time%60)}s")
    
    print(f"\n{'P1':<7} {'P2':<7} {'Return %':<12} {'Trades':<8} {'Win %':<8} {'PF':<8}")
    print("-"*60)
    
    for _, row in df_results.iterrows():
        print(f"{row['p1_threshold']:<7.2f} "
              f"{row['p2_threshold']:<7.2f} "
              f"{row['return_pct']:<12.2f} "
              f"{int(row['num_trades']):<8} "
              f"{row['win_rate']:<8.1f} "
              f"{row['profit_factor']:<8.2f}")
    
    best = df_results.iloc[0]
    print(f"\n🏆 Best: P1={best['p1_threshold']:.2f}, P2={best['p2_threshold']:.2f} → {best['return_pct']:.2f}% return")
    
    return df_results


def main():
    """Run threshold optimization"""
    # Load tickers
    tickers = []
    with open('data/processed/unique_traded_jun1_oct31_2025.txt', 'r') as f:
        for line in f:
            ticker = line.strip()
            if ticker:
                tickers.append(ticker)
    
    print(f"Loaded {len(tickers)} tickers\n")
    
    # Ask user for mode
    print("Select optimization mode:")
    print("1. Quick test (16 combinations, ~30 min)")
    print("2. Full optimization (42 combinations, ~80 min)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == '1':
        results = quick_test(tickers)
    elif choice == '2':
        results = optimize_thresholds(tickers)
    else:
        print("Invalid choice. Defaulting to quick test.")
        results = quick_test(tickers)


if __name__ == '__main__':
    main()
