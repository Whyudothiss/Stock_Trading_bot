#!/usr/bin/env python3
"""
Portfolio Simulator with 3R TRIMMING (instead of 5R).

This version trims at 3R instead of 5R and filters out split-error trades.
"""

import json
import sys
from pathlib import Path

# Import the realistic simulator
sys.path.append(str(Path(__file__).parent))
from portfolio_simulator_realistic import RealisticPortfolioSimulator

def filter_split_errors(trades_file, max_return_pct=500):
    """
    Pre-filter trades to exclude those likely to have split errors.
    We can't know the return beforehand, so we'll run a quick check.
    """
    print(f"Loading trades from: {trades_file}")
    trades = []
    with open(trades_file, 'r') as f:
        for line in f:
            if line.strip():
                trades.append(json.loads(line))
    
    print(f"Total trades loaded: {len(trades)}")
    
    # Known tickers with split issues (from previous analysis)
    split_error_tickers = {
        'WHLR', 'JTAI', 'PHUN', 'IPW', 'TPET', 
        'KULR', 'WOLF', 'DUO', 'GOEV'
    }
    
    filtered_trades = [t for t in trades if t['ticker'] not in split_error_tickers]
    excluded = len(trades) - len(filtered_trades)
    
    print(f"Filtered out {excluded} trades from known split-error tickers")
    print(f"Remaining trades: {len(filtered_trades)}")
    print()
    
    return filtered_trades

if __name__ == '__main__':
    import argparse
    import pandas as pd
    
    parser = argparse.ArgumentParser(description='Run portfolio simulation excluding split-error trades')
    parser.add_argument('--starting-capital', type=float, default=10000.0,
                      help='Starting capital (default: 10000)')
    parser.add_argument('--risk-pct', type=float, default=1.0,
                      help='Default risk per trade as percentage (default: 1.0)')
    parser.add_argument('--max-position-pct', type=float, default=40.0,
                      help='Max position size as percentage (default: 40.0)')
    parser.add_argument('--portfolio-cap', type=float, default=None,
                      help='Cap portfolio size for position scaling (default: None for unlimited)')
    parser.add_argument('--slippage', type=float, default=0.1,
                      help='Slippage percentage (default: 0.1)')
    parser.add_argument('--trades-file', type=str, default='ml_labels.jsonl',
                      help='Path to trades JSONL file (default: ml_labels.jsonl for 3R data)')
    
    args = parser.parse_args()
    
    # Navigate to workspace root
    script_dir = Path(__file__).parent.parent.parent
    trades_path = script_dir / args.trades_file
    
    if not trades_path.exists():
        print(f"Error: Trades file not found: {trades_path}")
        sys.exit(1)
    
    # Filter out split-error trades
    filtered_trades = filter_split_errors(trades_path)
    
    # Save filtered trades to temp file
    temp_file = script_dir / 'ml_3R_labels_filtered.jsonl'
    with open(temp_file, 'w') as f:
        for trade in filtered_trades:
            f.write(json.dumps(trade) + '\n')
    
    print(f"Saved filtered trades to: {temp_file}")
    print()
    
    # Run simulation on filtered trades with 3R trimming
    simulator = RealisticPortfolioSimulator(
        starting_capital=args.starting_capital,
        risk_per_trade_pct=args.risk_pct,
        max_position_pct=args.max_position_pct,
        max_portfolio_for_scaling=args.portfolio_cap,
        slippage_pct=args.slippage,
        use_trade_risk=True,
        trim_r_multiplier=3  # Trim at 3R instead of 5R
    )
    
    simulator.run_simulation(temp_file)
    simulator.print_results()
    
    print("\n" + "=" * 80)
    print("NOTE: This simulation TRIMS AT 3R (instead of 5R) and excludes")
    print("9 tickers with known stock split data errors:")
    print("WHLR, JTAI, PHUN, IPW, TPET, KULR, WOLF, DUO, GOEV")
    print("=" * 80)
