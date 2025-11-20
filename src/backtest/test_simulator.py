#!/usr/bin/env python3
"""
Quick test of the replay simulator with just a few trades
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.replay_simulator import TradeReplaySimulator
from datetime import datetime

# Load just first 10 trades for testing
trades_file = 'data/processed/ml_5R_labels_filtered.jsonl'

with open(trades_file, 'r') as f:
    all_trades = [json.loads(line) for line in f]

# Filter to Dec 2023 - Oct 2025 and take first 10
start_date = datetime(2023, 12, 1)
end_date = datetime(2025, 10, 31)

filtered_trades = []
for trade in all_trades:
    trade_date = datetime.strptime(trade['entry_date'][:10], '%Y-%m-%d')
    if start_date <= trade_date <= end_date:
        filtered_trades.append(trade)

test_trades = filtered_trades[:10]

print(f"\n🧪 TESTING WITH {len(test_trades)} TRADES")
print(f"{'='*80}\n")

# Run simulator
simulator = TradeReplaySimulator(
    pipeline1_model_path='models/consolidation_detector.pkl',
    pipeline2_model_path='models/5R_predictor_5min.pkl',
    starting_cash=30000.0,
    risk_per_trade_pct=1.0,
    pipeline1_threshold=0.70,
    pipeline2_threshold=0.37
)

simulator.run_simulation(test_trades)

print(f"\n✅ Test complete! If this worked, you can run the full simulation.")
