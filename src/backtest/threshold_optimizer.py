"""
Threshold Optimizer - Test different Pipeline 1 & 2 thresholds without re-running simulation

This analyzes the completed simulation log to see what would happen with different thresholds.
"""

import re
from collections import defaultdict

def parse_simulation_log(log_file):
    """Extract all consolidation detections and trade outcomes from log"""
    consolidations = []  # All stocks that passed Pipeline 1 each day
    trades = []  # All trades that were actually taken
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    current_date = None
    current_watchlist = []
    
    for i, line in enumerate(lines):
        # Extract date
        date_match = re.search(r'\[Day \d+/\d+\] (\d{4}-\d{2}-\d{2})', line)
        if date_match:
            current_date = date_match.group(1)
            current_watchlist = []
        
        # Extract watchlist (these passed Pipeline 1)
        if 'Watchlist:' in line and current_date:
            # Parse tickers from watchlist line
            watchlist_str = line.split('Watchlist:')[1].strip()
            # Remove "+ X more" suffix
            watchlist_str = re.sub(r'\s*\+\s*\d+\s*more.*', '', watchlist_str)
            tickers = [t.strip() for t in watchlist_str.split(',')]
            current_watchlist = [t for t in tickers if t and not t.startswith('+')]
        
        # Extract trade entries (these passed Pipeline 2 and triggered)
        entry_match = re.search(r'🚨 (\w+) at (\d{2}:\d{2})', line)
        if entry_match and current_date:
            ticker = entry_match.group(1)
            time = entry_match.group(2)
            
            # Get P2 probability from next few lines
            p2_prob = None
            for j in range(i+1, min(i+5, len(lines))):
                prob_match = re.search(r'P2 \(pre-screened\): ([\d.]+)%', lines[j])
                if prob_match:
                    p2_prob = float(prob_match.group(1)) / 100
                    break
            
            # Get outcome from trade list at end
            trades.append({
                'date': current_date,
                'ticker': ticker,
                'time': time,
                'p2_prob': p2_prob,
                'outcome': None,  # Will fill from final results
                'pnl': None
            })
    
    # Parse final results for outcomes
    in_results = False
    for line in lines:
        if '📋 All Trades:' in line:
            in_results = True
            continue
        
        if in_results:
            # Parse trade line: "2025-10-03 09:35 GPRO   $   2.38 → $   2.16 (exit: 2025-10-06) P&L: $   -158 (-9.1%) | STOPPED_OUT"
            trade_match = re.search(
                r'(\d{4}-\d{2}-\d{2})\s+\d{2}:\d{2}\s+(\w+)\s+.*P&L:\s+\$\s*([-\d]+)',
                line
            )
            if trade_match:
                date = trade_match.group(1)
                ticker = trade_match.group(2)
                pnl = int(trade_match.group(3))
                
                # Find matching trade
                for trade in trades:
                    if trade['date'] == date and trade['ticker'] == ticker:
                        trade['pnl'] = pnl
                        trade['outcome'] = 'WIN' if pnl > 0 else 'LOSS'
                        break
    
    return trades


def test_thresholds(trades):
    """Test different Pipeline 2 threshold combinations"""
    print("\n" + "="*80)
    print("THRESHOLD OPTIMIZATION ANALYSIS")
    print("="*80)
    print(f"\nTotal trades in log: {len(trades)}")
    print(f"With P2 probability data: {sum(1 for t in trades if t['p2_prob'] is not None)}")
    print(f"With outcome data: {sum(1 for t in trades if t['outcome'] is not None)}\n")
    
    # Test different Pipeline 2 thresholds
    thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    
    print("\n" + "="*80)
    print("PIPELINE 2 THRESHOLD TESTING (keeping Pipeline 1 at 70%)")
    print("="*80)
    print(f"\n{'P2 Thresh':<12} {'Trades':<8} {'Winners':<10} {'Win Rate':<10} {'Total P&L':<12} {'Avg P&L':<10}")
    print("-" * 80)
    
    for threshold in thresholds:
        # Filter trades by threshold
        filtered_trades = [t for t in trades if t['p2_prob'] and t['p2_prob'] >= threshold and t['outcome']]
        
        if not filtered_trades:
            print(f"{threshold*100:>3.0f}%         0        -          -          -            -")
            continue
        
        winners = [t for t in filtered_trades if t['outcome'] == 'WIN']
        total_pnl = sum(t['pnl'] for t in filtered_trades)
        win_rate = len(winners) / len(filtered_trades) * 100
        avg_pnl = total_pnl / len(filtered_trades)
        
        print(f"{threshold*100:>3.0f}%         {len(filtered_trades):<8} {len(winners):<10} {win_rate:>6.1f}%     ${total_pnl:>7,}      ${avg_pnl:>6.0f}")
    
    # Show trade distribution by P2 probability
    print("\n" + "="*80)
    print("P2 PROBABILITY DISTRIBUTION")
    print("="*80)
    
    prob_ranges = [(0.35, 0.45), (0.45, 0.55), (0.55, 0.65), (0.65, 0.75), (0.75, 1.0)]
    
    print(f"\n{'P2 Range':<15} {'Trades':<8} {'Winners':<10} {'Win Rate':<10} {'Avg P&L':<10}")
    print("-" * 80)
    
    for low, high in prob_ranges:
        range_trades = [t for t in trades 
                       if t['p2_prob'] and t['outcome'] and low <= t['p2_prob'] < high]
        
        if not range_trades:
            continue
        
        winners = [t for t in range_trades if t['outcome'] == 'WIN']
        win_rate = len(winners) / len(range_trades) * 100
        avg_pnl = sum(t['pnl'] for t in range_trades) / len(range_trades)
        
        print(f"{low*100:.0f}%-{high*100:.0f}%        {len(range_trades):<8} {len(winners):<10} {win_rate:>6.1f}%     ${avg_pnl:>6.0f}")


def main():
    import sys
    
    log_file = 'oct_simulation_corrected.log'
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    
    print(f"Analyzing: {log_file}")
    
    trades = parse_simulation_log(log_file)
    test_thresholds(trades)
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)
    print("• Higher P2 threshold = fewer trades but better quality")
    print("• Lower P2 threshold = more trades but lower win rate")
    print("• Look for the threshold with best Total P&L or Win Rate")
    print("• Current setting: P2 threshold = 45%")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
