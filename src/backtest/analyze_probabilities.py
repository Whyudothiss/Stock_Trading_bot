#!/usr/bin/env python3
"""
Analyze Pipeline 2 probabilities for all trades to determine optimal threshold
"""

import subprocess
import re

tickers = ['RGTI', 'ARQQ', 'BITF']

print("\n" + "="*80)
print("PIPELINE 2 PROBABILITY ANALYSIS")
print("="*80)
print("\nCurrent threshold: 30%")
print("Goal: Find if higher threshold would improve win rate\n")

all_trades = []

for ticker in tickers:
    result = subprocess.run(
        ['python3', 'src/backtest/single_ticker_test.py', ticker],
        capture_output=True, text=True
    )
    
    lines = result.stdout.split('\n')
    in_trades = False
    trade_num = 0
    
    for i, line in enumerate(lines):
        if 'ALL TRADES:' in line:
            in_trades = True
            continue
        
        if in_trades and '. 2025-' in line:
            trade_num += 1
            if i+1 < len(lines) and i+2 < len(lines):
                entry_line = lines[i+1]
                size_line = lines[i+2]
                prob_line = lines[i+3] if i+3 < len(lines) else ''
                
                # Extract data
                entry_match = re.search(r'Entry: \$(\d+\.\d+).*Exit: \$(\d+\.\d+)', entry_line)
                size_match = re.search(r'Size: (\d+) shares.*P&L: \$?([-\d,]+)\s*\(([-+\d\.]+)%\)', size_line)
                prob_match = re.search(r'Probability: ([\d\.]+)%', prob_line)
                
                if entry_match and size_match and prob_match:
                    entry = float(entry_match.group(1))
                    exit_price = float(entry_match.group(2))
                    pnl_dollars = float(size_match.group(2).replace(',', ''))
                    pnl_pct = float(size_match.group(3))
                    prob = float(prob_match.group(1))
                    
                    is_winner = pnl_pct > 0
                    
                    all_trades.append({
                        'ticker': ticker,
                        'trade_num': trade_num,
                        'prob': prob,
                        'pnl_pct': pnl_pct,
                        'pnl_dollars': pnl_dollars,
                        'is_winner': is_winner,
                        'entry': entry,
                        'exit': exit_price
                    })

# Sort by probability
all_trades.sort(key=lambda x: x['prob'], reverse=True)

print("="*80)
print("ALL TRADES BY PROBABILITY (Highest to Lowest)")
print("="*80)
print(f"{'Ticker':<6} {'Trade':<6} {'Prob':<7} {'Result':<8} {'P&L %':<8} {'P&L $':<10}")
print("-"*80)

for trade in all_trades:
    result = "WIN ✅" if trade['is_winner'] else "LOSS ❌"
    prob_str = f"{trade['prob']:.1f}%"
    pnl_pct_str = f"{trade['pnl_pct']:+.1f}%"
    pnl_dollar_str = f"${trade['pnl_dollars']:+,.0f}"
    
    print(f"{trade['ticker']:<6} #{trade['trade_num']:<5} {prob_str:<7} {result:<8} {pnl_pct_str:<8} {pnl_dollar_str:<10}")

# Analysis by probability threshold
print("\n" + "="*80)
print("THRESHOLD ANALYSIS")
print("="*80)
print(f"{'Threshold':<12} {'Trades':<8} {'Winners':<9} {'Losers':<8} {'Win Rate':<10} {'Avg Win':<10} {'Avg Loss':<10}")
print("-"*80)

thresholds = [30, 35, 40, 45, 50, 55, 60, 65, 70]

for threshold in thresholds:
    filtered = [t for t in all_trades if t['prob'] >= threshold]
    
    if len(filtered) == 0:
        continue
    
    winners = [t for t in filtered if t['is_winner']]
    losers = [t for t in filtered if not t['is_winner']]
    
    win_rate = (len(winners) / len(filtered)) * 100 if filtered else 0
    avg_win = sum(t['pnl_pct'] for t in winners) / len(winners) if winners else 0
    avg_loss = sum(t['pnl_pct'] for t in losers) / len(losers) if losers else 0
    
    status = ""
    if threshold == 30:
        status = " ← Current"
    elif win_rate >= 50:
        status = " ✅ Good"
    
    print(f"{threshold}%{status:<10} {len(filtered):<8} {len(winners):<9} {len(losers):<8} {win_rate:<9.1f}% {avg_win:<9.1f}% {avg_loss:<9.1f}%")

# Detailed analysis
print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

winners = [t for t in all_trades if t['is_winner']]
losers = [t for t in all_trades if not t['is_winner']]

if winners:
    winner_probs = [t['prob'] for t in winners]
    avg_winner_prob = sum(winner_probs) / len(winner_probs)
    min_winner_prob = min(winner_probs)
    max_winner_prob = max(winner_probs)
    
    print(f"\nWinning trades ({len(winners)} trades):")
    print(f"  Average probability: {avg_winner_prob:.1f}%")
    print(f"  Range: {min_winner_prob:.1f}% - {max_winner_prob:.1f}%")

if losers:
    loser_probs = [t['prob'] for t in losers]
    avg_loser_prob = sum(loser_probs) / len(loser_probs)
    min_loser_prob = min(loser_probs)
    max_loser_prob = max(loser_probs)
    
    print(f"\nLosing trades ({len(losers)} trades):")
    print(f"  Average probability: {avg_loser_prob:.1f}%")
    print(f"  Range: {min_loser_prob:.1f}% - {max_loser_prob:.1f}%")

# BITF specific
print("\n" + "="*80)
print("BITF ANALYSIS (Most problematic)")
print("="*80)

bitf_trades = [t for t in all_trades if t['ticker'] == 'BITF']
bitf_winners = [t for t in bitf_trades if t['is_winner']]
bitf_losers = [t for t in bitf_trades if not t['is_winner']]

print(f"\nBITF: {len(bitf_trades)} trades, {len(bitf_winners)} winners, {len(bitf_losers)} losers")
print(f"Win rate: {(len(bitf_winners)/len(bitf_trades)*100):.1f}%")

if bitf_winners:
    print(f"\nBITF Winners (prob range: {min(t['prob'] for t in bitf_winners):.1f}% - {max(t['prob'] for t in bitf_winners):.1f}%):")
    for t in bitf_winners:
        print(f"  Trade #{t['trade_num']}: {t['prob']:.1f}% prob → {t['pnl_pct']:+.1f}%")

if bitf_losers:
    print(f"\nBITF Losers (prob range: {min(t['prob'] for t in bitf_losers):.1f}% - {max(t['prob'] for t in bitf_losers):.1f}%):")
    for t in bitf_losers:
        print(f"  Trade #{t['trade_num']}: {t['prob']:.1f}% prob → {t['pnl_pct']:.1f}%")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

# Find optimal threshold
best_threshold = 30
best_win_rate = 0
for threshold in thresholds:
    filtered = [t for t in all_trades if t['prob'] >= threshold]
    if len(filtered) >= 5:  # Need at least 5 trades
        win_rate = (len([t for t in filtered if t['is_winner']]) / len(filtered)) * 100
        if win_rate > best_win_rate:
            best_win_rate = win_rate
            best_threshold = threshold

print(f"""
Current threshold: 30%
  - Total trades: {len(all_trades)}
  - Win rate: {(len(winners)/len(all_trades)*100):.1f}%

Optimal threshold: {best_threshold}%
  - Total trades: {len([t for t in all_trades if t['prob'] >= best_threshold])}
  - Win rate: {best_win_rate:.1f}%

{'⚠️  RECOMMENDATION: Raise threshold to ' + str(best_threshold) + '%' if best_threshold > 30 else '✅ Current threshold (30%) is optimal'}
""")
