#!/usr/bin/env python3
"""
Analyze risk per trade to verify losses don't exceed expected amounts
"""

import subprocess
import re
import sys

tickers = ['RGTI', 'ARQQ', 'BITF']

print("\n" + "="*80)
print("TRADE RISK ANALYSIS - Verifying Stop Loss Effectiveness")
print("="*80)
print("\nExpected: Max loss = 0.5% of capital per trade")
print("With 0.5% slippage: Total max loss = ~0.5-0.6% of capital\n")

all_losing_trades = []

for ticker in tickers:
    result = subprocess.run(
        ['python3', 'src/backtest/single_ticker_test.py', ticker],
        capture_output=True, text=True
    )
    
    print("\n" + "="*80)
    print(f"{ticker} - RISK ANALYSIS")
    print("="*80)
    
    lines = result.stdout.split('\n')
    in_trades = False
    trade_num = 0
    
    for i, line in enumerate(lines):
        if 'ALL TRADES:' in line:
            in_trades = True
            continue
        
        if in_trades and '. 2025-' in line:  # Start of trade entry
            trade_num += 1
            # Next line has Entry/Exit
            if i+1 < len(lines):
                entry_line = lines[i+1]
                size_line = lines[i+2] if i+2 < len(lines) else ''
                
                # Extract entry and exit prices
                entry_match = re.search(r'Entry: \$(\d+\.\d+).*Exit: \$(\d+\.\d+)', entry_line)
                # Extract size and P&L
                size_match = re.search(r'Size: (\d+) shares.*P&L: \$?([-\d,]+)\s*\(([-+\d\.]+)%\)', size_line)
                
                if entry_match and size_match:
                    entry = float(entry_match.group(1))
                    exit_price = float(entry_match.group(2))
                    size = int(size_match.group(1))
                    pnl_dollars = float(size_match.group(2).replace(',', ''))
                    pnl_pct = float(size_match.group(3))
                    
                    if pnl_pct < 0:  # Only check losing trades
                        position_value = size * entry
                        actual_loss_pct = abs(pnl_pct)
                        
                        print(f"\n  Trade #{trade_num} - LOSS:")
                        print(f"    Entry: ${entry:.2f} | Exit: ${exit_price:.2f}")
                        print(f"    Size: {size} shares (${position_value:,.2f} position)")
                        print(f"    Actual loss: ${abs(pnl_dollars):,.2f} ({actual_loss_pct:.2f}%)")
                        
                        # Check against expected risk
                        if actual_loss_pct > 4.0:  # If loss > 4%, something is very wrong
                            print(f"    🚨 CRITICAL: Loss {actual_loss_pct:.2f}% >> expected 0.5%!")
                            all_losing_trades.append({
                                'ticker': ticker,
                                'trade': trade_num,
                                'loss_pct': actual_loss_pct,
                                'loss_dollars': abs(pnl_dollars),
                                'entry': entry,
                                'exit': exit_price
                            })
                        elif actual_loss_pct > 1.0:
                            print(f"    ⚠️  WARNING: Loss {actual_loss_pct:.2f}% > 1.0%")
                            all_losing_trades.append({
                                'ticker': ticker,
                                'trade': trade_num,
                                'loss_pct': actual_loss_pct,
                                'loss_dollars': abs(pnl_dollars),
                                'entry': entry,
                                'exit': exit_price
                            })
                        else:
                            print(f"    ✅ Loss within acceptable range")

# Summary
print("\n\n" + "="*80)
print("SUMMARY - Trades with Excessive Losses")
print("="*80)

if all_losing_trades:
    print(f"\nFound {len(all_losing_trades)} trades with losses > 1.0%:\n")
    for trade in sorted(all_losing_trades, key=lambda x: x['loss_pct'], reverse=True):
        print(f"  {trade['ticker']} Trade #{trade['trade']}: {trade['loss_pct']:.2f}% loss (${trade['loss_dollars']:.2f})")
        print(f"    Entry: ${trade['entry']:.2f} → Exit: ${trade['exit']:.2f}")
        
    print("\n⚠️  These losses exceed the expected 0.5% risk per trade!")
    print("     Possible causes:")
    print("     - Stop loss not executing properly")
    print("     - Slippage calculation incorrect")
    print("     - Position sizing issue")
else:
    print("\n✅ All losing trades kept losses within acceptable limits (<1.0%)")
    print("   Stop loss system working correctly!")
