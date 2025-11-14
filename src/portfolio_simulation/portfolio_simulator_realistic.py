#!/usr/bin/env python3
"""
Portfolio Simulator with REALISTIC constraints for SAR Trading Strategy.

This version adds:
- Max portfolio cap (e.g., $1M) after which we use fixed dollar risk
- Slippage simulation  
- Better reporting
"""

import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import sys
import pandas as pd
import numpy as np

# Add parent directories to path to import Trade_Analyzer
sys.path.append(str(Path(__file__).parent.parent / 'data_processing'))
from Trade_Analyzer import fetch_historical_data


class RealisticPortfolioSimulator:
    """Simulates portfolio with realistic constraints."""
    
    def __init__(
        self, 
        starting_capital=10000.0, 
        risk_per_trade_pct=1.0,  # Default risk (used if trade doesn't specify)
        max_position_pct=40.0,
        max_portfolio_for_scaling=None,  # No cap - scale infinitely with portfolio
        slippage_pct=0.1,  # 0.1% slippage on entries/exits
        use_trade_risk=True,  # Use portfolio_risk_percent from each trade
        trim_r_multiplier=5  # Trim at this R multiple (default 5R)
    ):
        self.starting_capital = starting_capital
        self.cash = starting_capital
        self.positions = {}  # ticker -> position info
        self.default_risk_per_trade_pct = risk_per_trade_pct
        self.max_position_pct = max_position_pct
        self.max_portfolio_for_scaling = max_portfolio_for_scaling
        self.slippage_pct = slippage_pct
        self.use_trade_risk = use_trade_risk
        self.trim_r_multiplier = trim_r_multiplier
        
        self.closed_trades = []
        self.trade_log = []
        
    def get_portfolio_value(self):
        """Current portfolio value (cash + open positions)."""
        position_value = sum(
            pos['shares'] * pos['current_price'] 
            for pos in self.positions.values()
        )
        return self.cash + position_value
    
    def calculate_position_size(self, entry_price, stop_price, portfolio_value, risk_pct=None):
        """
        Calculate position size using Risk / (Entry - Stop) formula.
        
        Args:
            entry_price: Entry price
            stop_price: Stop loss price
            portfolio_value: Current portfolio value
            risk_pct: Risk percent for this specific trade (overrides default)
        """
        # Use trade-specific risk or default
        if risk_pct is None:
            risk_pct = self.default_risk_per_trade_pct
        
        # Apply portfolio cap if set, otherwise use full portfolio value
        if self.max_portfolio_for_scaling is not None:
            capped_portfolio = min(portfolio_value, self.max_portfolio_for_scaling)
        else:
            capped_portfolio = portfolio_value
        
        risk_per_trade = capped_portfolio * (risk_pct / 100)
        price_risk = entry_price - stop_price
        
        if price_risk <= 0:
            return 0
        
        # Calculate shares based on risk
        risk_based_shares = int(risk_per_trade / price_risk)
        
        # Also cap by max position percentage of ACTUAL portfolio
        max_position_value = portfolio_value * (self.max_position_pct / 100)
        max_position_shares = int(max_position_value / entry_price)
        
        shares = min(risk_based_shares, max_position_shares)
        return max(shares, 0)
    
    def apply_slippage(self, price, is_buy=True):
        """Apply slippage to entry/exit prices."""
        if is_buy:
            # Pay more on entries
            return price * (1 + self.slippage_pct / 100)
        else:
            # Get less on exits
            return price * (1 - self.slippage_pct / 100)
    
    def simulate_trade(self, trade):
        """
        Simulate a single trade with SAR strategy rules:
        - Enter at entry_price
        - Stop at initial_stop_loss_price
        - Trim 20% at 5R (target = entry + 5 * risk)
        - Exit remaining if close below 10SMA
        """
        ticker = trade['ticker']
        # Handle datetime parsing - remove microseconds if present
        entry_date_str = trade['entry_date'].replace('Z', '+00:00')
        # Remove fractional seconds if they cause issues
        if '.' in entry_date_str and '+' in entry_date_str:
            parts = entry_date_str.split('.')
            entry_date_str = parts[0] + '+' + entry_date_str.split('+')[1]
        entry_date = datetime.fromisoformat(entry_date_str)
        entry_price = float(trade['entry_price'])
        stop_price = float(trade['initial_stop_loss_price'])
        
        # Get the actual risk percent used for this trade (if available)
        trade_risk_pct = None
        if self.use_trade_risk and 'portfolio_risk_percent' in trade:
            trade_risk_pct = float(trade['portfolio_risk_percent'])
        
        # Apply slippage to entry
        actual_entry_price = self.apply_slippage(entry_price, is_buy=True)
        
        # Calculate position size using trade-specific risk
        portfolio_value = self.get_portfolio_value()
        shares = self.calculate_position_size(actual_entry_price, stop_price, portfolio_value, risk_pct=trade_risk_pct)
        
        if shares == 0:
            self.trade_log.append({
                'ticker': ticker,
                'entry_date': entry_date.isoformat(),
                'status': 'SKIPPED',
                'reason': 'Position size = 0 (risk too small or stop too tight)'
            })
            return
        
        cost = shares * actual_entry_price
        if cost > self.cash:
            self.trade_log.append({
                'ticker': ticker,
                'entry_date': entry_date.isoformat(),
                'status': 'SKIPPED',
                'reason': f'Insufficient cash (need ${cost:.2f}, have ${self.cash:.2f})'
            })
            return
        
        # Enter position
        self.cash -= cost
        risk = actual_entry_price - stop_price
        target_trim = actual_entry_price + (self.trim_r_multiplier * risk)
        
        self.positions[ticker] = {
            'shares': shares,
            'entry_price': actual_entry_price,
            'entry_date': entry_date,
            'stop_price': stop_price,
            'target_trim': target_trim,
            'current_price': actual_entry_price,
            'trimmed': False,
            'initial_shares': shares,
            'cost_basis': cost
        }
        
        # Fetch historical data to simulate day-by-day
        # Scan for 45 trading days (~2 months)
        end_date = entry_date + timedelta(days=90)
        bars = fetch_historical_data(ticker, entry_date, end_date)
        
        if bars is None or len(bars) == 0:
            # Can't simulate, close at entry (scratch trade)
            self.cash += shares * actual_entry_price
            del self.positions[ticker]
            self.trade_log.append({
                'ticker': ticker,
                'entry_date': entry_date.isoformat(),
                'status': 'DATA_ERROR',
                'reason': 'No historical data available'
            })
            return
        
        # Filter to bars after entry (bars are list of dicts with 'date' key)
        bars = [b for b in bars if datetime.strptime(b['date'], '%Y-%m-%d') > entry_date.replace(tzinfo=None)]
        
        if len(bars) == 0:
            # No data after entry, close at entry
            self.cash += shares * actual_entry_price
            del self.positions[ticker]
            self.trade_log.append({
                'ticker': ticker,
                'entry_date': entry_date.isoformat(),
                'status': 'NO_DATA_AFTER_ENTRY'
            })
            return
        
        # Limit to 45 trading days
        bars = bars[:45]
        
        # Simulate day by day
        for bar in bars:
            bar_date = datetime.strptime(bar['date'], '%Y-%m-%d')
            row = bar  # bar is already a dict
            pos = self.positions.get(ticker)
            if not pos:
                break  # Position already closed
            
            pos['current_price'] = row['Close']
            
            # Check for trim target hit (for trimming)
            if not pos['trimmed'] and row['High'] >= pos['target_trim']:
                # Trim 20% at target R
                trim_shares = int(pos['shares'] * 0.20)
                if trim_shares > 0:
                    trim_price = self.apply_slippage(pos['target_trim'], is_buy=False)
                    trim_proceeds = trim_shares * trim_price
                    self.cash += trim_proceeds
                    pos['shares'] -= trim_shares
                    pos['trimmed'] = True
                    
                    # Move stop to breakeven
                    pos['stop_price'] = pos['entry_price']
                    
                    self.trade_log.append({
                        'ticker': ticker,
                        'date': bar_date.isoformat(),
                        'action': f'TRIM_{self.trim_r_multiplier}R',
                        'shares': trim_shares,
                        'price': trim_price,
                        'proceeds': trim_proceeds
                    })
            
            # Check for stop loss (Low of day < stop)
            if row['Low'] < pos['stop_price']:
                # Stopped out
                exit_price = self.apply_slippage(pos['stop_price'], is_buy=False)
                proceeds = pos['shares'] * exit_price
                self.cash += proceeds
                
                profit = proceeds - (pos['shares'] * pos['entry_price'])
                pct_return = (profit / (pos['shares'] * pos['entry_price'])) * 100
                
                self.closed_trades.append({
                    'ticker': ticker,
                    'entry_date': pos['entry_date'].isoformat(),
                    'exit_date': bar_date.isoformat(),
                    'exit_type': 'STOP_LOSS',
                    'shares': pos['initial_shares'],
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'profit': profit,
                    'return_pct': pct_return,
                    'trimmed': pos['trimmed']
                })
                
                del self.positions[ticker]
                break
            
            # Check for 10SMA exit (Close < 10SMA)
            if 'SMA_10' in row or '10sma' in row:
                sma_val = row.get('10sma') or row.get('SMA_10')
                if sma_val is not None and not pd.isna(sma_val):
                    if row['Close'] < sma_val:
                        # Exit at close
                        exit_price = self.apply_slippage(row['Close'], is_buy=False)
                        proceeds = pos['shares'] * exit_price
                        self.cash += proceeds
                        
                        profit = proceeds - (pos['shares'] * pos['entry_price'])
                        pct_return = (profit / (pos['shares'] * pos['entry_price'])) * 100
                        
                        self.closed_trades.append({
                            'ticker': ticker,
                            'entry_date': pos['entry_date'].isoformat(),
                            'exit_date': bar_date.isoformat(),
                            'exit_type': 'SMA_EXIT',
                            'shares': pos['initial_shares'],
                            'entry_price': pos['entry_price'],
                            'exit_price': exit_price,
                            'profit': profit,
                            'return_pct': pct_return,
                            'trimmed': pos['trimmed']
                        })
                        
                        del self.positions[ticker]
                        break
        
        # If still in position after scan window, close at last close
        if ticker in self.positions:
            pos = self.positions[ticker]
            last_price = bars[-1]['Close'] if len(bars) > 0 else pos['entry_price']
            exit_price = self.apply_slippage(last_price, is_buy=False)
            proceeds = pos['shares'] * exit_price
            self.cash += proceeds
            
            profit = proceeds - (pos['shares'] * pos['entry_price'])
            pct_return = (profit / (pos['shares'] * pos['entry_price'])) * 100
            
            self.closed_trades.append({
                'ticker': ticker,
                'entry_date': pos['entry_date'].isoformat(),
                'exit_date': bars[-1]['date'] if len(bars) > 0 else 'N/A',
                'exit_type': 'END_OF_WINDOW',
                'shares': pos['initial_shares'],
                'entry_price': pos['entry_price'],
                'exit_price': exit_price,
                'profit': profit,
                'return_pct': pct_return,
                'trimmed': pos['trimmed']
            })
            
            del self.positions[ticker]
    
    def run_simulation(self, trades_file):
        """Run simulation on all trades from JSONL file."""
        trades = []
        with open(trades_file, 'r') as f:
            for line in f:
                if line.strip():
                    trades.append(json.loads(line))
        
        print(f"Running REALISTIC portfolio simulation on {len(trades)} trades...")
        print(f"Starting capital: ${self.starting_capital:,.2f}")
        print(f"Default risk per trade: {self.default_risk_per_trade_pct}%")
        print(f"Using trade-specific risk: {self.use_trade_risk}")
        print(f"Max position: {self.max_position_pct}%")
        if self.max_portfolio_for_scaling:
            print(f"Portfolio cap for scaling: ${self.max_portfolio_for_scaling:,.2f}")
        else:
            print(f"Portfolio cap for scaling: NONE (unlimited growth)")
        print(f"Slippage: {self.slippage_pct}%")
        print()
        
        for i, trade in enumerate(trades, 1):
            if i % 50 == 0:
                portfolio_value = self.get_portfolio_value()
                print(f"Processing trade {i}/{len(trades)}... Portfolio: ${portfolio_value:,.2f}")
            
            self.simulate_trade(trade)
        
        print("\nSimulation complete!")
        print()
        
    def print_results(self):
        """Print detailed results."""
        final_value = self.get_portfolio_value()
        total_return = ((final_value - self.starting_capital) / self.starting_capital) * 100
        
        print("=" * 70)
        print("REALISTIC PORTFOLIO SIMULATION RESULTS")
        print("=" * 70)
        print()
        print(f"Starting Capital: ${self.starting_capital:,.2f}")
        print(f"Ending Capital:   ${final_value:,.2f}")
        print(f"Total Profit:     ${final_value - self.starting_capital:,.2f}")
        print(f"Total Return:     {total_return:,.2f}%")
        print()
        
        if len(self.closed_trades) == 0:
            print("No closed trades to analyze.")
            return
        
        winners = [t for t in self.closed_trades if t['profit'] > 0]
        losers = [t for t in self.closed_trades if t['profit'] <= 0]
        trimmed_trades = [t for t in self.closed_trades if t.get('trimmed')]
        
        print(f"Total Closed Trades: {len(self.closed_trades)}")
        print(f"  Winners: {len(winners)} ({len(winners)/len(self.closed_trades)*100:.1f}%)")
        print(f"  Losers:  {len(losers)} ({len(losers)/len(self.closed_trades)*100:.1f}%)")
        print(f"  Trimmed at 5R: {len(trimmed_trades)}")
        print()
        
        if winners:
            avg_win = sum(t['profit'] for t in winners) / len(winners)
            avg_win_pct = sum(t['return_pct'] for t in winners) / len(winners)
            max_win = max(winners, key=lambda t: t['profit'])
            print(f"Average Win: ${avg_win:,.2f} ({avg_win_pct:.1f}%)")
            print(f"Largest Win: ${max_win['profit']:,.2f} ({max_win['ticker']} on {max_win['entry_date'][:10]})")
            print()
        
        if losers:
            avg_loss = sum(t['profit'] for t in losers) / len(losers)
            avg_loss_pct = sum(t['return_pct'] for t in losers) / len(losers)
            max_loss = min(losers, key=lambda t: t['profit'])
            print(f"Average Loss: ${avg_loss:,.2f} ({avg_loss_pct:.1f}%)")
            print(f"Largest Loss: ${max_loss['profit']:,.2f} ({max_loss['ticker']} on {max_loss['entry_date'][:10]})")
            print()
        
        # Exit type breakdown
        exit_types = {}
        for trade in self.closed_trades:
            exit_type = trade['exit_type']
            exit_types[exit_type] = exit_types.get(exit_type, 0) + 1
        
        print("Exit Type Breakdown:")
        for exit_type, count in sorted(exit_types.items()):
            print(f"  {exit_type}: {count}")
        print()
        
        # Save detailed results
        results = {
            'summary': {
                'starting_capital': self.starting_capital,
                'ending_capital': final_value,
                'total_return_pct': total_return,
                'total_trades': len(self.closed_trades),
                'winners': len(winners),
                'losers': len(losers),
                'win_rate': len(winners)/len(self.closed_trades)*100 if self.closed_trades else 0
            },
            'closed_trades': self.closed_trades,
            'trade_log': self.trade_log
        }
        
        output_file = 'realistic_portfolio_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Detailed results saved to: {output_file}")


if __name__ == '__main__':
    import pandas as pd
    
    parser = argparse.ArgumentParser(description='Run realistic portfolio simulation with SAR strategy')
    parser.add_argument('--starting-capital', type=float, default=10000.0,
                      help='Starting capital (default: 10000)')
    parser.add_argument('--risk-pct', type=float, default=1.0,
                      help='Risk per trade as percentage (default: 1.0)')
    parser.add_argument('--max-position-pct', type=float, default=40.0,
                      help='Max position size as percentage (default: 40.0)')
    parser.add_argument('--portfolio-cap', type=float, default=None,
                      help='Cap portfolio size for position scaling (default: None for unlimited)')
    parser.add_argument('--slippage', type=float, default=0.1,
                      help='Slippage percentage (default: 0.1)')
    parser.add_argument('--trades-file', type=str, default='ml_5R_labels.jsonl',
                      help='Path to trades JSONL file')
    
    args = parser.parse_args()
    
    # Navigate to workspace root
    script_dir = Path(__file__).parent.parent.parent
    trades_path = script_dir / args.trades_file
    
    if not trades_path.exists():
        print(f"Error: Trades file not found: {trades_path}")
        sys.exit(1)
    
    simulator = RealisticPortfolioSimulator(
        starting_capital=args.starting_capital,
        risk_per_trade_pct=args.risk_pct,
        max_position_pct=args.max_position_pct,
        max_portfolio_for_scaling=args.portfolio_cap,
        slippage_pct=args.slippage
    )
    
    simulator.run_simulation(trades_path)
    simulator.print_results()
