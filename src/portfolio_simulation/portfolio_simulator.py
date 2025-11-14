"""
Portfolio Simulator for SAR Trading Strategy
Simulates portfolio performance using actual trade data with realistic position sizing and exits.
"""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent to path to import Trade_Analyzer
sys.path.insert(0, str(Path(__file__).parent))
from Trade_Analyzer import fetch_historical_data


class PortfolioSimulator:
    def __init__(self, starting_capital=10000, risk_per_trade_pct=1.0, max_position_pct=40.0):
        """
        Initialize portfolio simulator.
        
        Args:
            starting_capital: Starting cash amount
            risk_per_trade_pct: Percentage of portfolio to risk per trade (default 1%)
            max_position_pct: Maximum position size as % of portfolio (default 40%)
        """
        self.starting_capital = starting_capital
        self.cash = starting_capital
        self.risk_per_trade_pct = risk_per_trade_pct / 100.0  # Convert to decimal
        self.max_position_pct = max_position_pct / 100.0  # Convert to decimal
        
        # Track positions and results
        self.positions = []  # Active positions
        self.closed_trades = []  # Completed trades with P/L
        self.trade_log = []  # Detailed log of all actions
        
    def calculate_position_size(self, entry_price, stop_price, portfolio_value):
        """Calculate number of shares based on risk management rules."""
        risk_dollar = portfolio_value * self.risk_per_trade_pct
        price_risk_per_share = entry_price - stop_price
        
        if price_risk_per_share <= 0:
            return 0
        
        # Calculate shares based on risk
        shares = int(risk_dollar / price_risk_per_share)
        
        # Apply max position size constraint (e.g. 40% of portfolio)
        max_position_value = portfolio_value * self.max_position_pct
        max_shares_by_position = int(max_position_value / entry_price)
        
        # Take the smaller of the two
        shares = min(shares, max_shares_by_position)
        
        return max(shares, 0)  # Never negative
    
    def simulate_trade(self, trade):
        """
        Simulate a single trade using historical bar data.
        
        Args:
            trade: Trade dict from ml_5R_labels.jsonl
            
        Returns:
            dict with trade results (P/L, exit reason, etc.)
        """
        ticker = trade['ticker']
        entry_date_str = str(trade['entry_date']).split('T')[0]
        entry_price = trade['entry_price']
        stop_loss = trade['initial_stop_loss_price']
        ml_label = trade['ml_label']
        
        # Skip data errors
        if ml_label == 'DATA_ERROR':
            return {
                'ticker': ticker,
                'entry_date': entry_date_str,
                'status': 'SKIPPED',
                'reason': 'DATA_ERROR',
                'pnl': 0,
                'pnl_pct': 0
            }
        
        # Calculate position size based on current portfolio value
        portfolio_value = self.cash
        shares = self.calculate_position_size(entry_price, stop_loss, portfolio_value)
        
        if shares == 0:
            return {
                'ticker': ticker,
                'entry_date': entry_date_str,
                'status': 'SKIPPED',
                'reason': 'INSUFFICIENT_CAPITAL',
                'pnl': 0,
                'pnl_pct': 0
            }
        
        # Calculate position cost
        position_cost = shares * entry_price
        
        # Check if we have enough cash
        if position_cost > self.cash:
            # Scale down shares to fit available cash
            shares = int(self.cash / entry_price)
            position_cost = shares * entry_price
            
            if shares == 0:
                return {
                    'ticker': ticker,
                    'entry_date': entry_date_str,
                    'status': 'SKIPPED',
                    'reason': 'INSUFFICIENT_CAPITAL',
                    'pnl': 0,
                    'pnl_pct': 0
                }
        
        # Deduct position cost from cash
        self.cash -= position_cost
        
        # Fetch historical data for this trade
        try:
            entry_dt = datetime.strptime(entry_date_str, '%Y-%m-%d')
            end_dt = (entry_dt + timedelta(days=90)).strftime('%Y-%m-%d')
            bars = fetch_historical_data(ticker, entry_date_str, end_date_override=end_dt)
        except Exception as e:
            # Return capital if we can't fetch data
            self.cash += position_cost
            return {
                'ticker': ticker,
                'entry_date': entry_date_str,
                'status': 'SKIPPED',
                'reason': f'FETCH_ERROR: {e}',
                'pnl': 0,
                'pnl_pct': 0
            }
        
        if not bars:
            self.cash += position_cost
            return {
                'ticker': ticker,
                'entry_date': entry_date_str,
                'status': 'SKIPPED',
                'reason': 'NO_DATA',
                'pnl': 0,
                'pnl_pct': 0
            }
        
        # Simulate day-by-day price action
        remaining_shares = shares
        trimmed = False
        trimmed_shares = 0
        trimmed_price = 0
        exit_price = None
        exit_date = None
        exit_reason = None
        
        # Calculate 5R target
        risk_per_share = entry_price - stop_loss
        target_5R = entry_price + (5 * risk_per_share)
        
        for bar in bars:
            bar_date = bar['date']
            high = bar['High']
            low = bar['Low']
            close = bar['Close']
            sma_10 = bar['10sma']
            
            # Check if we hit 5R for trimming (only trim once)
            if not trimmed and high is not None and high >= target_5R:
                # Trim 20% at 5R
                trim_shares = int(shares * 0.20)
                if trim_shares > 0:
                    trimmed = True
                    trimmed_shares = trim_shares
                    trimmed_price = target_5R  # Assume we got the target price
                    remaining_shares -= trim_shares
                    
                    # Add trimmed profit back to cash
                    trim_proceeds = trim_shares * trimmed_price
                    self.cash += trim_proceeds
                    
                    # Move stop to breakeven after trimming
                    stop_loss = entry_price
            
            # Check for stop loss (Low < stop_loss)
            if low is not None and low < stop_loss:
                exit_price = stop_loss  # Assume we got stopped at stop price
                exit_date = bar_date
                exit_reason = 'STOP_LOSS'
                break
            
            # Check for 10SMA close (Close < 10SMA)
            if sma_10 is not None and close is not None and close < sma_10:
                exit_price = close
                exit_date = bar_date
                exit_reason = 'SMA_CLOSE'
                break
        
        # If no exit found in scan window, exit at last close
        if exit_price is None and bars:
            last_bar = bars[-1]
            exit_price = last_bar['Close']
            exit_date = last_bar['date']
            exit_reason = 'SCAN_END'
        
        # Calculate P/L
        trim_pnl = trimmed_shares * (trimmed_price - entry_price) if trimmed else 0
        exit_proceeds = remaining_shares * exit_price if exit_price else position_cost
        self.cash += exit_proceeds
        
        total_pnl = trim_pnl + (exit_proceeds - (remaining_shares * entry_price))
        pnl_pct = (total_pnl / position_cost) * 100 if position_cost > 0 else 0
        
        result = {
            'ticker': ticker,
            'entry_date': entry_date_str,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'shares': shares,
            'position_cost': position_cost,
            'trimmed': trimmed,
            'trimmed_shares': trimmed_shares,
            'trimmed_price': trimmed_price if trimmed else None,
            'trim_pnl': trim_pnl,
            'exit_date': exit_date,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'remaining_shares': remaining_shares,
            'exit_proceeds': exit_proceeds,
            'total_pnl': total_pnl,
            'pnl_pct': pnl_pct,
            'status': 'CLOSED'
        }
        
        self.closed_trades.append(result)
        return result
    
    def run_simulation(self, trades_file='ml_5R_labels.jsonl'):
        """Run simulation on all trades from file."""
        print(f"Starting Portfolio Simulation")
        print(f"Starting Capital: ${self.starting_capital:,.2f}")
        print(f"Risk per trade: {self.risk_per_trade_pct * 100:.1f}%")
        print("=" * 70)
        print()
        
        # Load trades
        trades = []
        with open(trades_file, 'r') as f:
            for line in f:
                try:
                    trades.append(json.loads(line))
                except:
                    continue
        
        print(f"Loaded {len(trades)} trades")
        print()
        
        # Simulate each trade
        for i, trade in enumerate(trades, 1):
            if i % 50 == 0:
                print(f"Processing trade {i}/{len(trades)}... Portfolio: ${self.cash:,.2f}")
            
            result = self.simulate_trade(trade)
            self.trade_log.append(result)
        
        # Calculate final statistics
        self.print_results()
    
    def print_results(self):
        """Print final simulation results."""
        print()
        print("=" * 70)
        print("SIMULATION RESULTS")
        print("=" * 70)
        print()
        
        winning_trades = [t for t in self.closed_trades if t['total_pnl'] > 0]
        losing_trades = [t for t in self.closed_trades if t['total_pnl'] < 0]
        breakeven_trades = [t for t in self.closed_trades if t['total_pnl'] == 0]
        skipped_trades = [t for t in self.trade_log if t.get('status') == 'SKIPPED']
        
        total_pnl = sum(t['total_pnl'] for t in self.closed_trades)
        win_rate = len(winning_trades) / len(self.closed_trades) * 100 if self.closed_trades else 0
        
        print(f"Starting Capital:  ${self.starting_capital:,.2f}")
        print(f"Ending Capital:    ${self.cash:,.2f}")
        print(f"Total P/L:         ${total_pnl:,.2f}")
        print(f"Return:            {(self.cash / self.starting_capital - 1) * 100:.2f}%")
        print()
        
        print(f"Total Trades:      {len(self.trade_log)}")
        print(f"  Executed:        {len(self.closed_trades)}")
        print(f"  Skipped:         {len(skipped_trades)}")
        print()
        
        print(f"Win Rate:          {win_rate:.1f}%")
        print(f"  Winners:         {len(winning_trades)}")
        print(f"  Losers:          {len(losing_trades)}")
        print(f"  Breakeven:       {len(breakeven_trades)}")
        print()
        
        if winning_trades:
            avg_win = sum(t['total_pnl'] for t in winning_trades) / len(winning_trades)
            avg_win_pct = sum(t['pnl_pct'] for t in winning_trades) / len(winning_trades)
            max_win = max(winning_trades, key=lambda t: t['total_pnl'])
            print(f"Average Win:       ${avg_win:,.2f} ({avg_win_pct:.1f}%)")
            print(f"Largest Win:       ${max_win['total_pnl']:,.2f} ({max_win['ticker']} on {max_win['entry_date']})")
        
        if losing_trades:
            avg_loss = sum(t['total_pnl'] for t in losing_trades) / len(losing_trades)
            avg_loss_pct = sum(t['pnl_pct'] for t in losing_trades) / len(losing_trades)
            max_loss = min(losing_trades, key=lambda t: t['total_pnl'])
            print(f"Average Loss:      ${avg_loss:,.2f} ({avg_loss_pct:.1f}%)")
            print(f"Largest Loss:      ${max_loss['total_pnl']:,.2f} ({max_loss['ticker']} on {max_loss['entry_date']})")
        
        print()
        
        # Trim statistics
        trimmed_trades = [t for t in self.closed_trades if t['trimmed']]
        if trimmed_trades:
            total_trim_pnl = sum(t['trim_pnl'] for t in trimmed_trades)
            print(f"Trimmed Trades:    {len(trimmed_trades)}")
            print(f"Total Trim P/L:    ${total_trim_pnl:,.2f}")
            print()
        
        # Save detailed results
        output_file = 'portfolio_simulation_results.json'
        with open(output_file, 'w') as f:
            json.dump({
                'starting_capital': self.starting_capital,
                'ending_capital': self.cash,
                'total_pnl': total_pnl,
                'return_pct': (self.cash / self.starting_capital - 1) * 100,
                'win_rate': win_rate,
                'total_trades': len(self.trade_log),
                'executed_trades': len(self.closed_trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'trade_log': self.trade_log
            }, f, indent=2)
        
        print(f"Detailed results saved to {output_file}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Simulate portfolio performance')
    parser.add_argument('--starting-capital', type=float, default=10000, 
                        help='Starting portfolio capital (default: 10000)')
    parser.add_argument('--risk-pct', type=float, default=1.0,
                        help='Risk per trade as percentage (default: 1.0)')
    parser.add_argument('--max-position-pct', type=float, default=40.0,
                        help='Maximum position size as percentage (default: 40.0)')
    parser.add_argument('--trades-file', type=str, default='../../ml_5R_labels.jsonl',
                        help='Path to trades JSONL file')
    
    args = parser.parse_args()
    
    sim = PortfolioSimulator(
        starting_capital=args.starting_capital,
        risk_per_trade_pct=args.risk_pct,
        max_position_pct=args.max_position_pct
    )
    
    sim.run_simulation(trades_file=args.trades_file)
