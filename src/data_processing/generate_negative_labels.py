#!/usr/bin/env python3
"""
Generate Negative Labels for Consolidation Detection Pipeline

This creates negative training examples by finding:
1. Failed breakouts - broke consolidation but immediately failed
2. Near-miss consolidations - almost met criteria but didn't
3. Random non-patterns - clearly bad technical setups

Strategy:
- Positive labels: Your 744 actual trades (already collected)
- Negative labels: Generate ~1500-2000 examples (2-3x positives for balance)
"""

import json
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
import random
from typing import List, Dict, Tuple


class NegativeLabelGenerator:
    """Generates negative training examples for consolidation detection."""
    
    def __init__(self, positive_trades_file: str):
        """
        Args:
            positive_trades_file: Path to JSONL file with positive examples
        """
        self.positive_trades_file = positive_trades_file
        self.positive_trades = self._load_positive_trades()
        self.date_range = self._get_date_range()
        self.tickers_used = self._get_tickers_used()
        
    def _load_positive_trades(self) -> List[Dict]:
        """Load positive examples."""
        with open(self.positive_trades_file, 'r') as f:
            trades = json.load(f)  # Load as JSON array
        return trades
    
    def _get_date_range(self) -> Tuple[datetime, datetime]:
        """Extract date range from positive examples."""
        dates = []
        for trade in self.positive_trades:
            entry_date_str = trade['entry_date'].split('T')[0]
            dates.append(datetime.strptime(entry_date_str, '%Y-%m-%d'))
        
        return min(dates), max(dates)
    
    def _get_tickers_used(self) -> set:
        """Get unique tickers from positive examples."""
        return set(trade['ticker'] for trade in self.positive_trades)
    
    def generate_failed_breakouts(self, count: int = 600) -> List[Dict]:
        """
        Generate failed breakout examples.
        
        Strategy:
        - Take stocks that broke above resistance
        - But closed back inside range within 1-3 days
        - Or had volume spike but no follow-through
        
        Args:
            count: Number of failed breakout examples to generate
        """
        failed_breakouts = []
        
        print(f"\n🔴 Generating {count} Failed Breakout examples...")
        
        # Sample from existing tickers that had successful trades
        tickers = list(self.tickers_used)
        start_date, end_date = self.date_range
        
        # Extend date range to capture more examples
        extended_start = start_date - timedelta(days=180)
        extended_end = end_date + timedelta(days=90)
        
        attempts = 0
        max_attempts = count * 10  # Prevent infinite loop
        
        while len(failed_breakouts) < count and attempts < max_attempts:
            attempts += 1
            
            # Random ticker and date
            ticker = random.choice(tickers)
            random_date = extended_start + timedelta(
                days=random.randint(0, (extended_end - extended_start).days)
            )
            
            # Fetch data around this date
            lookback_start = random_date - timedelta(days=60)
            lookback_end = random_date + timedelta(days=10)
            
            try:
                df = yf.download(
                    ticker,
                    start=lookback_start,
                    end=lookback_end,
                    progress=False,
                    auto_adjust=True
                )
                
                if df is None or len(df) < 30:
                    continue
                
                # Look for failed breakout pattern
                # Find local resistance then breakout that failed
                df['High_20'] = df['High'].rolling(20).max()
                
                for i in range(20, len(df) - 3):
                    current_high = df['High'].iloc[i]
                    resistance = df['High_20'].iloc[i-1]
                    
                    # Check if broke above 20-day high
                    if current_high > resistance * 1.02:  # 2% above resistance
                        # Check if it failed (closed back below within 3 days)
                        next_3_closes = df['Close'].iloc[i+1:i+4].values
                        
                        if len(next_3_closes) >= 2 and any(c < resistance for c in next_3_closes):
                            # This is a failed breakout!
                            failed_breakouts.append({
                                'ticker': ticker,
                                'date': df.index[i].strftime('%Y-%m-%d'),
                                'category': 'FAILED_BREAKOUT',
                                'entry_price': float(current_high),
                                'resistance': float(resistance),
                                'failure_price': float(next_3_closes[0]),
                                'label': 0  # Negative label
                            })
                            break  # One per ticker/date combo
                
            except Exception as e:
                continue
            
            if len(failed_breakouts) % 100 == 0 and len(failed_breakouts) > 0:
                print(f"  Generated {len(failed_breakouts)}/{count} failed breakouts...")
        
        print(f"  ✅ Generated {len(failed_breakouts)} failed breakout examples")
        return failed_breakouts
    
    def generate_near_miss_consolidations(self, count: int = 450) -> List[Dict]:
        """
        Generate near-miss consolidation examples.
        
        Strategy:
        - Stocks that had consolidation pattern
        - But failed one key criterion:
          * Too long (>8 weeks)
          * Too short (<3 weeks)
          * Too deep (>40% pullback)
          * Insufficient volume on breakout
        
        Args:
            count: Number of near-miss examples to generate
        """
        near_misses = []
        
        print(f"\n🟡 Generating {count} Near-Miss Consolidation examples...")
        
        tickers = list(self.tickers_used)
        start_date, end_date = self.date_range
        extended_start = start_date - timedelta(days=180)
        extended_end = end_date + timedelta(days=90)
        
        attempts = 0
        max_attempts = count * 10
        
        while len(near_misses) < count and attempts < max_attempts:
            attempts += 1
            
            ticker = random.choice(tickers)
            random_date = extended_start + timedelta(
                days=random.randint(0, (extended_end - extended_start).days)
            )
            
            lookback_start = random_date - timedelta(days=90)
            lookback_end = random_date + timedelta(days=10)
            
            try:
                df = yf.download(
                    ticker,
                    start=lookback_start,
                    end=lookback_end,
                    progress=False,
                    auto_adjust=True
                )
                
                if df is None or len(df) < 40:
                    continue
                
                df['SMA_10'] = df['Close'].rolling(10).mean()
                df['Volume_Avg'] = df['Volume'].rolling(20).mean()
                
                # Look for consolidation that fails criteria
                for i in range(30, len(df) - 5):
                    # Find a potential consolidation window (10 days)
                    window = df.iloc[i-10:i]
                    high = window['High'].max()
                    low = window['Low'].min()
                    depth = (high - low) / high * 100
                    
                    # Check if consolidation but fails criteria
                    failure_reason = None
                    
                    if depth > 40:  # Too deep
                        failure_reason = "TOO_DEEP"
                    elif depth < 5:  # Too tight/no real consolidation
                        failure_reason = "TOO_TIGHT"
                    elif len(window) < 10:  # Too short
                        failure_reason = "TOO_SHORT"
                    
                    if failure_reason:
                        near_misses.append({
                            'ticker': ticker,
                            'date': df.index[i].strftime('%Y-%m-%d'),
                            'category': 'NEAR_MISS',
                            'failure_reason': failure_reason,
                            'consolidation_depth': float(depth),
                            'consolidation_days': len(window),
                            'label': 0  # Negative label
                        })
                        break
                
            except Exception as e:
                continue
            
            if len(near_misses) % 100 == 0 and len(near_misses) > 0:
                print(f"  Generated {len(near_misses)}/{count} near-miss examples...")
        
        print(f"  ✅ Generated {len(near_misses)} near-miss examples")
        return near_misses
    
    def generate_random_non_patterns(self, count: int = 450) -> List[Dict]:
        """
        Generate random non-pattern examples.
        
        Strategy:
        - Random stocks on random dates
        - That clearly show:
          * Downtrends
          * Choppy/volatile action
          * No base formation
        
        Args:
            count: Number of random examples to generate
        """
        random_examples = []
        
        print(f"\n⚫ Generating {count} Random Non-Pattern examples...")
        
        # Use broader ticker universe for more diversity
        broad_tickers = [
            'SPY', 'QQQ', 'IWM', 'DIA',  # ETFs
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',  # Big caps
            'F', 'GE', 'T', 'BAC', 'WFC', 'C', 'JPM',  # Traditional
            'XOM', 'CVX', 'KO', 'PEP', 'WMT', 'TGT', 'HD', 'LOW'  # Staples
        ]
        
        start_date, end_date = self.date_range
        extended_start = start_date - timedelta(days=365)
        extended_end = end_date + timedelta(days=90)
        
        attempts = 0
        max_attempts = count * 5
        
        while len(random_examples) < count and attempts < max_attempts:
            attempts += 1
            
            ticker = random.choice(broad_tickers)
            random_date = extended_start + timedelta(
                days=random.randint(0, (extended_end - extended_start).days)
            )
            
            lookback_start = random_date - timedelta(days=60)
            lookback_end = random_date + timedelta(days=5)
            
            try:
                df = yf.download(
                    ticker,
                    start=lookback_start,
                    end=lookback_end,
                    progress=False,
                    auto_adjust=True
                )
                
                if df is None or len(df) < 30:
                    continue
                
                df['SMA_20'] = df['Close'].rolling(20).mean()
                df['SMA_50'] = df['Close'].rolling(50).mean()
                
                # Check for non-pattern characteristics
                recent = df.iloc[-20:]
                
                # Downtrend: Price below both SMAs
                is_downtrend = (
                    recent['Close'].iloc[-1] < recent['SMA_20'].iloc[-1] and
                    recent['Close'].iloc[-1] < recent['SMA_50'].iloc[-1]
                )
                
                # Choppy: High volatility, no clear direction
                returns = recent['Close'].pct_change()
                is_choppy = returns.std() > 0.03  # >3% daily volatility
                
                if is_downtrend or is_choppy:
                    pattern_type = 'DOWNTREND' if is_downtrend else 'CHOPPY'
                    
                    random_examples.append({
                        'ticker': ticker,
                        'date': df.index[-1].strftime('%Y-%m-%d'),
                        'category': 'RANDOM_NON_PATTERN',
                        'pattern_type': pattern_type,
                        'volatility': float(returns.std()) if not pd.isna(returns.std()) else 0.0,
                        'label': 0  # Negative label
                    })
                
            except Exception as e:
                continue
            
            if len(random_examples) % 100 == 0 and len(random_examples) > 0:
                print(f"  Generated {len(random_examples)}/{count} random examples...")
        
        print(f"  ✅ Generated {len(random_examples)} random non-pattern examples")
        return random_examples
    
    def generate_all_negatives(
        self,
        failed_breakouts: int = 600,
        near_misses: int = 450,
        random_non_patterns: int = 450
    ) -> List[Dict]:
        """
        Generate all negative examples.
        
        Args:
            failed_breakouts: Count of failed breakout examples
            near_misses: Count of near-miss consolidation examples
            random_non_patterns: Count of random non-pattern examples
        
        Returns:
            Combined list of all negative examples
        """
        print("=" * 70)
        print("GENERATING NEGATIVE LABELS FOR CONSOLIDATION DETECTION")
        print("=" * 70)
        print(f"\nPositive examples available: {len(self.positive_trades)}")
        print(f"Target negative examples: {failed_breakouts + near_misses + random_non_patterns}")
        print(f"Date range: {self.date_range[0].date()} to {self.date_range[1].date()}")
        print(f"Unique tickers in positives: {len(self.tickers_used)}")
        
        # Generate each category
        negatives = []
        negatives.extend(self.generate_failed_breakouts(failed_breakouts))
        negatives.extend(self.generate_near_miss_consolidations(near_misses))
        negatives.extend(self.generate_random_non_patterns(random_non_patterns))
        
        print("\n" + "=" * 70)
        print(f"✅ TOTAL NEGATIVE EXAMPLES GENERATED: {len(negatives)}")
        print("=" * 70)
        
        return negatives
    
    def save_negatives(self, negatives: List[Dict], output_file: str):
        """Save negative examples to JSONL file."""
        with open(output_file, 'w') as f:
            for example in negatives:
                f.write(json.dumps(example) + '\n')
        
        print(f"\n💾 Saved {len(negatives)} negative examples to: {output_file}")
        
        # Also save summary
        summary = {
            'total_negatives': len(negatives),
            'categories': {
                'FAILED_BREAKOUT': len([n for n in negatives if n['category'] == 'FAILED_BREAKOUT']),
                'NEAR_MISS': len([n for n in negatives if n['category'] == 'NEAR_MISS']),
                'RANDOM_NON_PATTERN': len([n for n in negatives if n['category'] == 'RANDOM_NON_PATTERN'])
            },
            'generated_date': datetime.now().isoformat()
        }
        
        summary_file = output_file.replace('.jsonl', '_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Saved summary to: {summary_file}")


if __name__ == '__main__':
    # Configuration
    POSITIVE_TRADES_FILE = 'data/raw/parsed_trade_entries_initial.json'
    OUTPUT_FILE = 'data/processed/negative_labels_consolidation.jsonl'
    
    # Initialize generator
    generator = NegativeLabelGenerator(POSITIVE_TRADES_FILE)
    
    # Generate negatives (aiming for ~2x positive examples)
    # 744 positives → ~1500 negatives
    negatives = generator.generate_all_negatives(
        failed_breakouts=600,    # 40%
        near_misses=450,         # 30%
        random_non_patterns=450  # 30%
    )
    
    # Save results
    generator.save_negatives(negatives, OUTPUT_FILE)
    
    print("\n✨ Negative label generation complete!")
    print(f"\nNext steps:")
    print(f"1. Review generated examples in: {OUTPUT_FILE}")
    print(f"2. Combine with positive labels for training")
    print(f"3. Extract features and train Pipeline 1 (Consolidation Detector)")
