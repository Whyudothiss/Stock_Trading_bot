#!/usr/bin/env python3
"""
Generate Simple Synthetic Negative Labels

Since we don't have cache and can't download due to rate limits,
create synthetic negative labels based on your positive trades.

Strategy:
- Use your positive trades as templates
- Modify key characteristics to make them "bad" examples
- Quick, no downloads needed
"""

import json
import random
from datetime import datetime, timedelta
from typing import List, Dict
import pandas as pd


def adjust_to_trading_day(date: datetime) -> datetime:
    """Adjust date to next trading day if it falls on weekend."""
    # Monday = 0, Sunday = 6
    weekday = date.weekday()
    
    if weekday == 5:  # Saturday
        date += timedelta(days=2)  # Move to Monday
    elif weekday == 6:  # Sunday
        date += timedelta(days=1)  # Move to Monday
    
    return date


def generate_synthetic_negatives(positive_file: str, output_file: str):
    """Generate synthetic negative examples from positive trades."""
    
    print("=" * 70)
    print("GENERATING SYNTHETIC NEGATIVE LABELS")
    print("=" * 70)
    
    # Load positives
    with open(positive_file, 'r') as f:
        positives = json.load(f)
    
    print(f"\nPositive examples: {len(positives)}")
    
    negatives = []
    
    # Strategy 1: Temporal negatives (30-60 days before actual entry)
    print(f"\n🔵 Generating temporal negatives (before consolidation formed)...")
    for trade in positives[:200]:  # Use 200 trades
        entry_date_str = trade['entry_date'].split('T')[0]
        entry_date = datetime.strptime(entry_date_str, '%Y-%m-%d')
        
        # Random date 30-60 days before
        days_before = random.randint(30, 60)
        early_date = entry_date - timedelta(days=days_before)
        early_date = adjust_to_trading_day(early_date)  # Fix weekends
        
        negatives.append({
            'ticker': trade['ticker'],
            'date': early_date.strftime('%Y-%m-%d'),
            'category': 'TOO_EARLY',
            'notes': f'{days_before} days before actual setup',
            'label': 0
        })
    
    # Strategy 2: Wrong tickers at right times
    print(f"🟡 Generating wrong-ticker negatives...")
    for i, trade in enumerate(positives[:200]):
        # Swap ticker with a different one
        other_trade = positives[(i + 100) % len(positives)]
        
        negatives.append({
            'ticker': other_trade['ticker'],  # Wrong ticker
            'date': trade['entry_date'].split('T')[0],  # Right date
            'category': 'WRONG_TICKER',
            'notes': 'Different stock at this time',
            'label': 0
        })
    
    # Strategy 3: Same ticker, wrong time (random other dates)
    print(f"🔴 Generating wrong-timing negatives...")
    for trade in positives[:200]:
        entry_date_str = trade['entry_date'].split('T')[0]
        entry_date = datetime.strptime(entry_date_str, '%Y-%m-%d')
        
        # Random date 100-200 days away (either direction)
        offset = random.randint(100, 200) * random.choice([-1, 1])
        random_date = entry_date + timedelta(days=offset)
        random_date = adjust_to_trading_day(random_date)  # Fix weekends
        
        negatives.append({
            'ticker': trade['ticker'],
            'date': random_date.strftime('%Y-%m-%d'),
            'category': 'WRONG_TIMING',
            'notes': f'{abs(offset)} days {"before" if offset < 0 else "after"} actual setup',
            'label': 0
        })
    
    # Strategy 4: Modified entries (simulate bad criteria)
    print(f"⚫ Generating modified-criteria negatives...")
    for trade in positives[:200]:
        entry_date_str = trade['entry_date'].split('T')[0]
        entry_date = datetime.strptime(entry_date_str, '%Y-%m-%d')
        
        # Slight time offset
        offset = random.randint(10, 20)
        mod_date = entry_date - timedelta(days=offset)
        mod_date = adjust_to_trading_day(mod_date)  # Fix weekends
        
        negatives.append({
            'ticker': trade['ticker'],
            'date': mod_date.strftime('%Y-%m-%d'),
            'category': 'NEAR_MISS',
            'notes': f'{offset} days before breakout (premature)',
            'label': 0
        })
    
    print(f"\n✅ Generated {len(negatives)} synthetic negative examples")
    
    # Save
    with open(output_file, 'w') as f:
        for neg in negatives:
            f.write(json.dumps(neg) + '\n')
    
    print(f"💾 Saved to: {output_file}")
    
    # Summary
    from collections import Counter
    categories = Counter([n['category'] for n in negatives])
    
    summary = {
        'total_negatives': len(negatives),
        'categories': dict(categories),
        'generated_date': datetime.now().isoformat(),
        'method': 'synthetic'
    }
    
    summary_file = output_file.replace('.jsonl', '_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"💾 Saved summary to: {summary_file}")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for cat, count in categories.items():
        print(f"  {cat}: {count}")
    
    return negatives


if __name__ == '__main__':
    negatives = generate_synthetic_negatives(
        positive_file='data/raw/parsed_trade_entries_initial.json',
        output_file='data/processed/negative_labels_consolidation.jsonl'
    )
    
    print("\n✨ Complete! You now have negative labels.")
    print("\nNext step:")
    print("  python3 src/ml_pipeline/consolidation_features.py")
