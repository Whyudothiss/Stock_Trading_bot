"""
Batch Feature Extraction for Pipeline 2 (5R Predictor)

Processes all trades from ml_5R_labels_filtered.jsonl and extracts features
for both 1-min and 5-min timeframes using actual entry prices.

Output:
- data/processed/features_5R_1min.jsonl (707 trades with 19 features each)
- data/processed/features_5R_5min.jsonl (707 trades with 19 features each)
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import time

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from ml_pipeline.breakout_features import BreakoutFeatureExtractor

def load_trades(filepath):
    """Load trades from JSONL file, filtering out excluded labels"""
    trades = []
    
    with open(filepath, 'r') as f:
        for line in f:
            trade = json.loads(line)
            
            # Only include trades we want to train on
            if trade['ml_label'] in ['HIT_5R', 'LOSS_1R', 'STOPPED_BEFORE_3R']:
                trades.append(trade)
    
    return trades


def extract_features_for_trade(trade, extractor, timeframe):
    """Extract features for a single trade"""
    try:
        # Parse entry date (handle timezone if present)
        entry_date_str = trade['entry_date']
        
        # Convert from SGT (UTC+8) to US Eastern Time for correct date
        if 'T' in entry_date_str:
            # Parse full timestamp with timezone
            from datetime import datetime
            import pytz
            import re
            
            # Normalize timestamp format (handle varying microsecond precision)
            # 2023-11-13T22:36:03.17+08:00 -> 2023-11-13T22:36:03.170000+08:00
            ts = entry_date_str.replace('+08:00', '')
            if '.' in ts:
                base, frac = ts.rsplit('.', 1)
                frac = frac[:6].ljust(6, '0')  # Pad or truncate to 6 digits
                ts = f"{base}.{frac}"
            
            # Parse the SGT timestamp
            dt_sgt = datetime.fromisoformat(ts)
            sgt = pytz.timezone('Asia/Singapore')
            dt_sgt = sgt.localize(dt_sgt)
            
            # Convert to US Eastern Time
            eastern = pytz.timezone('US/Eastern')
            dt_et = dt_sgt.astimezone(eastern)
            
            # Use the ET date for data fetching
            entry_date = dt_et.strftime('%Y-%m-%d')
        else:
            entry_date = entry_date_str
        
        # Extract features
        features = extractor.extract_features(
            ticker=trade['ticker'],
            entry_date=entry_date,
            entry_price=trade['entry_price'],
            stop_price=trade['initial_stop_loss_price'],
            target_price=trade['ml_target']
        )
        
        if features is None:
            return None
        
        # Add label (1 for HIT_5R, 0 for losses)
        features['label'] = 1 if trade['ml_label'] == 'HIT_5R' else 0
        features['ticker'] = trade['ticker']
        features['entry_date'] = entry_date
        
        return features
        
    except Exception as e:
        print(f"  ❌ Error processing {trade['ticker']} ({entry_date_str[:10]}): {e}")
        return None


def process_batch(trades, timeframe, output_file):
    """Process all trades for a specific timeframe"""
    print(f"\n{'='*80}")
    print(f"Processing {len(trades)} trades with {timeframe} features...")
    print(f"{'='*80}")
    
    extractor = BreakoutFeatureExtractor(timeframe=timeframe)
    
    results = []
    failed = []
    
    start_time = time.time()
    
    for i, trade in enumerate(trades, 1):
        # Progress update every 50 trades
        if i % 50 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            remaining = (len(trades) - i) / rate
            print(f"Progress: {i}/{len(trades)} ({i/len(trades)*100:.1f}%) - "
                  f"ETA: {remaining/60:.1f} min")
        
        features = extract_features_for_trade(trade, extractor, timeframe)
        
        if features is not None:
            results.append(features)
        else:
            failed.append({
                'ticker': trade['ticker'],
                'entry_date': trade['entry_date'][:10],
                'reason': 'Feature extraction failed'
            })
    
    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    # Summary
    elapsed = time.time() - start_time
    success_rate = len(results) / len(trades) * 100
    
    print(f"\n{'='*80}")
    print(f"✅ {timeframe} Feature Extraction Complete")
    print(f"{'='*80}")
    print(f"Successful: {len(results)}/{len(trades)} ({success_rate:.1f}%)")
    print(f"Failed: {len(failed)}/{len(trades)} ({len(failed)/len(trades)*100:.1f}%)")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"Saved to: {output_path}")
    
    # Show label distribution
    positive = sum(1 for r in results if r['label'] == 1)
    negative = len(results) - positive
    print(f"\nLabel distribution:")
    print(f"  Positive (HIT_5R): {positive} ({positive/len(results)*100:.1f}%)")
    print(f"  Negative (LOSS): {negative} ({negative/len(results)*100:.1f}%)")
    
    if failed:
        print(f"\n⚠️  Failed trades (first 10):")
        for fail in failed[:10]:
            print(f"  - {fail['ticker']} ({fail['entry_date']}): {fail['reason']}")
    
    return results, failed


def main():
    # Paths
    labels_file = Path(__file__).parent.parent.parent / "data" / "processed" / "ml_5R_labels_filtered.jsonl"
    output_dir = Path(__file__).parent.parent.parent / "data" / "processed"
    
    # Load trades
    print(f"Loading trades from: {labels_file}")
    trades = load_trades(labels_file)
    print(f"✅ Loaded {len(trades)} trades for training")
    
    # Show distribution
    from collections import Counter
    label_dist = Counter([t['ml_label'] for t in trades])
    print(f"\nLabel distribution:")
    for label, count in sorted(label_dist.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count} ({count/len(trades)*100:.1f}%)")
    
    # Process 1-min features
    results_1min, failed_1min = process_batch(
        trades,
        timeframe='1Min',
        output_file=output_dir / "features_5R_1min.jsonl"
    )
    
    # Process 5-min features
    results_5min, failed_5min = process_batch(
        trades,
        timeframe='5Min',
        output_file=output_dir / "features_5R_5min.jsonl"
    )
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"🎉 BATCH FEATURE EXTRACTION COMPLETE")
    print(f"{'='*80}")
    print(f"1-Min features: {len(results_1min)} trades")
    print(f"5-Min features: {len(results_5min)} trades")
    print(f"\nNext steps:")
    print(f"1. Train model on 1-min features: python3 src/ml_pipeline/train_5R_model.py --timeframe 1min")
    print(f"2. Train model on 5-min features: python3 src/ml_pipeline/train_5R_model.py --timeframe 5min")
    print(f"3. Compare performance to decide which timeframe works better")


if __name__ == "__main__":
    main()
