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
import pandas as pd
import argparse

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from ml_pipeline.breakout_features import BreakoutFeatureExtractor
from data_processing.download_minute_data import TwelveDataMinuteDownloader, DailyLimitExceededError

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
    
    # Use the new TwelveData downloader and inject it into the extractor
    downloader = TwelveDataMinuteDownloader()
    extractor = BreakoutFeatureExtractor(timeframe=timeframe, downloader=downloader)

    # Show which (ticker, date) have already been downloaded (helps resume and avoid re-requests)
    try:
        downloaded = downloader.list_downloaded()
        if downloaded:
            print(f"Found {len(downloaded)} cached downloads in Twelve Data manifest (first 10 shown):")
            for d in downloaded[:10]:
                print(f"  - {d.get('ticker')} on {d.get('date')} ({d.get('timeframe')})")
        else:
            print("No entries found in Twelve Data manifest (no cached downloads yet)")
    except Exception:
        pass
    
    # --- Logic for resuming progress ---
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        processed_df = pd.read_json(output_path, lines=True)
        # Create a set of already processed trades for quick lookup
        processed_trades = set(zip(processed_df['ticker'], processed_df['entry_date']))
        print(f"Resuming. Found {len(processed_trades)} already processed trades in {output_path.name}")
    except (FileNotFoundError, ValueError):
        processed_df = pd.DataFrame()
        processed_trades = set()
    # ------------------------------------

    newly_extracted_features = []
    failed_count = 0
    
    start_time = time.time()
    
    try:
        for i, trade in enumerate(trades, 1):
            # --- Check if trade is already processed ---
            # Normalize date for the check
            entry_date_check = trade['entry_date'].split('T')[0]
            if (trade['ticker'], entry_date_check) in processed_trades:
                continue
            # -----------------------------------------

            print(f"Processing {i}/{len(trades)}: {trade['ticker']} on {entry_date_check}")
            
            features = extract_features_for_trade(trade, extractor, timeframe)
            
            if features:
                newly_extracted_features.append(features)
            else:
                failed_count += 1

    except DailyLimitExceededError:
        print("\n🛑 Daily API limit reached. Saving progress and shutting down.")
        print("   Run the script again tomorrow to resume.")
    
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

    finally:
        # --- Save Progress ---
        if newly_extracted_features:
            new_features_df = pd.DataFrame(newly_extracted_features)
            # Append new features to the existing ones
            final_df = pd.concat([processed_df, new_features_df], ignore_index=True)
            final_df.to_json(output_path, orient='records', lines=True)
            print(f"\n✅ Saved {len(newly_extracted_features)} new feature sets. Total is now {len(final_df)}.")
        else:
            print("\nNo new features were extracted in this run.")
        
        elapsed = time.time() - start_time
        print(f"Time elapsed this run: {elapsed/60:.1f} minutes")
        print(f"Failures this run: {failed_count}")

    return len(processed_df) + len(newly_extracted_features), failed_count


def main():
    # Paths
    labels_file = Path(__file__).parent.parent.parent / "data" / "processed" / "ml_5R_labels_filtered.jsonl"
    output_dir = Path(__file__).parent.parent.parent / "data" / "processed"
    
    # CLI: allow running only 1min or 5min, and optional limit for smoke tests
    parser = argparse.ArgumentParser(description='Batch feature extraction (Twelve Data)')
    parser.add_argument('--timeframe', choices=['1min', '5min', 'both'], default='both',
                        help='Which timeframe to extract (default: both)')
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit number of trades to process (0 = no limit)')
    args = parser.parse_args()

    # Load trades
    print(f"Loading trades from: {labels_file}")
    trades = load_trades(labels_file)
    if args.limit and args.limit > 0:
        trades = trades[:args.limit]
        print(f"🔎 Running in limited mode: processing first {len(trades)} trades")
    print(f"✅ Loaded {len(trades)} trades for training")
    
    # Show distribution
    from collections import Counter
    label_dist = Counter([t['ml_label'] for t in trades])
    print(f"\nLabel distribution:")
    for label, count in sorted(label_dist.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count} ({count/len(trades)*100:.1f}%)")
    
    results_1min = results_5min = failed_1min = failed_5min = 0

    if args.timeframe in ('1min', 'both'):
        results_1min, failed_1min = process_batch(
            trades,
            timeframe='1Min',
            output_file=output_dir / "features_5R_1min_twelve.jsonl"
        )

    if args.timeframe in ('5min', 'both'):
        results_5min, failed_5min = process_batch(
            trades,
            timeframe='5Min',
            output_file=output_dir / "features_5R_5min_twelve.jsonl"
        )

    # Final summary
    print(f"\n{'='*80}")
    print(f"🎉 BATCH FEATURE EXTRACTION COMPLETE")
    print(f"{'='*80}")
    print(f"1-Min features: {results_1min} trades")
    print(f"5-Min features: {results_5min} trades")
    print(f"\nNext steps:")
    print(f"1. Train model on 1-min features: python3 src/ml_pipeline/train_5R_model.py --timeframe 1min")
    print(f"2. Train model on 5-min features: python3 src/ml_pipeline/train_5R_model.py --timeframe 5min")
    print(f"3. Compare performance to decide which timeframe works better")


if __name__ == "__main__":
    main()
