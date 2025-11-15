#!/usr/bin/env python3
"""
Verify Existing Negative Labels

Takes the synthetic negatives and verifies they're not actually valid consolidations.
Uses caching and rate limiting.
"""

import json
import time
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import yfinance as yf
import pandas as pd


class NegativeLabelVerifier:
    def __init__(self, cache_dir: str = 'data/cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        self.verified_negatives = []
        self.rejected_as_consolidations = []
        self.failed_downloads = []
    
    def _get_cache_path(self, ticker: str) -> str:
        """Get cache file path for a ticker."""
        return os.path.join(self.cache_dir, f"{ticker}.parquet")
    
    def _load_cached_data(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Load data from cache or download if needed."""
        cache_path = self._get_cache_path(ticker)
        
        # Try cache first
        if os.path.exists(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                df.index = pd.to_datetime(df.index)
                
                req_start = pd.to_datetime(start_date)
                req_end = pd.to_datetime(end_date)
                
                if df.index.min() <= req_start and df.index.max() >= req_end:
                    return df[start_date:end_date]
            except Exception as e:
                pass
        
        # Download if needed
        try:
            print(f"  ⬇ {ticker}...", end='', flush=True)
            time.sleep(0.5)  # Rate limiting
            
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(start=start_date, end=end_date, auto_adjust=True)
            
            if df.empty:
                print(" ✗ No data")
                self.failed_downloads.append(ticker)
                return None
            
            # Cache for future use
            try:
                full_df = ticker_obj.history(period="2y", auto_adjust=True)
                if not full_df.empty:
                    full_df.to_parquet(cache_path)
                    print(" ✓ Cached")
            except:
                print(" ✓ OK")
            
            return df
            
        except Exception as e:
            print(f" ✗ {str(e)[:50]}")
            self.failed_downloads.append(ticker)
            return None
    
    def _is_valid_consolidation(self, df: pd.DataFrame, check_date: str) -> bool:
        """
        Check if data shows a valid consolidation pattern.
        Returns True = consolidation (REJECT as negative)
        Returns False = NOT consolidation (KEEP as negative)
        """
        try:
            check_date_dt = pd.to_datetime(check_date)
            
            # Find the date in the dataframe
            if check_date_dt not in df.index:
                # Try to find closest date
                closest_idx = df.index.get_indexer([check_date_dt], method='nearest')[0]
                if closest_idx == -1 or abs((df.index[closest_idx] - check_date_dt).days) > 5:
                    return False  # Date too far, can't verify
                check_date_dt = df.index[closest_idx]
            
            check_idx = df.index.get_loc(check_date_dt)
            
            # Need at least 10 days before for analysis
            if check_idx < 10:
                return False
            
            # 10-day window
            window = df.iloc[check_idx-10:check_idx+1]
            if len(window) < 10:
                return False
            
            # Consolidation metrics
            high = window['High'].max()
            low = window['Low'].min()
            close = window['Close'].iloc[-1]
            
            # Range
            range_pct = ((high - low) / low) * 100
            
            # Position in range (near high = bullish)
            position_in_range = (close - low) / (high - low) if high != low else 0
            
            # Volume trend
            vol_first_half = window['Volume'].iloc[:5].mean()
            vol_second_half = window['Volume'].iloc[5:].mean()
            vol_declining = vol_second_half < vol_first_half
            
            # Volatility
            returns = window['Close'].pct_change()
            volatility = returns.std() * 100
            
            # Consolidation criteria (same as Trade_Analyzer)
            is_consolidation = (
                5 <= range_pct <= 40 and
                position_in_range > 0.6 and
                volatility < 5 and
                vol_declining
            )
            
            return is_consolidation
            
        except Exception as e:
            return False  # Can't verify, assume it's OK as negative
    
    def verify_negatives(self, negatives: List[Dict], max_to_verify: int = None) -> Dict:
        """Verify a list of negative labels."""
        if max_to_verify:
            negatives = negatives[:max_to_verify]
        
        print(f"\nVerifying {len(negatives)} negative labels...")
        print("=" * 70)
        
        for i, neg in enumerate(negatives, 1):
            ticker = neg['ticker']
            date = neg['date']
            
            print(f"{i}/{len(negatives)} {ticker} {date}...", end=' ')
            
            # Download data (15 days before for analysis window)
            start = (pd.to_datetime(date) - timedelta(days=15)).strftime('%Y-%m-%d')
            end = (pd.to_datetime(date) + timedelta(days=5)).strftime('%Y-%m-%d')
            
            df = self._load_cached_data(ticker, start, end)
            
            if df is None:
                print("SKIP (download failed)")
                continue
            
            # Check if it's actually a consolidation
            is_consolidation = self._is_valid_consolidation(df, date)
            
            if is_consolidation:
                print("❌ REJECT (looks like consolidation!)")
                self.rejected_as_consolidations.append(neg)
            else:
                print("✅ KEEP")
                self.verified_negatives.append(neg)
        
        return {
            'verified': self.verified_negatives,
            'rejected': self.rejected_as_consolidations,
            'failed': self.failed_downloads
        }


def main():
    """Verify existing synthetic negatives."""
    print("=" * 70)
    print("VERIFYING EXISTING NEGATIVE LABELS")
    print("=" * 70)
    
    # Load existing negatives
    input_file = 'data/processed/negative_labels_consolidation.jsonl'
    negatives = []
    with open(input_file, 'r') as f:
        for line in f:
            negatives.append(json.loads(line))
    
    print(f"\nLoaded {len(negatives)} synthetic negatives")
    
    # TEST MODE: Only verify first 20
    TEST_MODE = False
    test_count = 20
    
    if TEST_MODE:
        print(f"\n⚠ TEST MODE: Verifying only first {test_count} labels")
        print("(Edit script to disable TEST_MODE for full verification)\n")
    
    verifier = NegativeLabelVerifier(cache_dir='data/cache')
    
    results = verifier.verify_negatives(
        negatives,
        max_to_verify=test_count if TEST_MODE else None
    )
    
    # Report
    print("\n" + "=" * 70)
    print("VERIFICATION RESULTS")
    print("=" * 70)
    print(f"✅ Verified as valid negatives: {len(results['verified'])}")
    print(f"❌ Rejected (actually consolidations): {len(results['rejected'])}")
    print(f"⚠ Failed downloads: {len(set(results['failed']))}")
    
    rejection_rate = len(results['rejected']) / (len(results['verified']) + len(results['rejected'])) * 100 if results['verified'] or results['rejected'] else 0
    print(f"\n📊 Rejection rate: {rejection_rate:.1f}%")
    
    if results['rejected']:
        print(f"\nRejected examples (first 5):")
        for neg in results['rejected'][:5]:
            print(f"  - {neg['ticker']} on {neg['date']} ({neg['category']})")
    
    # Save verified negatives
    if results['verified']:
        output_file = 'data/processed/negative_labels_consolidation_verified.jsonl'
        with open(output_file, 'w') as f:
            for neg in results['verified']:
                f.write(json.dumps(neg) + '\n')
        print(f"\n💾 Saved {len(results['verified'])} verified negatives to:")
        print(f"   {output_file}")
    
    if TEST_MODE:
        print("\n✨ Test complete! If results look good:")
        print("   1. Edit script: Set TEST_MODE = False")
        print("   2. Run again to verify all 800 negatives")
        print("   3. This will take ~10 minutes with rate limiting")


if __name__ == '__main__':
    main()
