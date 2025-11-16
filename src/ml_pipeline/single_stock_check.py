#!/usr/bin/env python3
"""
Single Stock Checker - Manually enter ticker and date to check consolidation probability
"""

import sys
from pathlib import Path
from datetime import datetime
sys.path.append(str(Path(__file__).parent.parent))

from ml_pipeline.scan_stocks import StockScanner


def main():
    """Check a single stock manually"""
    
    print("\n" + "="*60)
    print("🔍 SINGLE STOCK CONSOLIDATION CHECKER")
    print("="*60 + "\n")
    
    # Initialize scanner
    scanner = StockScanner()
    
    while True:
        # Get ticker
        ticker = input("Enter ticker symbol (or 'quit' to exit): ").strip().upper()
        
        if ticker.lower() == 'quit' or ticker.lower() == 'q':
            print("\n👋 Goodbye!")
            break
        
        if not ticker:
            print("⚠️  Please enter a valid ticker\n")
            continue
        
        # Get date
        date_input = input("Enter date (YYYY-MM-DD) or press Enter for yesterday: ").strip()
        
        if date_input:
            try:
                # Validate date format
                datetime.strptime(date_input, '%Y-%m-%d')
                date = date_input
            except ValueError:
                print("⚠️  Invalid date format. Use YYYY-MM-DD\n")
                continue
        else:
            date = None  # Will default to yesterday
        
        # Scan the ticker
        print(f"\n🔍 Analyzing {ticker}...")
        try:
            proba, features = scanner.scan_ticker(ticker, date)
            
            if proba is None:
                print(f"❌ Failed to analyze {ticker}. Check if ticker exists or has sufficient data.\n")
                continue
            
            # Display results
            confidence = proba * 100
            print("\n" + "="*60)
            print(f"📊 RESULTS FOR {ticker}")
            print("="*60)
            print(f"\n🎯 Confidence: {confidence:.2f}%")
            
            if confidence >= 70:
                print("   ✅ HIGH - Strong consolidation pattern!")
            elif confidence >= 60:
                print("   🟡 MEDIUM-HIGH - Good consolidation candidate")
            elif confidence >= 50:
                print("   🟠 MEDIUM - Possible consolidation")
            else:
                print("   🔴 LOW - Weak or no consolidation pattern")
            
            print(f"\n📈 Key Metrics:")
            print(f"   Price Range:        {features['price_range_pct']:6.2f}%")
            print(f"   Position in Range:  {features['position_in_range']:6.2f} (0=low, 1=high)")
            print(f"   Volatility:         {features['volatility']:6.2f}%")
            print(f"   10 SMA Distance:    {features['dist_from_sma10_pct']:+6.2f}%")
            print(f"   20 SMA Distance:    {features['dist_from_sma20_pct']:+6.2f}%")
            print(f"   50 SMA Distance:    {features['dist_from_sma50_pct']:+6.2f}%")
            print(f"   Volume Trend:       {features['volume_trend']:+.2f}")
            print(f"   Consolidation Range:{features['consolidation_range_pct']:6.2f}%")
            print(f"   Days Hugging 10SMA: {features['days_hugging_10sma']}/10 days")
            
            print("\n" + "="*60 + "\n")
            
        except Exception as e:
            print(f"❌ Error analyzing {ticker}: {str(e)}\n")
            continue
        
        # Ask if user wants to check another
        another = input("Check another stock? (y/n) [y]: ").strip().lower()
        if another == 'n' or another == 'no':
            print("\n👋 Goodbye!")
            break
        print()


if __name__ == "__main__":
    main()
