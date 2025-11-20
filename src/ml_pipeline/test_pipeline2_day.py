"""
Test Pipeline 2 for entire trading day
Checks every 5-minute candle from 9:30 AM to 4:00 PM
Shows when probability crosses the 30% threshold
"""

import sys
sys.path.append('.')


import joblib
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from src.ml_pipeline.breakout_features import BreakoutFeatureExtractor

def test_pipeline2_full_day(ticker, date_str):
    """
    Test Pipeline 2 for entire trading day, checking every 5-minute candle
    
    Args:
        ticker: Stock ticker (e.g., 'RGTI')
        date_str: Date in format 'YYYY-MM-DD' (e.g., '2025-10-02')
    
    Shows all candles and highlights when probability >= 30%
    """
    print(f"\n{'='*80}")
    print(f"Pipeline 2 Full Day Analysis: {ticker} on {date_str}")
    print(f"{'='*80}\n")
    
    # Load Pipeline 2 model and feature extractor
    try:
        model = joblib.load('models/5R_predictor_5min.pkl')
        with open('models/5R_predictor_5min_features.txt', 'r') as f:
            feature_names = [line.strip() for line in f]
        extractor = BreakoutFeatureExtractor(timeframe='5Min')
        print("✅ Pipeline 2 model and BreakoutFeatureExtractor loaded")
    except Exception as e:
        print(f"❌ Error loading model or extractor: {e}")
        return None
    
    # Get 5-minute intraday data
    try:
        yf_ticker = yf.Ticker(ticker)
        next_date = (datetime.strptime(date_str, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
        intraday_data = yf_ticker.history(start=date_str, end=next_date, interval='5m')
        if intraday_data.empty:
            print(f"❌ No intraday data found for {ticker} on {date_str}")
            return None
        print(f"✅ Downloaded {len(intraday_data)} 5-minute bars\n")
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        return None

    results = []
    for i, (candle_time, candle_row) in enumerate(intraday_data.iterrows()):
        data_up_to_candle = intraday_data.iloc[:i+1]
        high_of_day = data_up_to_candle['High'].max()
        low_of_day = data_up_to_candle['Low'].min()
        entry_price = high_of_day * 1.005
        stop_price = low_of_day
        risk = entry_price - stop_price
        if risk <= 0:
            continue
        target_price = entry_price + (5 * risk)
        try:
            features_dict = extractor.extract_features(
                ticker, datetime.strptime(date_str, '%Y-%m-%d'), entry_price, stop_price, target_price
            )
            if not features_dict:
                print(f"❌ Error extracting features for {ticker} at {candle_time}")
                continue
        except Exception as e:
            print(f"❌ Exception in feature extraction: {e}")
            continue
        feature_vector = [features_dict.get(fname, 0) for fname in feature_names]
        X = np.array([feature_vector])
        prob = model.predict_proba(X)[0][1]
        time_str = candle_time.strftime('%H:%M') if hasattr(candle_time, 'strftime') else str(candle_time)[11:16]
        results.append({
            'time': time_str,
            'close': candle_row['Close'],
            'high_of_day': high_of_day,
            'low_of_day': low_of_day,
            'entry': entry_price,
            'stop': stop_price,
            'risk_pct': features_dict.get('risk_pct', 0),
            'probability': prob,
            'passes': prob >= 0.30
        })
    
    # Display results
    if not results:
        print("❌ No valid results")
        return None
    
    print(f"{'Time':<8} {'Close':<8} {'Entry':<8} {'Stop':<8} {'Risk%':<7} {'Prob':<7} {'Status'}")
    print("-" * 80)
    
    pass_count = 0
    for r in results:
        status = "✅ PASS" if r['passes'] else "❌ FAIL"
        if r['passes']:
            pass_count += 1
        
        # Highlight passing trades
        marker = ">>> " if r['passes'] else "    "
        
        print(f"{marker}{r['time']:<8} ${r['close']:<7.2f} ${r['entry']:<7.2f} ${r['stop']:<7.2f} {r['risk_pct']:<6.2f}% {r['probability']*100:<6.1f}% {status}")
    
    print(f"\n{'='*80}")
    print(f"Summary: {pass_count}/{len(results)} candles passed Pipeline 2 ({pass_count/len(results)*100:.1f}%)")
    print(f"{'='*80}\n")
    
    # Show first passing time
    first_pass = next((r for r in results if r['passes']), None)
    if first_pass:
        print(f"🎯 First PASS at {first_pass['time']}")
        print(f"   Entry: ${first_pass['entry']:.2f}, Stop: ${first_pass['stop']:.2f}, Probability: {first_pass['probability']*100:.1f}%")
    else:
        print("❌ No candles passed Pipeline 2 threshold (30%)")
    
    return results


def main():
    """Main entry point"""
    
    # Check command line arguments
    if len(sys.argv) >= 3:
        ticker = sys.argv[1]
        date_str = sys.argv[2]
    else:
        print("\n" + "="*80)
        print("Pipeline 2 Full Day Testing Tool")
        print("="*80)
        print("\nUsage: python test_pipeline2_day.py TICKER DATE")
        print("Example: python test_pipeline2_day.py RGTI 2025-10-02")
        print("\n" + "="*80 + "\n")
        
        # Interactive mode
        ticker = input("Enter ticker: ").strip().upper()
        date_str = input("Enter date (YYYY-MM-DD): ").strip()
    
    # Validate date format
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        print("❌ Invalid date format. Use YYYY-MM-DD")
        return
    
    # Run test
    test_pipeline2_full_day(ticker, date_str)


if __name__ == '__main__':
    main()
