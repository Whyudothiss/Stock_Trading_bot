"""
Test Pipeline 2 with live-trading logic
Enter a ticker, date, and time to get probability prediction
Uses high-of-day as entry and low-of-day as stop (like continuous evaluation)
NOW INCLUDES CONSOLIDATION FEATURES from previous days!
"""

import sys
sys.path.append('.')

import joblib
import numpy as np
import yfinance as yf
from datetime import datetime
import pandas as pd
from src.ml_pipeline.breakout_features import BreakoutFeatureExtractor
from src.data_processing.download_minute_data import TwelveDataMinuteDownloader

def test_pipeline2(ticker, date_str, time_str):
    """
    Test Pipeline 2 for a specific ticker at a specific time
    
    Args:
        ticker: Stock ticker (e.g., 'BITF')
        date_str: Date in format 'YYYY-MM-DD' (e.g., '2025-10-01')
        time_str: Time in format 'HH:MM' (e.g., '10:15')
    
    Returns probability and trade details
    """
    print(f"\n{'='*70}")
    print(f"Testing Pipeline 2: {ticker} on {date_str} at {time_str}")
    print(f"{'='*70}\n")
    
    # Load Pipeline 2 model
    try:
        model = joblib.load('models/5R_predictor_5min.pkl')
        with open('models/5R_predictor_5min_features.txt', 'r') as f:
            feature_names = [line.strip() for line in f]
        td_downloader = TwelveDataMinuteDownloader()
        extractor = BreakoutFeatureExtractor(timeframe='5min', downloader=td_downloader)
        print("✅ Pipeline 2 model loaded")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Get 5-minute intraday data from TwelveData
    try:
        intraday_data = td_downloader.download_minute_data(ticker, date_str, timeframe='5min')
        
        if intraday_data is None or intraday_data.empty:
            print(f"❌ No intraday data found for {ticker} on {date_str}")
            return None
        
        print(f"✅ Downloaded {len(intraday_data)} 5-minute bars from TwelveData")
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        return None
    
    # Find the specific time candle (convert target to timezone-aware if data is timezone-aware)
    target_time = datetime.strptime(f"{date_str} {time_str}", '%Y-%m-%d %H:%M')
    
    # Check if data is timezone-aware and convert target_time accordingly
    if intraday_data.index.tzinfo is not None:
        import pytz
        target_time = pytz.timezone('US/Eastern').localize(target_time)
    
    # Find candle at or before target time
    matching_candle = None
    for idx, row in intraday_data.iterrows():
        if idx <= target_time:
            matching_candle = (idx, row)
        else:
            break
    
    if matching_candle is None:
        print(f"❌ No candle found at or before {time_str}")
        return None
    
    candle_time, candle_row = matching_candle
    
    # Calculate high and low of day UP TO this candle
    data_up_to_candle = intraday_data.loc[:candle_time]
    high_of_day = data_up_to_candle['High'].max()
    low_of_day = data_up_to_candle['Low'].min()
    
    # Entry = high of day + 0.5% slippage
    entry_price = high_of_day * 1.005
    stop_price = low_of_day
    risk = entry_price - stop_price
    
    if risk <= 0:
        print(f"❌ Invalid risk: entry ${entry_price:.2f} <= stop ${stop_price:.2f}")
        return None
    
    target_price = entry_price + 5 * risk
    
    print(f"\n📊 Market Data at {candle_time.strftime('%H:%M')}:")
    print(f"   Candle: O=${candle_row['Open']:.2f} H=${candle_row['High']:.2f} L=${candle_row['Low']:.2f} C=${candle_row['Close']:.2f}")
    print(f"   High of day so far: ${high_of_day:.2f}")
    print(f"   Low of day so far: ${low_of_day:.2f}")
    
    print(f"\n💰 Hypothetical Trade Setup:")
    print(f"   Entry: ${entry_price:.2f} (high of day + 0.5%)")
    print(f"   Stop:  ${stop_price:.2f} (low of day)")
    print(f"   Risk:  ${risk:.2f} per share")
    print(f"   Target: ${target_price:.2f} (5R)")
    print(f"   Risk/Reward: 1:5")
    
    # Extract features using BreakoutFeatureExtractor (includes consolidation features!)
    try:
        entry_date = datetime.strptime(date_str, '%Y-%m-%d')
        features_dict = extractor.extract_features(
            ticker, entry_date, entry_price, stop_price, target_price
        )
        
        if not features_dict:
            print(f"❌ Error extracting features")
            return None
        
        # Build feature vector
        feature_values = []
        for fname in feature_names:
            feature_values.append(features_dict.get(fname, 0))
        
        X = np.array([feature_values])
        
        # Get probability
        prob = model.predict_proba(X)[0][1]
        
        print(f"\n🤖 Pipeline 2 Prediction:")
        print(f"   Probability: {prob*100:.1f}%")
        print(f"   Threshold: 30%")
        
        if prob >= 0.30:
            print(f"   ✅ PASS - Would add to hot list")
        else:
            print(f"   ❌ FAIL - Would not trade")
        
        print(f"\n📈 Key Breakout Features:")
        print(f"   Risk %: {features_dict.get('risk_pct', 0):.2f}%")
        print(f"   Entry position in bar: {features_dict.get('entry_position_in_bar', 0):.2f}")
        print(f"   Gap %: {features_dict.get('gap_pct', 0):.2f}%")
        print(f"   First candle range: {features_dict.get('first_candle_range_pct', 0):.2f}%")
        
        print(f"\n📊 Consolidation Features (from previous 10 days):")
        print(f"   Consol range: {features_dict.get('consol_range_pct', 0):.2f}%")
        print(f"   Consol position: {features_dict.get('consol_position_in_range', 0):.2f}")
        print(f"   Days near high: {features_dict.get('consol_days_near_high', 0)}")
        print(f"   Volume trend: {features_dict.get('consol_volume_trend', 0):.0f}")
        
        return {
            'ticker': ticker,
            'date': date_str,
            'time': time_str,
            'probability': prob,
            'entry_price': entry_price,
            'stop_price': stop_price,
            'target_price': target_price,
            'risk': risk,
            'passes': prob >= 0.30
        }
        
    except Exception as e:
        print(f"❌ Error extracting features: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Interactive testing"""
    print("\n" + "="*70)
    print("Pipeline 2 Live Testing Tool")
    print("="*70)
    print("\nTest Pipeline 2 with your intended usage:")
    print("- Enter ticker, date, and time")
    print("- Entry = high of day so far + 0.5%")
    print("- Stop = low of day so far")
    print("- Get probability prediction")
    print("\n" + "="*70 + "\n")
    
    # Check if running with command line args
    if len(sys.argv) == 4:
        ticker = sys.argv[1].upper()
        date_str = sys.argv[2]
        time_str = sys.argv[3]
        test_pipeline2(ticker, date_str, time_str)
        return
    
    # Interactive mode
    print("Interactive Mode (or run: python test_pipeline2_live.py TICKER DATE TIME)")
    print("Example: python test_pipeline2_live.py BITF 2025-10-01 10:15\n")
    
    while True:
        try:
            ticker = input("\nEnter ticker (or 'quit' to exit): ").strip().upper()
            if ticker.lower() in ['quit', 'exit', 'q', '']:
                print("\n👋 Goodbye!")
                break
            
            date_str = input("Enter date (YYYY-MM-DD): ").strip()
            if not date_str:
                print("⚠️  Please enter a valid date\n")
                continue
            
            time_str = input("Enter time (HH:MM): ").strip()
            if not time_str:
                print("⚠️  Please enter a valid time\n")
                continue
            
            test_pipeline2(ticker, date_str, time_str)
            
            print("\n" + "-"*70)
            
            # Ask if user wants to check another
            another = input("\nCheck another stock? (y/n) [y]: ").strip().lower()
            if another in ['n', 'no']:
                print("\n👋 Goodbye!")
                break
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            continue


if __name__ == "__main__":
    main()
