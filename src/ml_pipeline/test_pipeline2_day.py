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
from src.data_processing.download_minute_data import TwelveDataMinuteDownloader

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
    
    # Load Pipeline 2 model
    try:
        model = joblib.load('models/5R_predictor_5min.pkl')
        with open('models/5R_predictor_5min_features.txt', 'r') as f:
            feature_names = [line.strip() for line in f]
        td_downloader = TwelveDataMinuteDownloader()
        print("✅ Pipeline 2 model loaded")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Get daily data from yfinance for consolidation features (10-day lookback)
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        lookback_days = 10
        start_date = date_obj - timedelta(days=lookback_days + 5)  # Extra buffer
        end_date = date_obj + timedelta(days=1)
        
        daily_df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True
        )
        
        # Flatten MultiIndex columns if present
        if isinstance(daily_df.columns, pd.MultiIndex):
            daily_df.columns = daily_df.columns.get_level_values(0)
        
        if daily_df is None or len(daily_df) < lookback_days:
            print(f"❌ Not enough daily data for consolidation features (need {lookback_days} days)")
            return None
        
        print(f"✅ Downloaded {len(daily_df)} days of data from yfinance for consolidation features")
        
        # Get the 10-day window before entry date
        entry_idx = daily_df.index.get_indexer([date_obj], method='nearest')[0]
        if entry_idx < lookback_days:
            consol_window = daily_df.iloc[:entry_idx]
        else:
            consol_window = daily_df.iloc[entry_idx - lookback_days:entry_idx]
        
        if len(consol_window) < 5:
            print(f"❌ Not enough data in consolidation window ({len(consol_window)} days)")
            return None
        
        # Calculate consolidation features from 10-day window
        high_max = float(consol_window['High'].max())
        low_min = float(consol_window['Low'].min())
        last_close = float(consol_window['Close'].iloc[-1])
        
        consol_range_pct = ((high_max - low_min) / high_max) * 100 if high_max > 0 else 0
        consol_position_in_range = ((last_close - low_min) / (high_max - low_min)) if (high_max - low_min) > 0 else 0.5
        
        # Volatility (std dev of returns)
        returns = consol_window['Close'].pct_change().dropna()
        consol_volatility = float(returns.std()) if len(returns) > 0 else 0
        
        # Volume trend (declining = good consolidation)
        if len(consol_window) >= 2:
            x = np.arange(len(consol_window))
            y = consol_window['Volume'].values
            consol_volume_trend = float(np.polyfit(x, y, 1)[0])
        else:
            consol_volume_trend = 0
        
        # Days near high (bullish consolidation)
        high_threshold = high_max * 0.95
        consol_days_near_high = int(np.sum(consol_window['Close'] > high_threshold))
        
        print(f"✅ Calculated consolidation features:")
        print(f"   Range: {consol_range_pct:.2f}%, Position: {consol_position_in_range:.3f}")
        print(f"   Volatility: {consol_volatility:.4f}, Volume trend: {consol_volume_trend:.2f}")
        print(f"   Days near high: {consol_days_near_high}\n")
        
    except Exception as e:
        print(f"❌ Error downloading daily data: {e}")
        return None
    
    # Get 5-minute intraday data from TwelveData
    try:
        # Delete cache to force fresh download
        cache_file = td_downloader.cache_dir / f"{ticker}_{date_str}_5min.parquet"
        if cache_file.exists():
            cache_file.unlink()
            print(f"🗑️  Cleared cache for {ticker} {date_str}")
        
        intraday_data = td_downloader.download_minute_data(ticker, date_str, timeframe='5min')
        if intraday_data is None or intraday_data.empty:
            print(f"❌ No intraday data found for {ticker} on {date_str}")
            return None
        print(f"✅ Downloaded {len(intraday_data)} 5-minute bars from TwelveData\n")
    except Exception as e:
        print(f"❌ Error downloading intraday data: {e}")
        return None

    results = []
    for i, (candle_time, candle_row) in enumerate(intraday_data.iterrows()):
        # CRITICAL: Only use data UP TO this candle (simulate real-time)
        data_up_to_candle = intraday_data.iloc[:i+1]
        high_of_day = data_up_to_candle['High'].max()
        low_of_day = data_up_to_candle['Low'].min()
        entry_price = high_of_day * 1.005
        stop_price = low_of_day
        risk = entry_price - stop_price
        if risk <= 0:
            continue
        target_price = entry_price + (5 * risk)
        
        # IMPORTANT: Manually calculate features using ONLY data up to this candle
        # Cannot use extractor.extract_features() as it downloads full day data (lookahead bias)
        try:
            # Calculate intraday features manually from data_up_to_candle
            first_candle = data_up_to_candle.iloc[0]
            gap_pct = ((first_candle['Open'] - low_of_day) / low_of_day) * 100 if low_of_day > 0 else 0
            first_candle_range_pct = ((first_candle['High'] - first_candle['Low']) / first_candle['Low']) * 100 if first_candle['Low'] > 0 else 0
            first_candle_body_ratio = abs(first_candle['Close'] - first_candle['Open']) / (first_candle['High'] - first_candle['Low']) if (first_candle['High'] - first_candle['Low']) > 0 else 0
            first_candle_close_position = (first_candle['Close'] - first_candle['Low']) / (first_candle['High'] - first_candle['Low']) if (first_candle['High'] - first_candle['Low']) > 0 else 0.5
            
            # Volume features
            avg_volume = data_up_to_candle['Volume'].mean()
            first_candle_volume_ratio = first_candle['Volume'] / avg_volume if avg_volume > 0 else 1.0
            
            # Early high flag (high made in first 30 minutes)
            first_30min = data_up_to_candle.iloc[:6]  # 6 candles = 30 minutes
            early_high_flag = 1 if high_of_day == first_30min['High'].max() else 0
            
            # Entry timing
            market_open_time = candle_time.replace(hour=9, minute=30) if hasattr(candle_time, 'replace') else datetime.strptime(f"{date_str} 09:30", '%Y-%m-%d %H:%M')
            minutes_since_open = (candle_time - market_open_time).total_seconds() / 60 if hasattr(candle_time, 'replace') else 0
            
            # Entry position in current bar
            entry_position_in_bar = (entry_price - candle_row['Low']) / (candle_row['High'] - candle_row['Low']) if (candle_row['High'] - candle_row['Low']) > 0 else 0.5
            
            # Entry vs day high
            entry_vs_day_high_pct = ((entry_price - high_of_day) / high_of_day) * 100 if high_of_day > 0 else 0
            
            # Risk metrics
            risk_per_share = entry_price - stop_price
            risk_pct = (risk / entry_price) * 100
            risk_reward_ratio = 5.0
            
            # Build feature dict (matching Pipeline 2 feature names)
            # Uses consolidation features from yfinance + intraday features from TwelveData
            features_dict = {
                # Consolidation features (from yfinance daily data)
                'consol_range_pct': round(consol_range_pct, 2),
                'consol_position_in_range': round(consol_position_in_range, 3),
                'consol_volatility': round(consol_volatility, 4),
                'consol_volume_trend': round(consol_volume_trend, 2),
                'consol_days_near_high': consol_days_near_high,
                # Intraday features (from TwelveData minute data)
                'gap_pct': gap_pct,
                'first_candle_range_pct': first_candle_range_pct,
                'first_candle_body_ratio': first_candle_body_ratio,
                'first_candle_close_position': first_candle_close_position,
                'first_candle_volume_ratio': first_candle_volume_ratio,
                'early_high_flag': early_high_flag,
                'risk_per_share': risk_per_share,
                'risk_pct': risk_pct,
                'risk_reward_ratio': risk_reward_ratio,
                'minutes_since_open': minutes_since_open,
                'entry_position_in_bar': entry_position_in_bar,
                'entry_vs_day_high_pct': entry_vs_day_high_pct,
            }
        except Exception as e:
            print(f"❌ Exception in feature calculation: {e}")
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
        test_pipeline2_full_day(ticker, date_str)
        return
    
    # Interactive mode
    print("\n" + "="*80)
    print("Pipeline 2 Full Day Testing Tool")
    print("="*80)
    print("\nUsage: python test_pipeline2_day.py TICKER DATE")
    print("Example: python test_pipeline2_day.py RGTI 2025-10-02")
    print("\n" + "="*80 + "\n")
    
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
            
            # Validate date format
            try:
                datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                print("❌ Invalid date format. Use YYYY-MM-DD\n")
                continue
            
            # Run test
            test_pipeline2_full_day(ticker, date_str)
            
            print("\n" + "-"*80)
            
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


if __name__ == '__main__':
    main()

