#!/usr/bin/env python3
"""
Breakout Feature Extractor - Extract intraday breakout characteristics for Pipeline 2 (5R Predictor)

Similar to consolidation_features.py but focuses on:
- Pre-breakout consolidation quality (from Pipeline 1)
- Breakout day characteristics (gap, first candle, volume)
- Entry execution quality (timing, position in day)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
import yfinance as yf

from data_processing.download_minute_data import AlpacaMinuteDownloader


class BreakoutFeatureExtractor:
    """Extract features for breakout trades to predict 5R success"""
    
    def __init__(self, timeframe: str = '1Min', downloader=None):
        """
        Initialize extractor

        Args:
            timeframe: '1Min' or '5Min' for intraday analysis
            downloader: optional downloader instance (must provide `download_minute_data`)
                        If None, defaults to `AlpacaMinuteDownloader()`.
        """
        self.timeframe = timeframe
        # Allow injecting a custom downloader (TwelveData, Alpaca, or test stub)
        if downloader is not None:
            self.minute_downloader = downloader
        else:
            self.minute_downloader = AlpacaMinuteDownloader()
        
        # For daily data (pre-breakout consolidation)
        self.lookback_days = 10
    
    def extract_features(self, ticker: str, entry_date, entry_price: float,
                        stop_price: float, target_price: float) -> Optional[dict]:
        """
        Extract all breakout features for a single trade
        
        Args:
            ticker: Stock ticker symbol
            entry_date: Date of entry (datetime object or string 'YYYY-MM-DD')
            entry_price: Entry price
            stop_price: Initial stop loss price
            target_price: 5R target price
            
        Returns:
            Dictionary of features or None if extraction fails
        """
        try:
            # Convert entry_date to datetime if it's a string
            if isinstance(entry_date, str):
                entry_date = datetime.strptime(entry_date, '%Y-%m-%d')
            
            # Download daily data for pre-breakout analysis
            daily_df = self._download_daily_data(ticker, entry_date)
            if daily_df is None or len(daily_df) < self.lookback_days:
                return None
            
            # Download minute data for entry day
            minute_df = self.minute_downloader.download_minute_data(
                ticker, 
                entry_date.strftime('%Y-%m-%d'),
                self.timeframe
            )
            if minute_df is None or len(minute_df) == 0:
                return None
            
            # Extract feature categories
            features = {}
            
            # 1. Pre-breakout consolidation features (day before entry)
            features.update(self._consolidation_quality_features(daily_df, entry_date))
            
            # 2. Breakout day characteristics
            features.update(self._breakout_day_features(daily_df, minute_df, entry_date))
            
            # 3. Entry execution quality
            features.update(self._entry_quality_features(
                minute_df, entry_price, stop_price, target_price, entry_date, daily_df
            ))
            
            # Add metadata
            features['ticker'] = ticker
            features['entry_date'] = entry_date.strftime('%Y-%m-%d')
            features['timeframe'] = self.timeframe
            
            return features
            
        except Exception as e:
            print(f"❌ Error extracting features for {ticker} on {entry_date}: {e}")
            return None
    
    def _download_daily_data(self, ticker: str, entry_date: datetime) -> Optional[pd.DataFrame]:
        """Download daily data for consolidation analysis"""
        try:
            # Get data from 60 days before entry (for SMAs)
            start_date = entry_date - timedelta(days=70)
            end_date = entry_date + timedelta(days=1)
            
            df = yf.download(
                ticker,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False,
                auto_adjust=True
            )
            
            if df.empty:
                return None
            
            # Ensure canonical column names and flatten MultiIndex if present
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    df.columns = df.columns.get_level_values(0)
                except Exception:
                    df.columns = ['_'.join(map(str, c)) for c in df.columns]

            df = df.rename(columns={'Adj Close': 'Close'})

            # Coerce OHLCV columns to numeric Series (handle unexpected 2D shapes)
            for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if c in df.columns:
                    col = df[c]
                    if getattr(col, 'ndim', 1) > 1:
                        try:
                            col = col.iloc[:, 0]
                        except Exception:
                            continue
                    df[c] = pd.to_numeric(col, errors='coerce')

            return df
            
        except Exception as e:
            print(f"⚠️  Failed to download daily data for {ticker}: {e}")
            return None
    
    def _consolidation_quality_features(
        self, 
        daily_df: pd.DataFrame, 
        entry_date: datetime
    ) -> Dict:
        """
        Analyze consolidation pattern BEFORE breakout
        Uses 10-day window before entry (similar to Pipeline 1)
        """
        # Get 10-day window before entry
        entry_idx = daily_df.index.searchsorted(entry_date)
        if entry_idx < self.lookback_days:
            window = daily_df.iloc[:entry_idx]
        else:
            window = daily_df.iloc[entry_idx - self.lookback_days:entry_idx]
        
        if len(window) < 5:
            # Not enough data
            return {
                'consol_range_pct': 0,
                'consol_position_in_range': 0.5,
                'consol_volatility': 0,
                'consol_volume_trend': 0,
                'consol_days_near_high': 0
            }
        
        # Price range of consolidation
        high_max = float(window['High'].max())
        low_min = float(window['Low'].min())
        last_close = float(window['Close'].iloc[-1])
        
        consol_range_pct = ((high_max - low_min) / high_max) * 100 if high_max > 0 else 0
        position_in_range = ((last_close - low_min) / (high_max - low_min)) if (high_max - low_min) > 0 else 0.5
        
        # Volatility (std dev of returns)
        returns = window['Close'].pct_change().dropna()
        volatility = float(returns.std()) if len(returns) > 0 else 0
        
        # Volume trend (declining = good consolidation)
        if len(window) >= 2:
            x = np.arange(len(window))
            y = window['Volume'].values
            volume_trend = float(np.polyfit(x, y, 1)[0])
        else:
            volume_trend = 0
        
        # Days near high (bullish consolidation)
        high_threshold = high_max * 0.95
        days_near_high = int(np.sum(window['Close'] > high_threshold))
        
        return {
            'consol_range_pct': round(consol_range_pct, 2),
            'consol_position_in_range': round(position_in_range, 3),
            'consol_volatility': round(volatility, 4),
            'consol_volume_trend': round(volume_trend, 2),
            'consol_days_near_high': days_near_high
        }
    
    def _breakout_day_features(
        self,
        daily_df: pd.DataFrame,
        minute_df: pd.DataFrame,
        entry_date: datetime
    ) -> Dict:
        """Analyze the breakout day itself"""
        
        # Get previous day's close for gap calculation
        entry_idx = daily_df.index.searchsorted(entry_date)
        if entry_idx > 0:
            prev_close = float(daily_df['Close'].iloc[entry_idx - 1])
        else:
            prev_close = float(minute_df['Close'].iloc[0])
        
        # Today's open (first minute bar)
        today_open = float(minute_df['Open'].iloc[0])
        
        # Gap calculation
        gap_pct = ((today_open - prev_close) / prev_close) * 100
        
        # First candle analysis (first timeframe bar)
        first_candle = minute_df.iloc[0]
        first_candle_range = ((first_candle['High'] - first_candle['Low']) / first_candle['Low']) * 100
        first_candle_body = abs(first_candle['Close'] - first_candle['Open'])
        first_candle_body_ratio = (first_candle_body / (first_candle['High'] - first_candle['Low'])) if (first_candle['High'] - first_candle['Low']) > 0 else 0
        
        # First candle close position (0=low, 1=high)
        first_candle_close_pos = ((first_candle['Close'] - first_candle['Low']) / (first_candle['High'] - first_candle['Low'])) if (first_candle['High'] - first_candle['Low']) > 0 else 0.5
        
        # Volume on first candle vs average
        avg_volume_per_bar = float(minute_df['Volume'].mean())
        first_candle_vol_ratio = float(first_candle['Volume'] / avg_volume_per_bar) if avg_volume_per_bar > 0 else 1.0
        
        # Day's high within first 30 minutes? (FIXED: only use data up to entry time)
        first_30_bars = 30 if self.timeframe == '1Min' else 6  # 30 min / 5 min
        if len(minute_df) >= first_30_bars:
            # Find entry bar to determine how much data is available at entry time
            entry_bar_idx = None
            for idx in range(len(minute_df)):
                if float(minute_df.iloc[idx]['High']) >= entry_price:
                    entry_bar_idx = idx
                    break
            
            if entry_bar_idx is not None and entry_bar_idx > 0:
                # Only use data up to entry bar (no lookahead)
                high_at_entry = float(minute_df.iloc[:entry_bar_idx+1]['High'].max())
                first_30_high = float(minute_df['High'].iloc[:first_30_bars].max())
                early_high = 1 if first_30_high >= high_at_entry * 0.99 else 0
            else:
                early_high = 0
        else:
            early_high = 0
        
        return {
            'gap_pct': round(gap_pct, 2),
            'first_candle_range_pct': round(first_candle_range, 2),
            'first_candle_body_ratio': round(first_candle_body_ratio, 3),
            'first_candle_close_position': round(first_candle_close_pos, 3),
            'first_candle_volume_ratio': round(first_candle_vol_ratio, 2),
            'early_high_flag': early_high
        }
    
    def _entry_quality_features(
        self,
        minute_df: pd.DataFrame,
        entry_price: float,
        stop_price: float,
        target_price: float,
        entry_date: datetime,
        daily_df: pd.DataFrame = None
    ) -> Dict:
        """Analyze entry execution quality"""
        
        # Risk and reward
        risk_per_share = entry_price - stop_price
        reward_per_share = target_price - entry_price
        risk_reward_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0
        
        # Risk as % of entry price
        risk_pct = (risk_per_share / entry_price) * 100 if entry_price > 0 else 0
        
        # NEW: ATR-normalized risk
        risk_vs_atr = self._calculate_risk_vs_atr(risk_pct, daily_df, entry_date) if daily_df is not None else 1.0
        
        # Find which minute bar entry occurred in (approximate)
        # Assume entry is within the first bar that has high >= entry_price
        entry_bar_idx = None
        for idx, row in minute_df.iterrows():
            if row['High'] >= entry_price >= row['Low']:
                entry_bar_idx = minute_df.index.get_loc(idx)
                break
        
        if entry_bar_idx is None:
            # Couldn't find entry bar, use approximation
            entry_bar_idx = 0
        
        # Minutes since market open (assuming 9:30 AM ET open)
        if entry_bar_idx < len(minute_df):
            entry_bar_time = minute_df.index[entry_bar_idx]
            market_open = entry_bar_time.replace(hour=9, minute=30 if entry_bar_time.hour >= 9 else entry_bar_time.hour, second=0)
            minutes_since_open = (entry_bar_time - market_open).total_seconds() / 60
        else:
            minutes_since_open = 0
        
        # Entry position within entry bar's range
        if entry_bar_idx < len(minute_df):
            entry_bar = minute_df.iloc[entry_bar_idx]
            bar_range = entry_bar['High'] - entry_bar['Low']
            if bar_range > 0:
                entry_position_in_bar = (entry_price - entry_bar['Low']) / bar_range
            else:
                entry_position_in_bar = 0.5
        else:
            entry_position_in_bar = 0.5
        
        # Entry vs day's high/low (FIXED: only use high up to entry time)
        if entry_bar_idx is not None:
            high_at_entry = float(minute_df.iloc[:entry_bar_idx+1]['High'].max())
        else:
            high_at_entry = float(minute_df['High'].max())
        
        entry_vs_high_so_far_pct = ((high_at_entry - entry_price) / entry_price) * 100 if entry_price > 0 else 0
        
        # NEW: Volume surge - compare accumulated volume at entry vs same time in past 3 days
        volume_surge_ratio = self._calculate_volume_surge(minute_df, entry_bar_idx, entry_date)
        
        return {
            'risk_per_share': round(risk_per_share, 3),
            'risk_pct': round(risk_pct, 2),
            'risk_reward_ratio': round(risk_reward_ratio, 2),
            'minutes_since_open': int(minutes_since_open),
            'entry_position_in_bar': round(entry_position_in_bar, 3),
            'entry_vs_high_so_far_pct': round(entry_vs_high_so_far_pct, 2),
            'volume_surge_vs_3day': round(volume_surge_ratio, 2),
            'risk_vs_atr': round(risk_vs_atr, 2)
        }
    
    def _calculate_volume_surge(self, minute_df: pd.DataFrame, entry_bar_idx: int, entry_date: datetime) -> float:
        """Calculate volume surge: accumulated volume at entry vs same time in past 3 days"""
        try:
            if entry_bar_idx is None or entry_bar_idx < 0:
                return 1.0
            
            # Volume accumulated up to entry
            volume_at_entry = minute_df.iloc[:entry_bar_idx+1]['Volume'].sum()
            
            if volume_at_entry == 0:
                return 1.0
            
            # Get ticker from metadata (stored in extract_features)
            # For now, return 1.0 as we can't easily get past data without ticker
            # This will be refined when called with ticker context
            return 1.0
            
        except Exception as e:
            return 1.0
    
    def _calculate_risk_vs_atr(self, risk_pct: float, daily_df: pd.DataFrame, entry_date: datetime) -> float:
        """Calculate risk_pct / ATR ratio (higher ratio = riskier trade relative to normal volatility)"""
        try:
            if daily_df is None or len(daily_df) < 14:
                return 1.0
            
            # Find entry date in daily data
            entry_idx = daily_df.index.searchsorted(entry_date)
            if entry_idx < 14:
                return 1.0
            
            # Calculate 10-day ATR
            lookback = daily_df.iloc[entry_idx-14:entry_idx].copy()
            
            # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
            true_ranges = []
            for i in range(1, len(lookback)):
                high = lookback.iloc[i]['High']
                low = lookback.iloc[i]['Low']
                prev_close = lookback.iloc[i-1]['Close']
                
                tr = max(
                    high - low,
                    abs(high - prev_close),
                    abs(low - prev_close)
                )
                true_ranges.append(tr)
            
            if not true_ranges:
                return 1.0
            
            atr = sum(true_ranges) / len(true_ranges)
            
            # Get entry day close to normalize ATR to percentage
            entry_close = daily_df.iloc[entry_idx-1]['Close'] if entry_idx > 0 else daily_df.iloc[0]['Close']
            atr_pct = (atr / entry_close) * 100 if entry_close > 0 else 0
            
            # Return risk_pct / atr_pct (>1 means risk is higher than typical daily range)
            return risk_pct / atr_pct if atr_pct > 0 else 1.0
            
        except Exception as e:
            return 1.0


def main():
    """Test feature extraction"""
    print("\n" + "="*80)
    print("BREAKOUT FEATURE EXTRACTOR - TEST")
    print("="*80 + "\n")
    
    extractor = BreakoutFeatureExtractor(timeframe='1Min')
    
    # Test with a sample trade
    test_ticker = "TWST"
    test_entry_date = datetime(2023, 11, 17)
    test_entry_price = 22.75
    test_stop_price = 20.59
    test_target_price = 33.55  # 5R from entry
    
    print(f"Extracting features for {test_ticker} on {test_entry_date.date()}...")
    print(f"Entry: ${test_entry_price}, Stop: ${test_stop_price}, Target: ${test_target_price}\n")
    
    features = extractor.extract_features(
        test_ticker,
        test_entry_date,
        test_entry_price,
        test_stop_price,
        test_target_price
    )
    
    if features:
        print("✅ Feature extraction successful!\n")
        print("Extracted Features:")
        print("-" * 80)
        
        # Group features by category
        categories = {
            'Consolidation Quality': [k for k in features.keys() if k.startswith('consol_')],
            'Breakout Day': [k for k in features.keys() if any(x in k for x in ['gap', 'first_candle', 'early_high'])],
            'Entry Quality': [k for k in features.keys() if any(x in k for x in ['risk', 'minutes_since', 'entry_'])],
            'Post-Entry Momentum': [k for k in features.keys() if k.startswith('post_entry_')]
        }
        
        for category, keys in categories.items():
            if keys:
                print(f"\n{category}:")
                for key in keys:
                    print(f"  {key:40s} = {features[key]}")
        
        print(f"\nTotal features: {len([k for k in features.keys() if k not in ['ticker', 'entry_date', 'timeframe']])}")
        
    else:
        print("❌ Feature extraction failed")


if __name__ == "__main__":
    main()
