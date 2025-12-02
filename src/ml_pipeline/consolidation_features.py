#!/usr/bin/env python3
"""
Feature Extraction for Consolidation Detection (Pipeline 1)

Extracts technical features from 10-day lookback window to detect
valid consolidation patterns ready to break out.

Features extracted:
- Price action (range, volatility, position)
- Volume patterns (trend, spikes)
- SMA relationships (10/20/50)
- Momentum indicators
- Consolidation characteristics
"""

import json
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent / 'data_processing'))
from Trade_Analyzer import fetch_historical_data


class ConsolidationFeatureExtractor:
    """Extract features from 10-day price data for consolidation detection."""
    
    def __init__(self, lookback_days: int = 10):
        """
        Args:
            lookback_days: Number of days to look back (default 10)
        """
        self.lookback_days = lookback_days
    
    def extract_features(self, ticker: str, date: datetime, data: pd.DataFrame = None) -> Optional[Dict]:
        """
        Extract all features for a given ticker/date.
        
        Args:
            ticker: Stock ticker
            date: Date to extract features for
            data: Optional pre-loaded DataFrame (if None, will fetch)
        
        Returns:
            Dictionary of features, or None if insufficient data
        """
        # Fetch data if not provided
        if data is None:
            start_date = date - timedelta(days=self.lookback_days + 80)  # Extra for SMAs (was 60)
            end_date = date + timedelta(days=1)
            
            try:
                df = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=True
                )
                
                # Flatten MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                if df is None or len(df) < self.lookback_days + 50:
                    return None
            except:
                return None
        else:
            df = data.copy()
        
        # Calculate technical indicators
        df = self._add_indicators(df)
        
        # Get the lookback window (10 days before date)
        date_idx = df.index.get_indexer([date], method='nearest')[0]
        if date_idx < self.lookback_days + 50:  # Need room for SMAs
            return None
        
        window_start = date_idx - self.lookback_days
        window = df.iloc[window_start:date_idx]
        
        if len(window) < self.lookback_days:
            return None
        
        # Extract all feature categories
        features = {}
        features.update(self._price_features(window))
        features.update(self._volume_features(window))
        features.update(self._sma_features(window, df.iloc[:date_idx]))
        features.update(self._momentum_features(window))
        features.update(self._consolidation_features(window))
        
        # Add metadata
        features['ticker'] = ticker
        features['date'] = date.strftime('%Y-%m-%d')
        
        return features
    
    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators."""
        # Ensure we have 1D Series for Close/High/Low/Volume (handle MultiIndex)
        close_series = df['Close']
        if isinstance(close_series, pd.DataFrame):
            close_series = close_series.iloc[:, 0]
        
        high_series = df['High']
        if isinstance(high_series, pd.DataFrame):
            high_series = high_series.iloc[:, 0]
        
        low_series = df['Low']
        if isinstance(low_series, pd.DataFrame):
            low_series = low_series.iloc[:, 0]
        
        volume_series = df['Volume']
        if isinstance(volume_series, pd.DataFrame):
            volume_series = volume_series.iloc[:, 0]
        
        # SMAs
        df['SMA_10'] = close_series.rolling(10).mean()
        df['SMA_20'] = close_series.rolling(20).mean()
        df['SMA_50'] = close_series.rolling(50).mean()
        
        # Volume average
        df['Volume_Avg_20'] = volume_series.rolling(20).mean()
        
        # ATR (Average True Range)
        df['High_Low'] = high_series - low_series
        df['High_Close'] = abs(high_series - close_series.shift())
        df['Low_Close'] = abs(low_series - close_series.shift())
        df['TR'] = df[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
        df['ATR_14'] = df['TR'].rolling(14).mean()
        
        # Returns
        df['Return_1d'] = close_series.pct_change()
        
        return df
    
    def _price_features(self, window: pd.DataFrame) -> Dict:
        """Extract price action features."""
        high = float(window['High'].max())
        low = float(window['Low'].min())
        close = float(window['Close'].iloc[-1])
        open_price = float(window['Open'].iloc[0])
        
        price_range = high - low
        range_pct = (price_range / low) * 100 if low > 0 else 0
        
        # Position in range (0 = at low, 1 = at high)
        position_in_range = (close - low) / price_range if price_range > 0 else 0.5
        
        # Price change over window
        price_change_pct = ((close - open_price) / open_price) * 100 if open_price > 0 else 0
        
        # Volatility (std of daily returns)
        volatility = float(window['Return_1d'].std()) if 'Return_1d' in window.columns else 0
        
        # Average daily range
        avg_daily_range = float(((window['High'] - window['Low']) / window['Close']).mean() * 100)
        
        return {
            'price_high': float(high),
            'price_low': float(low),
            'price_close': float(close),
            'price_range_pct': float(range_pct),
            'position_in_range': float(position_in_range),
            'price_change_pct': float(price_change_pct),
            'volatility': float(volatility),
            'avg_daily_range_pct': float(avg_daily_range)
        }
    
    def _volume_features(self, window: pd.DataFrame) -> Dict:
        """Extract volume pattern features."""
        volumes = window['Volume'].values
        
        # Volume trend (declining = good for consolidation)
        volume_trend = np.polyfit(range(len(volumes)), volumes, 1)[0] if len(volumes) > 1 else 0
        
        # Relative volume (current vs average)
        current_volume = volumes[-1]
        avg_volume = volumes.mean()
        relative_volume = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Volume volatility
        volume_std = np.std(volumes) / avg_volume if avg_volume > 0 else 0
        
        # Volume spike count (days with 2x average)
        volume_spikes = np.sum(volumes > avg_volume * 2)
        
        return {
            'volume_trend': float(volume_trend),
            'relative_volume': float(relative_volume),
            'volume_volatility': float(volume_std),
            'volume_spike_count': int(volume_spikes),
            'avg_volume': float(avg_volume)
        }
    
    def _sma_features(self, window: pd.DataFrame, full_history: pd.DataFrame) -> Dict:
        """Extract SMA relationship features."""
        close = float(window['Close'].iloc[-1])
        closes = window['Close'].values
        lows = window['Low'].values
        highs = window['High'].values
        
        # Distance from SMAs
        sma_10 = float(full_history['SMA_10'].iloc[-1]) if not pd.isna(full_history['SMA_10'].iloc[-1]) else 0
        sma_20 = float(full_history['SMA_20'].iloc[-1]) if not pd.isna(full_history['SMA_20'].iloc[-1]) else 0
        sma_50 = float(full_history['SMA_50'].iloc[-1]) if not pd.isna(full_history['SMA_50'].iloc[-1]) else 0
        
        dist_sma10_pct = ((close - sma_10) / sma_10) * 100 if not pd.isna(sma_10) and sma_10 > 0 else 0
        dist_sma20_pct = ((close - sma_20) / sma_20) * 100 if not pd.isna(sma_20) and sma_20 > 0 else 0
        dist_sma50_pct = ((close - sma_50) / sma_50) * 100 if not pd.isna(sma_50) and sma_50 > 0 else 0
        
        # NEW: 10 SMA Hugging Features
        sma_10_values = full_history['SMA_10'].iloc[-len(window):].values
        
        # How many days price stayed within 3% of 10 SMA (tight hugging)
        days_hugging_10sma = 0
        avg_dist_from_10sma = 0
        max_dist_from_10sma = 0
        
        if not pd.isna(sma_10_values).all():
            distances = np.abs((closes - sma_10_values) / sma_10_values) * 100
            distances = distances[~np.isnan(distances)]
            if len(distances) > 0:
                days_hugging_10sma = np.sum(distances <= 3)  # Within 3%
                avg_dist_from_10sma = np.mean(distances)
                max_dist_from_10sma = np.max(distances)
        
        # How many days the LOW touched/tested the 10 SMA (price bouncing off support)
        days_testing_10sma = 0
        if not pd.isna(sma_10_values).all():
            for i, (low, high, sma) in enumerate(zip(lows, highs, sma_10_values)):
                if not np.isnan(sma):
                    # Low came within 2% of 10 SMA or touched it
                    if low <= sma * 1.02 and high >= sma * 0.98:
                        days_testing_10sma += 1
        
        # Percentage of time spent above 10 SMA during the window
        pct_above_10sma = 0
        if not pd.isna(sma_10_values).all():
            above_count = np.sum(closes > sma_10_values[~np.isnan(sma_10_values)])
            pct_above_10sma = above_count / len(closes) if len(closes) > 0 else 0
        
        # SMA alignment (1 = bullish stack, -1 = bearish stack)
        sma_alignment = 0
        if not pd.isna(sma_10) and not pd.isna(sma_20) and not pd.isna(sma_50):
            if sma_10 > sma_20 > sma_50:
                sma_alignment = 1  # Bullish
            elif sma_10 < sma_20 < sma_50:
                sma_alignment = -1  # Bearish
        
        # Price above/below SMAs
        above_sma10 = 1 if close > sma_10 else 0
        above_sma20 = 1 if close > sma_20 else 0
        above_sma50 = 1 if close > sma_50 else 0
        
        return {
            'dist_from_sma10_pct': float(dist_sma10_pct),
            'dist_from_sma20_pct': float(dist_sma20_pct),
            'dist_from_sma50_pct': float(dist_sma50_pct),
            'sma_alignment': int(sma_alignment),
            'above_sma10': int(above_sma10),
            'above_sma20': int(above_sma20),
            'above_sma50': int(above_sma50),
            # NEW: 10 SMA Hugging Features
            'days_hugging_10sma': int(days_hugging_10sma),
            'avg_dist_from_10sma_pct': float(avg_dist_from_10sma),
            'max_dist_from_10sma_pct': float(max_dist_from_10sma),
            'days_testing_10sma': int(days_testing_10sma),
            'pct_above_10sma': float(pct_above_10sma)
        }
    
    def _momentum_features(self, window: pd.DataFrame) -> Dict:
        """Extract momentum indicators."""
        closes = window['Close'].values
        
        # Rate of change over window
        roc = ((closes[-1] - closes[0]) / closes[0]) * 100 if closes[0] > 0 else 0
        
        # Days up vs days down
        returns = np.diff(closes) / closes[:-1]
        up_days = np.sum(returns > 0)
        down_days = np.sum(returns < 0)
        
        # Consecutive up/down days at end
        consecutive_up = 0
        consecutive_down = 0
        for i in range(len(returns) - 1, -1, -1):
            if returns[i] > 0:
                consecutive_up += 1
            else:
                break
        for i in range(len(returns) - 1, -1, -1):
            if returns[i] < 0:
                consecutive_down += 1
            else:
                break
        
        return {
            'roc_pct': float(roc),
            'up_days': int(up_days),
            'down_days': int(down_days),
            'consecutive_up': int(consecutive_up),
            'consecutive_down': int(consecutive_down)
        }
    
    def _consolidation_features(self, window: pd.DataFrame) -> Dict:
        """Extract consolidation-specific features."""
        highs = window['High'].values
        lows = window['Low'].values
        closes = window['Close'].values
        
        # Consolidation tightness (how much is price compressed)
        high_max = np.max(highs)
        low_min = np.min(lows)
        consolidation_range_pct = ((high_max - low_min) / high_max) * 100 if high_max > 0 else 0
        
        # Days near high of range
        high_threshold = high_max * 0.95  # Within 5% of high
        days_near_high = np.sum(closes > high_threshold)
        
        # Days near low of range
        low_threshold = low_min * 1.05  # Within 5% of low
        days_near_low = np.sum(closes < low_threshold)
        
        # Range contraction (first half vs second half)
        mid_point = len(window) // 2
        first_half_range = (highs[:mid_point].max() - lows[:mid_point].min())
        second_half_range = (highs[mid_point:].max() - lows[mid_point:].min())
        range_contraction = (first_half_range - second_half_range) / first_half_range if first_half_range > 0 else 0
        
        # ATR trend (declining ATR = tightening)
        if 'ATR_14' in window.columns:
            atr_values = window['ATR_14'].dropna().values
            if len(atr_values) >= 2:
                atr_trend = np.polyfit(range(len(atr_values)), atr_values, 1)[0]
            else:
                atr_trend = 0
        else:
            atr_trend = 0
        
        return {
            'consolidation_range_pct': float(consolidation_range_pct),
            'days_near_high': int(days_near_high),
            'days_near_low': int(days_near_low),
            'range_contraction': float(range_contraction),
            'atr_trend': float(atr_trend)
        }
    
    def extract_from_positive_trades(self, trades_file: str, output_file: str):
        """
        Extract features from positive trade examples.
        
        Args:
            trades_file: Path to JSON file with positive trades
            output_file: Path to save extracted features
        """
        print("=" * 70)
        print("EXTRACTING FEATURES FROM POSITIVE EXAMPLES")
        print("=" * 70)
        
        # Load trades
        with open(trades_file, 'r') as f:
            trades = json.load(f)
        
        print(f"\nLoaded {len(trades)} positive trades")
        print(f"Extracting features from 10-day lookback window...")
        
        features_list = []
        failed = 0
        
        for i, trade in enumerate(trades):
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(trades)} trades...")
            
            ticker = trade['ticker']
            entry_date_str = trade['entry_date'].split('T')[0]
            entry_date = datetime.strptime(entry_date_str, '%Y-%m-%d')
            
            # Extract features
            features = self.extract_features(ticker, entry_date)
            
            if features is not None:
                features['label'] = 1  # Positive label
                features_list.append(features)
            else:
                failed += 1
        
        print(f"\n✅ Extracted features from {len(features_list)} trades")
        print(f"❌ Failed to extract {failed} trades (insufficient data)")
        
        # Save to JSONL
        with open(output_file, 'w') as f:
            for features in features_list:
                f.write(json.dumps(features) + '\n')
        
        print(f"\n💾 Saved features to: {output_file}")
        
        return features_list
    
    def extract_from_negative_examples(self, negatives_file: str, output_file: str):
        """
        Extract features from negative examples.
        
        Args:
            negatives_file: Path to JSONL file with negative examples
            output_file: Path to save extracted features
        """
        print("=" * 70)
        print("EXTRACTING FEATURES FROM NEGATIVE EXAMPLES")
        print("=" * 70)
        
        # Load negatives
        negatives = []
        with open(negatives_file, 'r') as f:
            for line in f:
                if line.strip():
                    negatives.append(json.loads(line))
        
        print(f"\nLoaded {len(negatives)} negative examples")
        print(f"Extracting features from 10-day lookback window...")
        
        features_list = []
        failed = 0
        
        for i, negative in enumerate(negatives):
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(negatives)} examples...")
            
            ticker = negative['ticker']
            date = datetime.strptime(negative['date'], '%Y-%m-%d')
            
            # Extract features
            features = self.extract_features(ticker, date)
            
            if features is not None:
                features['label'] = 0  # Negative label
                features['negative_category'] = negative.get('category', 'UNKNOWN')
                features_list.append(features)
            else:
                failed += 1
        
        print(f"\n✅ Extracted features from {len(features_list)} examples")
        print(f"❌ Failed to extract {failed} examples (insufficient data)")
        
        # Save to JSONL
        with open(output_file, 'w') as f:
            for features in features_list:
                f.write(json.dumps(features) + '\n')
        
        print(f"\n💾 Saved features to: {output_file}")
        
        return features_list


if __name__ == '__main__':
    # Initialize extractor
    extractor = ConsolidationFeatureExtractor(lookback_days=10)
    
    # Extract features from positives
    print("\n🔵 Step 1: Extracting features from POSITIVE trades...\n")
    positive_features = extractor.extract_from_positive_trades(
        trades_file='data/raw/parsed_trade_entries_initial.json',
        output_file='data/processed/features_positive.jsonl'
    )
    
    # Extract features from negatives
    print("\n\n🔴 Step 2: Extracting features from NEGATIVE examples...\n")
    negative_features = extractor.extract_from_negative_examples(
        negatives_file='data/processed/negative_labels_consolidation_verified.jsonl',
        output_file='data/processed/features_negative.jsonl'
    )
    
    print("\n" + "=" * 70)
    print("✨ FEATURE EXTRACTION COMPLETE!")
    print("=" * 70)
    print(f"\nPositive features: {len(positive_features)}")
    print(f"Negative features: {len(negative_features)}")
    print(f"Total dataset size: {len(positive_features) + len(negative_features)}")
    print(f"\nNext step: Train the model using these features")
