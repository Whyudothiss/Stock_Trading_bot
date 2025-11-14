import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time
import os
from typing import Dict, List, Tuple, Optional

class BullFlagDetector:
    """
    Rule-based Bull Flag pattern detector based on the trading strategy:
    1. Big move up 30%+ over multiple days/weeks
    2. 10/20 SMA inclining
    3. Orderly pullback with higher lows and lower highs
    4. Volume drying up during pullback
    5. Ready for breakout on high volume
    """
    
    def __init__(self):
        self.min_price = 1.0
        self.min_adr_percent = 5.0
        self.min_daily_volume = 3500000  # $3.5M
        self.min_big_move_percent = 30.0
        self.lookback_days = 90
        
    def get_stock_data(self, symbol: str, period: str = "6mo") -> Optional[pd.DataFrame]:
        """Fetch stock data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                return None
                
            # Calculate indicators
            data['SMA_10'] = data['Close'].rolling(window=10, min_periods=10).mean()
            data['SMA_20'] = data['Close'].rolling(window=20, min_periods=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50, min_periods=50).mean()
            data['Volume_MA'] = data['Volume'].rolling(window=20, min_periods=20).mean()
            
            # Calculate daily dollar volume
            data['Dollar_Volume'] = data['Close'] * data['Volume']
            data['Dollar_Volume_MA'] = data['Dollar_Volume'].rolling(window=20, min_periods=20).mean()
            
            # Calculate ATR for ADR%
            data['High_Low'] = data['High'] - data['Low']
            data['High_Close'] = abs(data['High'] - data['Close'].shift(1))
            data['Low_Close'] = abs(data['Low'] - data['Close'].shift(1))
            data['True_Range'] = data[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
            data['ATR'] = data['True_Range'].rolling(window=14, min_periods=14).mean()
            data['ADR_Percent'] = (data['ATR'] / data['Close']) * 100
            
            return data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def meets_basic_criteria(self, data: pd.DataFrame, symbol: str) -> bool:
        """Check if stock meets basic screening criteria"""
        if data.empty or len(data) < 50:
            return False
            
        latest = data.iloc[-1]
        
        # Price above $1
        if latest['Close'] < self.min_price:
            return False
            
        # ADR% above 5%
        if pd.isna(latest['ADR_Percent']) or latest['ADR_Percent'] < self.min_adr_percent:
            return False
            
        # Average daily dollar volume above $3.5M
        if pd.isna(latest['Dollar_Volume_MA']) or latest['Dollar_Volume_MA'] < self.min_daily_volume:
            return False
            
        return True
    
    def find_big_move(self, data: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Step 1: Find 30%+ move up over multiple days/weeks
        Returns: (found, {start_idx, end_idx, move_percent, days})
        """
        if len(data) < 20:
            return False, {}
            
        # Look for the most recent significant move
        for lookback in range(10, min(60, len(data))):
            start_idx = len(data) - lookback - 1
            end_idx = len(data) - 1
            
            if start_idx < 0:
                continue
                
            start_price = data.iloc[start_idx]['Close']
            end_price = data.iloc[end_idx]['Close']
            
            move_percent = ((end_price - start_price) / start_price) * 100
            
            if move_percent >= self.min_big_move_percent:
                # Check if it's over multiple days (not just single day pump)
                daily_moves = []
                for i in range(start_idx + 1, end_idx + 1):
                    prev_close = data.iloc[i-1]['Close']
                    curr_close = data.iloc[i]['Close']
                    daily_move = ((curr_close - prev_close) / prev_close) * 100
                    daily_moves.append(daily_move)
                
                # Ensure it's not just a single day pump
                max_single_day = max(daily_moves) if daily_moves else 0
                if max_single_day < move_percent * 0.8:  # No single day accounts for 80%+ of move
                    return True, {
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'move_percent': move_percent,
                        'days': lookback,
                        'start_price': start_price,
                        'end_price': end_price
                    }
        
        return False, {}
    
    def check_sma_inclining(self, data: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Step 2: Check if 10/20 SMA are inclining
        """
        if len(data) < 25:
            return False, {}
            
        # Get recent SMA values
        recent_data = data.tail(10)
        
        sma_10_slope = self.calculate_slope(recent_data['SMA_10'].dropna())
        sma_20_slope = self.calculate_slope(recent_data['SMA_20'].dropna())
        
        # Both should be inclining (positive slope)
        sma_10_inclining = sma_10_slope > 0
        sma_20_inclining = sma_20_slope > 0
        
        # Check if all MAs are in order (best case)
        latest = data.iloc[-1]
        mas_in_order = (latest['SMA_10'] > latest['SMA_20'] > latest['SMA_50']) if not pd.isna(latest['SMA_50']) else (latest['SMA_10'] > latest['SMA_20'])
        
        return sma_10_inclining and sma_20_inclining, {
            'sma_10_slope': sma_10_slope,
            'sma_20_slope': sma_20_slope,
            'mas_in_order': mas_in_order,
            'sma_10_inclining': sma_10_inclining,
            'sma_20_inclining': sma_20_inclining
        }
    
    def calculate_slope(self, series: pd.Series) -> float:
        """Calculate the slope of a series using linear regression"""
        if len(series) < 2:
            return 0
        
        x = np.arange(len(series))
        y = series.values
        
        # Remove NaN values
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return 0
            
        x = x[mask]
        y = y[mask]
        
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    def check_pullback_pattern(self, data: pd.DataFrame, big_move_info: Dict) -> Tuple[bool, Dict]:
        """
        Step 3: Check for orderly pullback with higher lows and lower highs (tightening range)
        """
        if not big_move_info:
            return False, {}
            
        # Analyze period after the big move
        start_pullback = big_move_info['end_idx']
        pullback_data = data.iloc[start_pullback:]
        
        if len(pullback_data) < 5:
            return False, {}
        
        # Check for pullback to SMA area
        latest_close = data.iloc[-1]['Close']
        sma_10 = data.iloc[-1]['SMA_10']
        sma_20 = data.iloc[-1]['SMA_20']
        
        # Should be near or approaching SMA levels
        near_sma = (abs(latest_close - sma_10) / sma_10 < 0.05) or (abs(latest_close - sma_20) / sma_20 < 0.05)
        
        # Check for tightening range (higher lows, lower highs)
        lows = pullback_data['Low'].values
        highs = pullback_data['High'].values
        
        if len(lows) < 3 or len(highs) < 3:
            return False, {}
        
        # Simple trend check for lows (should be higher) and highs (should be lower)
        low_trend = self.calculate_slope(pd.Series(lows))
        high_trend = self.calculate_slope(pd.Series(highs))
        
        tightening = low_trend > 0 and high_trend < 0
        
        return near_sma or tightening, {
            'near_sma_10': abs(latest_close - sma_10) / sma_10 < 0.05 if not pd.isna(sma_10) else False,
            'near_sma_20': abs(latest_close - sma_20) / sma_20 < 0.05 if not pd.isna(sma_20) else False,
            'low_trend': low_trend,
            'high_trend': high_trend,
            'tightening': tightening,
            'pullback_days': len(pullback_data)
        }
    
    def check_volume_pattern(self, data: pd.DataFrame, big_move_info: Dict) -> Tuple[bool, Dict]:
        """
        Step 4: Check if volume is drying up during pullback
        """
        if not big_move_info:
            return False, {}
        
        # Compare recent volume to volume during big move
        start_pullback = big_move_info['end_idx']
        
        # Volume during big move
        move_data = data.iloc[big_move_info['start_idx']:big_move_info['end_idx']+1]
        avg_move_volume = move_data['Volume'].mean()
        
        # Recent volume (last 5 days)
        recent_volume = data.tail(5)['Volume'].mean()
        
        # Volume should be lower during pullback
        volume_drying = recent_volume < avg_move_volume * 0.7
        
        # Check latest volume vs average
        latest_volume = data.iloc[-1]['Volume']
        avg_volume = data['Volume'].tail(20).mean()
        
        return volume_drying, {
            'avg_move_volume': avg_move_volume,
            'recent_volume': recent_volume,
            'volume_ratio': recent_volume / avg_move_volume if avg_move_volume > 0 else 0,
            'volume_drying': volume_drying,
            'latest_vs_avg': latest_volume / avg_volume if avg_volume > 0 else 0
        }
    
    def check_breakout_setup(self, data: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Step 5: Check if ready for breakout
        Look for thin candles (doji-like) and proximity to resistance
        """
        if len(data) < 10:
            return False, {}
            
        recent_data = data.tail(5)
        
        # Check for thin candles (small body relative to range)
        thin_candles = []
        for _, candle in recent_data.iterrows():
            body_size = abs(candle['Close'] - candle['Open'])
            range_size = candle['High'] - candle['Low']
            
            if range_size > 0:
                body_ratio = body_size / range_size
                thin_candles.append(body_ratio < 0.3)  # Small body relative to range
        
        has_thin_candles = any(thin_candles)
        
        # Check if near recent highs (potential breakout level)
        recent_high = data.tail(20)['High'].max()
        current_close = data.iloc[-1]['Close']
        near_resistance = (recent_high - current_close) / recent_high < 0.03  # Within 3%
        
        return has_thin_candles or near_resistance, {
            'thin_candles_count': sum(thin_candles),
            'has_thin_candles': has_thin_candles,
            'near_resistance': near_resistance,
            'recent_high': recent_high,
            'current_close': current_close,
            'distance_to_high': (recent_high - current_close) / recent_high * 100
        }
    
    def calculate_pattern_score(self, results: Dict) -> int:
        """Calculate overall pattern score (0-100)"""
        score = 0
        
        if results['big_move']['found']:
            score += 25
        if results['sma_inclining']['found']:
            score += 20
        if results['pullback']['found']:
            score += 20
        if results['volume_pattern']['found']:
            score += 20
        if results['breakout_setup']['found']:
            score += 15
        
        return score
    
    def analyze_symbol(self, symbol: str) -> Dict:
        """
        Complete analysis of a symbol for bull flag pattern
        """
        print(f"Analyzing {symbol}...")
        
        # Get data
        data = self.get_stock_data(symbol)
        if data is None:
            return {'symbol': symbol, 'error': 'Could not fetch data'}
        
        # Check basic criteria
        if not self.meets_basic_criteria(data, symbol):
            return {'symbol': symbol, 'error': 'Does not meet basic criteria'}
        
        # Run the 5-step analysis
        big_move_found, big_move_info = self.find_big_move(data)
        sma_inclining_found, sma_info = self.check_sma_inclining(data)
        pullback_found, pullback_info = self.check_pullback_pattern(data, big_move_info)
        volume_found, volume_info = self.check_volume_pattern(data, big_move_info)
        breakout_found, breakout_info = self.check_breakout_setup(data)
        
        results = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'current_price': data.iloc[-1]['Close'],
            'big_move': {'found': big_move_found, 'details': big_move_info},
            'sma_inclining': {'found': sma_inclining_found, 'details': sma_info},
            'pullback': {'found': pullback_found, 'details': pullback_info},
            'volume_pattern': {'found': volume_found, 'details': volume_info},
            'breakout_setup': {'found': breakout_found, 'details': breakout_info},
        }
        
        # Calculate overall score
        results['score'] = self.calculate_pattern_score(results)
        
        # Determine if it's a valid setup
        results['is_valid_setup'] = results['score'] >= 60  # Need at least 60/100
        
        # Calculate potential entry and stop loss
        if results['is_valid_setup']:
            recent_high = data.tail(20)['High'].max()
            recent_low = data.tail(20)['Low'].min()
            
            results['suggested_entry'] = recent_high * 1.01  # 1% above recent high
            results['suggested_stop'] = recent_low  # Low of recent range
            results['risk_percent'] = ((results['suggested_entry'] - results['suggested_stop']) / results['suggested_entry']) * 100
        
        return results

def scan_watchlist(symbols: List[str], min_score: int = 70) -> List[Dict]:
    """
    Scan a list of symbols and return those with high bull flag scores
    """
    detector = BullFlagDetector()
    results = []
    
    for symbol in symbols:
        try:
            analysis = detector.analyze_symbol(symbol)
            if 'error' not in analysis and analysis.get('score', 0) >= min_score:
                results.append(analysis)
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
        
        # Add small delay to avoid rate limiting
        time.sleep(0.5)
    
    # Sort by score
    results.sort(key=lambda x: x.get('score', 0), reverse=True)
    return results

def print_analysis_report(analysis: Dict):
    """Print a formatted analysis report"""
    print(f"\n{'='*60}")
    print(f"BULL FLAG ANALYSIS: {analysis['symbol']}")
    print(f"{'='*60}")
    
    if 'error' in analysis:
        print(f"ERROR: {analysis['error']}")
        return
    
    print(f"Current Price: ${analysis['current_price']:.2f}")
    print(f"Overall Score: {analysis['score']}/100")
    print(f"Valid Setup: {'YES' if analysis['is_valid_setup'] else 'NO'}")
    
    print(f"\n📊 PATTERN ANALYSIS:")
    print(f"1. Big Move (30%+): {'✅' if analysis['big_move']['found'] else '❌'}")
    if analysis['big_move']['found'] and analysis['big_move']['details']:
        details = analysis['big_move']['details']
        print(f"   Move: {details['move_percent']:.1f}% over {details['days']} days")
    
    print(f"2. SMA Inclining: {'✅' if analysis['sma_inclining']['found'] else '❌'}")
    if analysis['sma_inclining']['details']:
        sma_details = analysis['sma_inclining']['details']
        print(f"   10 SMA Slope: {sma_details['sma_10_slope']:.4f}")
        print(f"   20 SMA Slope: {sma_details['sma_20_slope']:.4f}")
    
    print(f"3. Pullback Pattern: {'✅' if analysis['pullback']['found'] else '❌'}")
    print(f"4. Volume Drying: {'✅' if analysis['volume_pattern']['found'] else '❌'}")
    print(f"5. Breakout Setup: {'✅' if analysis['breakout_setup']['found'] else '❌'}")
    
    if analysis.get('is_valid_setup'):
        print(f"\n💰 TRADING SETUP:")
        print(f"Suggested Entry: ${analysis.get('suggested_entry', 0):.2f}")
        print(f"Suggested Stop: ${analysis.get('suggested_stop', 0):.2f}")
        print(f"Risk: {analysis.get('risk_percent', 0):.1f}%")

# Example usage
if __name__ == "__main__":
    # Example watchlist - you can replace with your own symbols
    watchlist = ['APLD', 'GAU', 'NUVB', 'FOSL', 'ESPR','SNDX','LFST','ARDX','CLNN','ARCT','CLYM','RUN','UUUU','ONDS']
    
    print("🔍 Scanning for Bull Flag patterns...")
    print("This may take a few minutes...")
    
    # Scan the watchlist
    promising_stocks = scan_watchlist(watchlist, min_score=60)
    
    if promising_stocks:
        print(f"\n🎯 Found {len(promising_stocks)} potential setups:")
        
        for stock in promising_stocks:
            print_analysis_report(stock)
    else:
        print("\n❌ No bull flag patterns found in the current watchlist.")
        print("Try expanding your watchlist or lowering the minimum score.")
    
    print(f"\n✅ Scan completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")