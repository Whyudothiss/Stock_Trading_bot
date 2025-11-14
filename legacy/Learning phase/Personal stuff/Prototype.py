import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import ta
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class SARTradingAI:
    """
    AI system that learns and implements Morgan's SAR Swing Trading System
    Focuses specifically on The Breakout setup with the 5-step process
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.trades_data = []
        self.performance_metrics = {}
        self.strategy_name = "SAR Breakout System"
        
    def add_breakout_trade(self, symbol: str, entry_date: str, entry_price: float, 
                          exit_date: str, exit_price: float, exit_reason: str = "stop_loss",
                          notes: str = ""):
        """
        Add a breakout trade following SAR system rules
        """
        profit_loss_pct = ((exit_price - entry_price) / entry_price) * 100
        profitable = exit_price > entry_price
        days_held = (pd.to_datetime(exit_date) - pd.to_datetime(entry_date)).days
        
        estimated_risk_pct = 3.0
        risk_reward_ratio = profit_loss_pct / estimated_risk_pct if estimated_risk_pct > 0 else 0
        
        trade = {
            'symbol': symbol,
            'entry_date': entry_date,
            'entry_price': entry_price,
            'exit_date': exit_date,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'notes': notes,
            'profit_loss_pct': profit_loss_pct,
            'profitable': profitable,
            'days_held': days_held,
            'risk_reward_ratio': risk_reward_ratio,
            'big_winner': profit_loss_pct > 15,
            'strategy_type': 'breakout'
        }
        
        self.trades_data.append(trade)
        status = "🟢 BIG WIN" if trade['big_winner'] else ("🟢 WIN" if profitable else "🔴 LOSS")
        print(f"✅ Added SAR trade: {symbol} {status} ({profit_loss_pct:.1f}%) - {exit_reason}")

    def calculate_sar_features(self, df: pd.DataFrame, entry_date: str) -> Dict:
        """Calculate features specific to SAR breakout system"""
        try:
            # Get data up to entry date
            entry_date_ts = pd.to_datetime(entry_date)
            
            # Find the closest previous trading date
            previous_dates = df.index[df.index <= entry_date_ts]
            if len(previous_dates) == 0:
                return {}
                
            entry_date_ts = previous_dates[-1]
            entry_idx = df.index.get_loc(entry_date_ts)
            historical_data = df.iloc[:entry_idx+1].copy()
        
            if len(historical_data) < 60:
                return {}
            
            # Current price and moving averages
            current_price = float(historical_data['Close'].iloc[-1])
            
            # Calculate SMAs with proper handling
            sma_10 = float(historical_data['Close'].rolling(10).mean().iloc[-1]) if len(historical_data) >= 10 else current_price
            sma_20 = float(historical_data['Close'].rolling(20).mean().iloc[-1]) if len(historical_data) >= 20 else current_price
            sma_50 = float(historical_data['Close'].rolling(50).mean().iloc[-1]) if len(historical_data) >= 50 else current_price
            sma_200 = float(historical_data['Close'].rolling(200).mean().iloc[-1]) if len(historical_data) >= 200 else sma_50
            
            # SAR System Key Features
            big_move_30d, max_gain_30d = self.check_big_move_sar(historical_data, 30)
            big_move_60d, max_gain_60d = self.check_big_move_sar(historical_data, 60)
            big_move_90d, max_gain_90d = self.check_big_move_sar(historical_data, 90)
            
            sma10_inclining = self.check_sma_inclining(historical_data, 10)
            sma20_inclining = self.check_sma_inclining(historical_data, 20)
            sma50_inclining = self.check_sma_inclining(historical_data, 50)
            
            pullback_quality, pullback_data = self.check_orderly_pullback(historical_data)
            volume_drying = self.check_volume_drying_up(historical_data)
            breakout_setup, resistance_level = self.check_breakout_setup(historical_data)
            
            # Stock Selection Rules
            adr_pct = self.calculate_adr_percent(historical_data)
            dollar_volume = self.calculate_avg_dollar_volume(historical_data)
            
            # Moving Average Order
            ma_order_score = self.calculate_ma_order_score(sma_10, sma_20, sma_50, sma_200)
            
            # Price relative to SMAs
            price_vs_sma10_pct = ((current_price - sma_10) / sma_10) * 100 if sma_10 > 0 else 0
            price_vs_sma20_pct = ((current_price - sma_20) / sma_20) * 100 if sma_20 > 0 else 0
            price_vs_sma50_pct = ((current_price - sma_50) / sma_50) * 100 if sma_50 > 0 else 0
            
            # Recent performance
            return_1m = self.calculate_period_return(historical_data, 20)
            return_3m = self.calculate_period_return(historical_data, 60)
            return_6m = self.calculate_period_return(historical_data, 120)
            
            # Market condition
            market_condition = self.get_market_condition(entry_date)
            
            # Consolidation metrics
            consolidation_days = pullback_data.get('consolidation_days', 0)
            consolidation_tightness = pullback_data.get('tightness_score', 0)
            
            # Volume patterns
            recent_avg_volume = float(historical_data['Volume'].tail(10).mean())
            long_avg_volume = float(historical_data['Volume'].tail(30).mean())
            volume_trend = recent_avg_volume / long_avg_volume if long_avg_volume > 0 else 1
            
            # Technical indicators
            try:
                rsi_values = ta.momentum.RSIIndicator(historical_data['Close']).rsi()
                rsi = float(rsi_values.iloc[-1]) if not pd.isna(rsi_values.iloc[-1]) else 50
            except:
                rsi = 50
            
            # Volatility
            try:
                volatility = historical_data['Close'].pct_change().rolling(20).std()
                volatility_20d = float(volatility.iloc[-1] * 100) if not pd.isna(volatility.iloc[-1]) else 2.0
            except:
                volatility_20d = 2.0
            
            features = {
                'big_move_30d': int(big_move_30d),
                'big_move_60d': int(big_move_60d), 
                'big_move_90d': int(big_move_90d),
                'max_gain_30d': max_gain_30d,
                'max_gain_60d': max_gain_60d,
                'max_gain_90d': max_gain_90d,
                
                'sma10_inclining': int(sma10_inclining),
                'sma20_inclining': int(sma20_inclining),
                'sma50_inclining': int(sma50_inclining),
                
                'pullback_quality': int(pullback_quality),
                'consolidation_days': consolidation_days,
                'consolidation_tightness': consolidation_tightness,
                
                'volume_drying': int(volume_drying),
                'volume_trend': volume_trend,
                
                'breakout_setup': int(breakout_setup),
                'resistance_level': resistance_level,
                
                'price_above_1': int(current_price > 1),
                'adr_above_5': int(adr_pct > 5),
                'volume_above_3_5m': int(dollar_volume > 3500000),
                'current_price': current_price,
                'adr_pct': adr_pct,
                'dollar_volume': dollar_volume,
                
                'ma_order_score': ma_order_score,
                'price_vs_sma10_pct': price_vs_sma10_pct,
                'price_vs_sma20_pct': price_vs_sma20_pct,
                'price_vs_sma50_pct': price_vs_sma50_pct,
                
                'return_1m': return_1m,
                'return_3m': return_3m,
                'return_6m': return_6m,
                
                'market_condition_bullish': int(market_condition),
                
                'rsi': rsi,
                'volatility_20d': volatility_20d,
                'day_of_week': pd.to_datetime(entry_date).dayofweek,
                'month': pd.to_datetime(entry_date).month,
                
                'sar_score': sum([big_move_30d or big_move_60d, 
                                 sma10_inclining and sma20_inclining,
                                 pullback_quality, volume_drying, breakout_setup])
            }
            
            return features
            
        except Exception as e:
            print(f"Error calculating SAR features: {e}")
            return {}

    def check_big_move_sar(self, df: pd.DataFrame, lookback_days: int) -> Tuple[bool, float]:
        """Check for 30%+ move over multiple days"""
        try:
            if len(df) < lookback_days:
                return False, 0.0
            
            # Convert to numpy array to avoid Series comparison issues
            closes = df['Close'].tail(lookback_days).values
            max_gain = 0.0
            
            # Look for multi-day moves
            for i in range(len(closes)-5):
                for j in range(i+5, len(closes)):
                    gain = (closes[j] - closes[i]) / closes[i] * 100
                    if gain > max_gain:
                        max_gain = gain
            
            return max_gain >= 30, max_gain
        except Exception as e:
            print(f"Error in check_big_move_sar: {e}")
            return False, 0.0

    def check_sma_inclining(self, df: pd.DataFrame, period: int) -> bool:
        """Check if SMA is inclining over recent days"""
        try:
            if len(df) < period + 5:
                return False
                
            sma = df['Close'].rolling(period).mean()
            current_sma = float(sma.iloc[-1])
            past_sma = float(sma.iloc[-6])
            
            return current_sma > past_sma
        except Exception as e:
            print(f"Error in check_sma_inclining: {e}")
            return False

    def check_orderly_pullback(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """Check for orderly pullback with higher lows and lower highs"""
        try:
            recent_df = df.tail(20)
            if len(recent_df) < 10:
                return False, {}
            
            # Convert to numpy arrays to avoid Series comparison issues
            highs = recent_df['High'].values
            lows = recent_df['Low'].values
            closes = recent_df['Close'].values
            
            # Calculate SMAs
            sma_10 = recent_df['Close'].rolling(10, min_periods=5).mean().values
            sma_20 = recent_df['Close'].rolling(20, min_periods=10).mean().values
            
            # Check if price is near 10/20 SMA
            current_price = closes[-1]
            current_sma10 = sma_10[-1] if not np.isnan(sma_10[-1]) else current_price
            current_sma20 = sma_20[-1] if not np.isnan(sma_20[-1]) else current_price
            
            near_sma10 = abs(current_price - current_sma10) / current_sma10 < 0.05
            near_sma20 = abs(current_price - current_sma20) / current_sma20 < 0.05
            
            # Find swing highs and lows
            swing_highs = []
            swing_lows = []
            
            for i in range(2, len(highs)-2):
                if (highs[i] >= highs[i-1] and highs[i] >= highs[i-2] and 
                    highs[i] >= highs[i+1] and highs[i] >= highs[i+2]):
                    swing_highs.append((i, highs[i]))
                if (lows[i] <= lows[i-1] and lows[i] <= lows[i-2] and 
                    lows[i] <= lows[i+1] and lows[i] <= lows[i+2]):
                    swing_lows.append((i, lows[i]))
            
            # Check for tightening range
            tightening = False
            tightness_score = 0.0
            
            if len(swing_highs) >= 2 and len(swing_lows) >= 2:
                swing_highs.sort()
                swing_lows.sort()
                
                recent_highs = [h[1] for h in swing_highs[-2:]]
                recent_lows = [l[1] for l in swing_lows[-2:]]
                
                lower_highs = recent_highs[-1] <= recent_highs[0]
                higher_lows = recent_lows[-1] >= recent_lows[0]
                
                if lower_highs and higher_lows:
                    tightening = True
                    range_now = recent_highs[-1] - recent_lows[-1]
                    range_before = recent_highs[0] - recent_lows[0]
                    if range_before > 0:
                        tightness_score = (range_before - range_now) / range_before * 100
            
            pullback_data = {
                'near_sma10': near_sma10,
                'near_sma20': near_sma20,
                'tightening': tightening,
                'tightness_score': max(0.0, tightness_score),
                'consolidation_days': len(recent_df),
            }
            
            orderly = (near_sma10 or near_sma20) and tightening
            
            return orderly, pullback_data
            
        except Exception as e:
            print(f"Error checking orderly pullback: {e}")
            return False, {}

    def check_volume_drying_up(self, df: pd.DataFrame) -> bool:
        """Check if volume is decreasing during consolidation"""
        try:
            if len(df) < 30:
                return False
            
            recent_volume = float(df['Volume'].tail(10).mean())
            avg_volume = float(df['Volume'].tail(30).mean())
            
            return recent_volume < avg_volume * 0.75
        except Exception as e:
            print(f"Error in check_volume_drying_up: {e}")
            return False

    def check_breakout_setup(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """Check if stock is setting up for breakout"""
        try:
            recent_df = df.tail(20)
            current_price = float(df['Close'].iloc[-1])
            
            # Find recent resistance level
            recent_highs = recent_df['High'].rolling(5).max()
            resistance_level = float(recent_highs.max())
            
            near_resistance = abs(current_price - resistance_level) / resistance_level < 0.03
            below_resistance = current_price < resistance_level
            
            setup_ready = near_resistance and below_resistance
            
            return setup_ready, resistance_level
            
        except Exception as e:
            print(f"Error in check_breakout_setup: {e}")
            return False, 0.0

    def calculate_adr_percent(self, df: pd.DataFrame, period: int = 20) -> float:
        """Calculate Average Daily Range percentage"""
        try:
            daily_range = ((df['High'] - df['Low']) / df['Close']) * 100
            result = daily_range.rolling(period).mean().iloc[-1]
            return float(result) if not pd.isna(result) else 0.0
        except Exception as e:
            print(f"Error in calculate_adr_percent: {e}")
            return 0.0

    def calculate_avg_dollar_volume(self, df: pd.DataFrame, period: int = 20) -> float:
        """Calculate average daily dollar volume"""
        try:
            dollar_volume = df['Volume'] * df['Close']
            result = dollar_volume.rolling(period).mean().iloc[-1]
            return float(result) if not pd.isna(result) else 0.0
        except Exception as e:
            print(f"Error in calculate_avg_dollar_volume: {e}")
            return 0.0

    def calculate_ma_order_score(self, sma10: float, sma20: float, sma50: float, sma200: float) -> float:
        """Calculate moving average order score"""
        try:
            score = 0
            if not pd.isna(sma10) and not pd.isna(sma20) and sma10 > sma20:
                score += 1
            if not pd.isna(sma20) and not pd.isna(sma50) and sma20 > sma50:
                score += 1  
            if not pd.isna(sma50) and not pd.isna(sma200) and sma50 > sma200:
                score += 1
            return score / 3.0
        except Exception as e:
            print(f"Error in calculate_ma_order_score: {e}")
            return 0.5

    def calculate_period_return(self, df: pd.DataFrame, days: int) -> float:
        """Calculate return over specified period"""
        try:
            if len(df) < days + 1:
                return 0.0
            current_price = float(df['Close'].iloc[-1])
            past_price = float(df['Close'].iloc[-(days+1)])
            return ((current_price - past_price) / past_price) * 100
        except Exception as e:
            print(f"Error in calculate_period_return: {e}")
            return 0.0

    def get_market_condition(self, date: str) -> bool:
        """Check if market condition is bullish"""
        try:
            start_date = (pd.to_datetime(date) - timedelta(days=50)).strftime('%Y-%m-%d')
            end_date = (pd.to_datetime(date) + timedelta(days=5)).strftime('%Y-%m-%d')
            
            market_data = yf.download('^IXIC', start=start_date, end=end_date, progress=False)
            if market_data.empty:
                return True
            
            trade_date = pd.to_datetime(date)
            market_data_filtered = market_data[market_data.index <= trade_date]
            
            if len(market_data_filtered) < 20:
                return True
            
            sma10 = market_data_filtered['Close'].rolling(10).mean().iloc[-1]
            sma20 = market_data_filtered['Close'].rolling(20).mean().iloc[-1]
            
            if pd.isna(sma10) or pd.isna(sma20):
                return True
            
            return sma10 > sma20
            
        except:
            return True

    def prepare_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare SAR-specific training data"""
        print(f"📊 Preparing SAR training data from {len(self.trades_data)} trades...")
        
        all_features = []
        all_targets = []
        successful_trades = 0
        
        for i, trade in enumerate(self.trades_data):
            try:
                symbol = trade['symbol']
                entry_date = trade['entry_date']
                
                print(f"Processing trade {i+1}/{len(self.trades_data)}: {symbol} on {entry_date}")
                
                # Get extended data for proper SMA calculations
                start_date = (pd.to_datetime(entry_date) - timedelta(days=250)).strftime('%Y-%m-%d')
                end_date = (pd.to_datetime(entry_date) + timedelta(days=10)).strftime('%Y-%m-%d')
                
                stock_data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                
                if stock_data.empty:
                    print(f"  ⚠️  No data available for {symbol}")
                    continue
                
                # Calculate SAR-specific features
                features = self.calculate_sar_features(stock_data, entry_date)
                
                if not features:
                    print(f"  ⚠️  Could not calculate features for {symbol}")
                    continue
                
                all_features.append(features)
                all_targets.append(1 if trade['big_winner'] else 0)
                successful_trades += 1
                print(f"  ✅ Successfully processed {symbol}")
                
            except Exception as e:
                print(f"  ❌ Error processing {trade['symbol']}: {e}")
                continue
        
        if successful_trades == 0:
            raise ValueError(f"No valid features extracted from {len(self.trades_data)} SAR trades.")
        
        features_df = pd.DataFrame(all_features)
        features_df = features_df.fillna(0)
        targets_series = pd.Series(all_targets)
        
        self.feature_columns = features_df.columns.tolist()
        
        print(f"✅ SAR training data ready: {len(features_df)} samples, {len(self.feature_columns)} features")
        return features_df, targets_series

    def train_sar_model(self):
        """Train AI model specifically for SAR breakout system"""
        if len(self.trades_data) < 5:
            print("❌ Need at least 5 SAR trades to train")
            return
        
        print("🤖 Training SAR Breakout System AI...")
        
        try:
            X, y = self.prepare_training_data()
            
            # Handle class imbalance if needed
            big_winner_pct = y.mean() * 100
            print(f"📈 Big Winner Rate: {big_winner_pct:.1f}%")
            
            X_scaled = self.scaler.fit_transform(X)
            
            if len(X) > 4:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.25, random_state=42
                )
            else:
                X_train, X_test, y_train, y_test = X_scaled, X_scaled, y, y
            
            # Use Random Forest
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                class_weight='balanced'
            )
            
            self.model.fit(X_train, y_train)
            
            # Evaluate
            train_score = self.model.score(X_train, y_train) 
            test_score = self.model.score(X_test, y_test) if len(X_test) > 0 else train_score
            
            print(f"🎯 SAR Model Training Complete!")
            print(f"Training Accuracy: {train_score:.1%}")
            print(f"Test Accuracy: {test_score:.1%}")
            
            # Calculate SAR-specific metrics
            self.performance_metrics = {
                'accuracy': test_score,
                'big_winner_rate': big_winner_pct,
                'total_trades': len(self.trades_data),
                'avg_winner_pct': np.mean([t['profit_loss_pct'] for t in self.trades_data if t['profitable']]),
                'avg_loser_pct': np.mean([t['profit_loss_pct'] for t in self.trades_data if not t['profitable']]),
                'avg_days_held': np.mean([t['days_held'] for t in self.trades_data]),
                'win_rate': sum([1 for t in self.trades_data if t['profitable']]) / len(self.trades_data) * 100
            }
            
        except Exception as e:
            print(f"❌ Error training model: {e}")

    def analyze_sar_factors(self):
        """Show most important factors in SAR system"""
        if self.model is None:
            print("❌ Model not trained yet!")
            return
        
        feature_importance = pd.DataFrame({
            'factor': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n📊 TOP SAR BREAKOUT FACTORS:")
        print("=" * 50)
        
        for i, row in feature_importance.head(10).iterrows():
            print(f"{row['factor']:25} {row['importance']:.3f}")
        
        return feature_importance

    def evaluate_sar_setup(self, symbol: str, date: str = None) -> Dict:
        """Evaluate a stock for SAR breakout setup"""
        if self.model is None:
            return {'error': 'Model not trained yet!'}
        
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            # Get data
            start_date = (pd.to_datetime(date) - timedelta(days=250)).strftime('%Y-%m-%d')
            end_date = (pd.to_datetime(date) + timedelta(days=5)).strftime('%Y-%m-%d')
            
            stock_data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if stock_data.empty:
                return {'error': 'Could not fetch data'}
            
            # Calculate SAR features
            features = self.calculate_sar_features(stock_data, date)
            if not features:
                return {'error': 'Could not calculate features'}
            
            # Prepare for prediction
            feature_vector = pd.DataFrame([features])
            for col in self.feature_columns:
                if col not in feature_vector.columns:
                    feature_vector[col] = 0
            
            feature_vector = feature_vector[self.feature_columns]
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            # Make prediction
            big_winner_prob = self.model.predict_proba(feature_vector_scaled)[0][1]
            prediction = self.model.predict(feature_vector_scaled)[0]
            
            # SAR System Check
            sar_score = features.get('sar_score', 0)
            
            result = {
                'symbol': symbol,
                'date': date,
                'current_price': features.get('current_price', 0),
                'big_winner_probability': big_winner_prob,
                'prediction': 'STRONG SETUP' if prediction == 1 else 'AVOID/WAIT',
                'sar_score': f"{sar_score}/5",
                'confidence': 'HIGH' if big_winner_prob > 0.7 else 'MEDIUM' if big_winner_prob > 0.5 else 'LOW'
            }
            
            return result
            
        except Exception as e:
            return {'error': f'Analysis failed: {str(e)}'}

    def save_sar_model(self, filename: str = "sar_trading_ai.pkl"):
        """Save the SAR model"""
        if self.model is None:
            print("❌ No model to save!")
            return
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler, 
            'feature_columns': self.feature_columns,
            'trades_data': self.trades_data,
            'performance_metrics': self.performance_metrics,
            'strategy_name': self.strategy_name
        }
        
        joblib.dump(model_data, filename)
        print(f"✅ SAR model saved to {filename}")

# Demo function for SAR system
def demo_sar_system():
    """Demonstrate SAR Trading System AI"""
    
    print("🚀 SAR BREAKOUT TRADING SYSTEM AI")
    print("=" * 50)
    print("Based on Morgan's proven swing trading strategy")
    
    ai = SARTradingAI()
    
    # Add sample SAR trades
    print("\n📝 Adding sample SAR breakout trades...")
    
    # Big winners (15%+ following SAR system)
    ai.add_breakout_trade('AAPL', '2024-01-15', 150.00, '2024-01-28', 175.00, '5x_risk_partial', 'Perfect breakout setup')
    ai.add_breakout_trade('MSFT', '2024-01-20', 380.00, '2024-02-05', 440.00, 'below_10sma', 'Strong momentum')
    ai.add_breakout_trade('NVDA', '2024-02-01', 600.00, '2024-02-15', 720.00, '5x_risk_partial', 'AI hype breakout')
    
    # Winners (5-15%)
    ai.add_breakout_trade('META', '2024-01-10', 350.00, '2024-01-22', 380.00, 'below_10sma', 'Good setup')
    ai.add_breakout_trade('GOOGL', '2024-01-25', 140.00, '2024-02-08', 155.00, '5x_risk_partial', 'Earnings run')
    
    # Losers (stopped out)
    ai.add_breakout_trade('TSLA', '2024-02-05', 200.00, '2024-02-06', 195.00, 'stop_loss', 'Failed breakout')

    ai.add_breakout_trade('ZEPP', '2024-08-04', 16.20, '2024-08-25', 38.00, 'below_10sma', 'Perfect breakout')
    ai.add_breakout_trade('RGTI', '2025-09-12', 17.3,)
    ai.add_breakout_trade('TEM', '2025-09-11', 82.001)

    
    print(f"\n🤖 Training SAR AI on {len(ai.trades_data)} breakout trades...")
    
    # Train the model
    ai.train_sar_model()
    
    # Show what the AI learned about SAR factors
    ai.analyze_sar_factors()
    
    # Test current setups
    print(f"\n🎯 EVALUATING CURRENT SETUPS:")
    print("=" * 40)
    
    test_symbols = ['NUVB', 'ONDS', 'AMPX', 'DMAC', 'WULF']
    for symbol in test_symbols:
        analysis = ai.evaluate_sar_setup(symbol)
        if 'error' not in analysis:
            print(f"\n📊 {analysis['symbol']} (${analysis['current_price']:.2f})")
            print(f"SAR Score: {analysis['sar_score']} | {analysis['prediction']}")
            print(f"Big Winner Probability: {analysis['big_winner_probability']:.1%}")
        else:
            print(f"❌ {symbol}: {analysis['error']}")
    
    # Save model
    ai.save_sar_model()
    
    print(f"\n✅ SAR AI system ready!")

if __name__ == "__main__":
    demo_sar_system()

