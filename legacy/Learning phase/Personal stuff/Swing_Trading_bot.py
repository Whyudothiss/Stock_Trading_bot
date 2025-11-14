import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import ta
import joblib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class SwingTradingAI:
    """
    AI ssytem that learns and implement my Swing Trading System
    """

    def __init__(self) -> None:
        self.model = None
        self.scalar = StandardScaler()
        self.features_columns = []


    def add_breakout_trade(self, symbol: str, entry_date: str, entry_price: float, exit_date: str, exit_price: float, exit_reason: str = "stop_loss", notes: str =" "):
        """
        Add a breakout trade that I have done following the strategy

        Args:
            symbol: Stock symbol
            entry_date: Entry date (YYYY-MM-DD)
            entry_price: Entry price
            exit_date: Exit date
            exit_price: Exit price
            exit_reason: 'stop_loss', '5x_risk_hit', 'below_10sma', 'breakeven'
            notes: Additional notes
        """

        profit_loss_pct = ((exit_price - entry_price) / entry_price) * 100
        profitable = exit_price > entry_price
        days_held = (pd.to_datetime(exit_date) - pd.to_datetime(entry_date))

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
        self.trades_date.append(trade)
        status = "BIG WIN" if trade['big_winner'] else ("WIN" if profitable else "LOSS")
        print(f"Added trade: {symbol} {status} ({profit_loss_pct:.1f}%) - {exit_reason}")

    def calculate_trade_features(self, df: pd.DataFrame, entry_date: str) -> Dict:
