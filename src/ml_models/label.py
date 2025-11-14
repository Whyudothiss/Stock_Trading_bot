from datetime import datetime
import pandas as pd
from .data import get_trading_days
from .config import FIXED_RISK, R_MULTIPLE_TARGET, FORWARD_DAYS

def calculate_label(ticker, entry_date, entry_price, stop_loss):
    """Return 1 if trade hits a 3R profit before stop loss, else 0."""
    try: 
        # Calculate position sizing
        risk_per_share = entry_price - stop_loss
        if risk_per_share <= 0:
            return 0
        
        shares = FIXED_RISK/risk_per_share
        target_gain_dollars = FIXED_RISK * R_MULTIPLE_TARGET
        target_gain_per_share = target_gain_dollars / shares
        target_3R_price = entry_price + target_gain_per_share

        future_data = get_trading_days(ticker, entry_date, FORWARD_DAYS)

        if len(future_data) < 10:
            return 0
        
        future_data["SMA10"] = future_data["Close"].rolling(window=10).mean()

        # Check for each day
        for i in range(len(future_data)):
            row = future_data.iloc[i]

            # Hit target
            if row['High'] >= target_3R_price:
                return 1
            
            # Stopped out
            if row["Low"] <= stop_loss:
                return 0
            
            # Closed below 10SMA
            if row["Close"] < row["SMA10"]:
                return 0
            
        # Did not hit 3R but also not stopped out
        return 0
    except Exception as e:
        print(f"Error labeling {ticker} on {entry_date}: {e}")
        return 0