import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def calculate_adr_percentage (ticker, period='3mo'):
    """
    Calculates the Average Daily Range (ADR) as a percentage of a given stock

    Args: 
        ticker: Stock symbol
        period: Time period for calculation ('1mo', '3mo', '6mo', '1yr')

    Returns: 
        Dictionary with ADR% 
    """
    try:
        print(f"\n---Analyzing {ticker} ---")

        stock = yf.ticker(ticker)
        data = stock.history(period=period)

        if len(data) == 0:
            return {"error": f"No data found for {ticker}"}
        
        daily_range = (data['High'] - data['Low']) / data['Close'] * 100

        adr_percentage = daily_range.mean()

        current_price = data['Close'].iloc[-1]
        avg_volume = data['Volume'].mean()
        dollar_volume = (data['close'] * data['Volume']).tail(20).mean() # 20-avg

        price_change_pct = (current_price - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100

        result = {
            'Ticker': ticker,
            'Current_price': round(current_price, 2),
            'ADR%': round(adr_percentage,2),
            'Avg_Volume': int(avg_volume),
            'Dollar_Volume': int(dollar_volume),
            'Period_Return': round(price_change_pct, 2),
            'Data_Period': period,
            'Days_Analyzed': len(data),
            'Date': data.index[-1].strftime('%Y-%m-%d')
        }

        return result
    except Exception as e:
        return {"error": f"Error analyzing {ticker}: {str(e)}"}
    

 