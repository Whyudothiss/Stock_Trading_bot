from datetime import datetime
import pandas as pd
import yfinance as yf

def get_trading_days(ticker: str, start_date: pd.Timestamp, num_days: int) -> pd.DataFrame:
    """ Gets N trading days of data after start_date """
    end_date = start_date + pd.Timedelta(days=num_days * 2)  # Buffer to account for weekends
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    return data.head(num_days)