import yfinance as yf;
import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
from datetime import datetime, timedelta;

def get_stock_data(symbol, period="1y") :

    try: 
        stock = yf.Ticker(symbol)
        data = stock.history(period = period)
        print(f"Fetched {len(data)} days of data for {symbol}")
        print("Columns:", list(data.columns))
        print(data.head())
        return data
    except Exception as e:
        print(f"Error: {e}")
        return None

def add_simple_indicators(df):
    # simple moving averages
    # SMA is just the sum of the closing prices of a stock over a period of time
    df['SMA_20'] = df['Close'].rolling(window=20).mean() # find mean of the closing prices 
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    # calculate RSI - measures if the stock is overbought (<70) or oversold( <30)
    def calculate_rsi(prices, window=14):
        delta = prices.diff() # calculate the change in price
        # delta.where(condition, other_value) if condition pass, keep the value
        # else replace with other value
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean() 
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs  = gain/loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df['RSI'] = calculate_rsi(df['Close'])

    df['Price_Change_Pct'] = df['Close'].pct_change() # calculates the price change of closing prices 

    return df


def simple_trading_stategy(df):

    signals = []

    for i in range(len(df)):
        # simple start:
        # if RSI < 30 (oversold) && price > 20-day average: BUY
        # if RSI > 70 (oversold) or price < 20-day average : SELL
        # Otherwise just hold

        if pd.isna(df['RSI'].iloc[i]) or pd.isna(df['SMA_20'].iloc[i]):
            signals.append('HOLD')
            continue

        rsi = df['RSI'].iloc[i]
        price = df['Close'].iloc[i]
        sma_20 = df['SMA_20'].iloc[i]

        if rsi < 30 and price > sma_20:
            signals.append('BUY')
        elif rsi > 70 or price < sma_20:
            signals.append('SELL')
        else:
            signals.append('HOLD')

    df['Signal'] = signals

    return df

def backtest_simple_strategy(df, initial_capital=10000):
    capital = initial_capital
    shares = 0
    portfolio_values = []

    for i in range(len(df)):
        price = df['Close'].iloc[i]
        signal = df['Signal'].iloc[i]

        if signal == 'BUY' and capital > price:
            # buy as many as possible
            shares_to_buy = int(capital/price)
            shares += shares_to_buy
            capital -= shares_to_buy * price
            print(f"{df.index[i].date()} BUY {shares_to_buy} shares at {price:.2f}")

        elif signal == 'SELL' and shares > 0:
            # just sell all 
            capital += shares * price
            print(f"{df.index[i].date()}: SELL {shares} shares at {price:.2f}")
            shares = 0
        
        # this to calculate the porfolio amount
        portfolio_value = capital + (shares * price)
        portfolio_values.append(portfolio_value)

    df['Portfolio_Value'] = portfolio_values

    #calculate returns
    # [-1] is to access the last value in the array
    total_return = (portfolio_values[-1] - initial_capital) / initial_capital * 100
    print(f"\nStrategy Performance")
    print(f"Initial capital: ${initial_capital:,.2f}")
    print(f"Final Value: ${portfolio_values[-1]:,.2f}")
    print(f"Total Return: {total_return:.2f}%")

    #compared to just holding the stock
    buy_hold_return = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
    print(f"Buy n Hold Return: {buy_hold_return:.2f}%")

    return df

def plot_results(df, symbol):
    """Plot the stock data, indicators, and strategy performance"""
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))
    
    # Plot 1: Price and Moving Averages
    ax1.plot(df.index, df['Close'], label='Price', color='black', linewidth=2)
    ax1.plot(df.index, df['SMA_20'], label='20-day SMA', color='blue', alpha=0.7)
    ax1.plot(df.index, df['SMA_50'], label='50-day SMA', color='red', alpha=0.7)
    ax1.set_title(f'{symbol} Stock Price and Moving Averages')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: RSI
    ax2.plot(df.index, df['RSI'], label='RSI', color='purple')
    ax2.axhline(y=70, color='r', linestyle='--', label='Overbought (70)')
    ax2.axhline(y=30, color='g', linestyle='--', label='Oversold (30)')
    ax2.set_title('RSI (Relative Strength Index)')
    ax2.legend()
    ax2.grid(True)
    ax2.set_ylim(0, 100)
    
    # Plot 3: Trading Signals
    buy_signals = df[df['Signal'] == 'BUY']
    sell_signals = df[df['Signal'] == 'SELL']
    
    ax3.plot(df.index, df['Close'], label='Price', color='black', alpha=0.7)
    ax3.scatter(buy_signals.index, buy_signals['Close'], 
               color='green', marker='^', s=100, label='Buy Signal')
    ax3.scatter(sell_signals.index, sell_signals['Close'], 
               color='red', marker='v', s=100, label='Sell Signal')
    ax3.set_title('Trading Signals')
    ax3.legend()
    ax3.grid(True)
    
    # Plot 4: Portfolio Performance
    ax4.plot(df.index, df['Portfolio_Value'], label='Strategy', color='blue', linewidth=2)
    # Calculate buy & hold performance for comparison
    initial_value = df['Portfolio_Value'].iloc[0]
    buy_hold = initial_value * (df['Close'] / df['Close'].iloc[0])
    ax4.plot(df.index, buy_hold, label='Buy & Hold', color='red', alpha=0.7)
    ax4.set_title('Portfolio Performance Comparison')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    symbol = "NVDA"
    print(f"Analyzing {symbol}")

    df = get_stock_data(symbol)
    if df is None: 
        return
    
    df = add_simple_indicators(df)

    df = simple_trading_stategy(df)

    df = backtest_simple_strategy(df)

    plot_results(df, symbol)

if __name__ == "__main__":
    main()