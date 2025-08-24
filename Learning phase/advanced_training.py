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

def add_advanced_indicators(df):
    # simple moving averages
    # SMA is just the sum of the closing prices of a stock over a period of time
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean() # find mean of the closing prices 
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    # Use Exponential Moving Average that is more sensitive to recent price changes
    df['EMA_12'] = df['Close'].ewm(span=12).mean() 

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

    # Moving average convergence divergence (MACD), trend, momentum
    # exp1 - exp2 is the zero line, meaning if exp1 - exp2 > 0 --> bullish, vice versa
    # MACD goes above signal line --> bullish
    # for histogram, if positive histogram, MACD rising faster than signal --> bullish momentum increasing, else decreasing
    # if negative histogram, MACD rising faster than signal --> bearish momentum decreasing, else increasing
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2 # MACD line
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    df['MACD_Histogram_Change'] = df['MACD_Histogram'].diff()

    # Bollinger bands - volatility indicator
    # two bands based on 20 day moving average
    # if price close to upper band --> potentilly overbough, vice versa
    # squuezing of bands could signify big price move
    df['BB_middle'] = df['Close'].rolling(window=20).mean() # middle line of indicator
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_middle'] - (bb_std * 2)

    # volume indicator
    # if volume raio is 1.0 -> average, > 1.0 is above average, < 1.0 is below average
    # a price move that has greater volume is more significant compared to when the volume ratio is lower
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean() # average volume
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']

    # Price relationships
    # for price vs SMA, if > 1 it means that potential to be bullish, prices are increasing
    # for SMA ratios, if the ratios > 1 -> bullish
    # if all ratios > 1, there is a very strong uptrend

    df['Price_vs_SMA20'] = df['Close'] / df['SMA_20'] # Price relative to medium trend
    df['Price_vs_SMA50'] = df['Close'] / df['SMA_50'] # Price relative to long trend
    df['SMA10_vs_SMA20'] = df['SMA_10'] / df['SMA_20'] # Short vs medium trend
    df['SMA20_vs_SMA50'] = df['SMA_20'] / df['SMA_50'] # medium vs long trend

    df['Price_Change_Pct'] = df['Close'].pct_change() # calculates the price change of closing prices 
    df['Price_Change_5d'] = df['Close'].pct_change(5) # 5 day change

    return df

def calculate_signal_strength(row):
    ## calculate how strong the buy/sell signals are 

    buy_score = 0
    sell_score = 0

    # check for NaN values
    if pd.isna(row['RSI']) or pd.isna(row['SMA_20']) or pd.isna(row['MACD']):
        return 'HOLD', 0.5

    # Moving day averages trend
    if row['SMA10_vs_SMA20'] > 1.01: # uptrend
        buy_score += 2
    elif row['SMA10_vs_SMA20'] < 0.99:
        sell_score += 2

    if row['SMA20_vs_SMA50'] > 1.02: # strong uptrend
        buy_score += 3
    elif row['SMA20_vs_SMA50'] < 0.98:
        sell_score += 3

    # Momntum signals - using RSI
    if row['RSI'] > 65: # Overbought
        sell_score += 1
    elif row['RSI'] < 35:
        buy_score += 1

    if row['RSI'] > 75: # Very overbought
        sell_score += 2
    elif row['RSI'] < 25:
        buy_score += 2

    # MACD signals
    if row['MACD_Histogram'] > 0 and row['MACD'] > 0:
        buy_score += 2
        # Additional score if momentum is increasing by comparing the difference in histogram size
        if row['MACD_Histogram_Change'] > 0:
            buy_score += 1 # Bullish momentum increasing

    elif row['MACD_Histogram'] < 0 and row['MACD'] < 0:
        sell_score += 2
        if row['MACD_Histogram_Change'] < 0:
            sell_score += 1 # Bearisg momentum decreasing


    # Bollinger bond for volatality signals
    if row['Close'] < row['BB_Lower']: # Price below lower band (oversold)
        buy_score += 1
    elif row['Close'] > row['BB_Upper']:
        sell_score += 1

    # Volume confirmation
    if row['Volume_Ratio'] > 1.5: # High volume day
        if buy_score > sell_score: # greater buy trend
            buy_score += 1
        elif sell_score > buy_score:
            sell_score += 1

    # Decide what to do
    # total_signals = buy_score + sell_score

    if buy_score >= 4 and buy_score > sell_score + 1:
        confidence = min(0.9, buy_score / 8)
        return 'BUY', confidence
    elif sell_score >= 4 and sell_score > buy_score + 1:
        confidence = min(0.9, sell_score / 8)
        return 'SELL', confidence
    else:
        confidence = 0.5
        return 'HOLD', confidence

def smart_trading_strategy(df):
    signals = []
    confidences = []

    for i in range(len(df)):
        row = df.iloc[i]
        signal, confidence = calculate_signal_strength(row)
        signals.append(signal)
        confidences.append(confidence)

    df['Signal'] = signals
    df['Confidences'] = confidences
    return df

def backtest_the_strategy(df, initial_capital=10000):
    capital = initial_capital
    shares = 0
    portfolio_values = []
    trades = []

    for i in range(len(df)):
        price = df['Close'].iloc[i]
        signal = df['Signal'].iloc[i]
        confidence = df['Confidences'].iloc[i]
        date = df.index[i]

        if signal == 'BUY' and capital > price and confidence > 0.6:
            # buy as many as possible
            # risk_amount = capital * min(0.10, confidence * 0.15) # Max 10% of capital
            shares_to_buy = int(capital/price)
            
            if shares_to_buy > 0:
                cost = shares_to_buy * price
                shares += shares_to_buy
                capital -= cost
                trades.append({
                    'date': date.date(),
                    'action': 'BUY',
                    'shares': shares_to_buy,
                    'price': price,
                    'confidence': confidence
                })
                print(f"{date.date()}: BUY {shares_to_buy} shares at ${price:.2f} (Confidence: {confidence:.2f})")

        elif signal == 'SELL' and shares > 0 and confidence > 0.6:
            # Also sell based on confidence level
            shares_to_sell = int(shares * confidence)
            if shares_to_sell > 0:
                revenue = shares_to_sell * price
                capital += revenue
                shares -= shares_to_sell
                trades.append({
                    'date': date.date(),
                    'action': 'SELL',
                    'shares': shares_to_sell,
                    'price': price,
                    'confidence': confidence
                })
            print(f"{date.date()}: SELL {shares_to_sell} shares at ${price:.2f} (Confidence: {confidence:.2f})")
        
        # this to calculate the porfolio amount
        portfolio_value = capital + (shares * price)
        portfolio_values.append(portfolio_value)

    df['Portfolio_Value'] = portfolio_values

    #calculate returns
    # [-1] is to access the last value in the array
    final_value = portfolio_values[-1]
    total_return = (final_value - initial_capital) / initial_capital * 100

    #compared to just holding the stock
    buy_hold_return = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100

    # To calculate the max drawdown
    peak = df['Portfolio_Value'].expanding().max()
    drawdown = (df['Portfolio_Value'] - peak) / peak * 100
    max_drawdown = drawdown.min()

    print(f"\n=== STRATEGY PERFORMANCE ===")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Final Value: ${final_value:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
    print(f"Total Trades: {len(trades)}")
    print(f"Max Drawdown: {max_drawdown:.2f}%")

    return df, trades

def plot_advanced_results(df, symbol):
    """Plot comprehensive analysis"""
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(16, 20))
    
    # Plot 1: Price with Moving Averages
    ax1.plot(df.index, df['Close'], label='Price', color='black', linewidth=2)
    ax1.plot(df.index, df['SMA_10'], label='SMA 10', color='blue', alpha=0.7)
    ax1.plot(df.index, df['SMA_20'], label='SMA 20', color='green', alpha=0.7)
    ax1.plot(df.index, df['SMA_50'], label='SMA 50', color='red', alpha=0.7)
    ax1.set_title(f'{symbol} Price with Moving Averages')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: RSI
    ax2.plot(df.index, df['RSI'], label='RSI', color='purple')
    ax2.axhline(y=75, color='r', linestyle='--', alpha=0.7)
    ax2.axhline(y=65, color='orange', linestyle='--', alpha=0.7)
    ax2.axhline(y=35, color='orange', linestyle='--', alpha=0.7)
    ax2.axhline(y=25, color='g', linestyle='--', alpha=0.7)
    ax2.set_title('RSI with Multiple Levels')
    ax2.set_ylim(0, 100)
    ax2.grid(True)
    
    # Plot 3: MACD
    ax3.plot(df.index, df['MACD'], label='MACD', color='blue')
    ax3.plot(df.index, df['MACD_Signal'], label='Signal', color='red')
    ax3.bar(df.index, df['MACD_Histogram'], label='Histogram', alpha=0.3, color='green')
    ax3.set_title('MACD Analysis')
    ax3.legend()
    ax3.grid(True)
    
    # Plot 4: Bollinger Bands
    ax4.plot(df.index, df['Close'], label='Price', color='black')
    ax4.plot(df.index, df['BB_Upper'], label='Upper Band', color='red', alpha=0.7)
    ax4.plot(df.index, df['BB_middle'], label='Middle Band', color='blue', alpha=0.7)
    ax4.plot(df.index, df['BB_Lower'], label='Lower Band', color='green', alpha=0.7)
    ax4.fill_between(df.index, df['BB_Upper'], df['BB_Lower'], alpha=0.1, color='gray')
    ax4.set_title('Bollinger Bands')
    ax4.legend()
    ax4.grid(True)
    
    # Plot 5: Trading Signals with Confidence
    buy_signals = df[df['Signal'] == 'BUY']
    sell_signals = df[df['Signal'] == 'SELL']
    
    ax5.plot(df.index, df['Close'], label='Price', color='black', alpha=0.7)
    
    # Size markers by confidence
    if len(buy_signals) > 0:
        ax5.scatter(buy_signals.index, buy_signals['Close'], 
                   s=buy_signals['Confidences']*200, color='green', marker='^', 
                   label='Buy Signal', alpha=0.8)
    if len(sell_signals) > 0:
        ax5.scatter(sell_signals.index, sell_signals['Close'], 
                   s=sell_signals['Confidences']*200, color='red', marker='v', 
                   label='Sell Signal', alpha=0.8)
    
    ax5.set_title('Trading Signals (Size = Confidence)')
    ax5.legend()
    ax5.grid(True)
    
    # Plot 6: Portfolio Performance
    ax6.plot(df.index, df['Portfolio_Value'], label='Strategy', color='blue', linewidth=2)
    initial_value = df['Portfolio_Value'].iloc[0]
    buy_hold = initial_value * (df['Close'] / df['Close'].iloc[0])
    ax6.plot(df.index, buy_hold, label='Buy & Hold', color='red', alpha=0.7)
    ax6.set_title('Portfolio Performance')
    ax6.legend()
    ax6.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    symbol = "NVDA"
    print(f"Analyzing {symbol}")

    df = get_stock_data(symbol)
    if df is None: 
        return
    
    df = add_advanced_indicators(df)

    df = smart_trading_strategy(df)

    df, trades = backtest_the_strategy(df)

    plot_advanced_results(df, symbol)

if __name__ == "__main__":
    main()
