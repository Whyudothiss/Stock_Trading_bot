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
        print(f"\n--- Analyzing {ticker} ---")
        
        # Get stock data
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        
        if len(data) == 0:
            return {"error": f"No data found for {ticker}"}
        
        daily_ranges = (data['High'] - data['Low']) / data['Close'] * 100
        
        adr_percentage = daily_ranges.mean()
        
        current_price = data['Close'].iloc[-1]
        avg_volume = data['Volume'].mean()
        dollar_volume = (data['Close'] * data['Volume']).tail(20).mean()  # 20-day avg
        
        price_change_pct = ((current_price - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
        
        result = {
            'Ticker': ticker,
            'Current_Price': round(current_price, 2),
            'ADR%': round(adr_percentage, 2),
            'Avg_Volume': int(avg_volume),
            'Dollar_Volume': int(dollar_volume),
            'Period_Return%': round(price_change_pct, 2),
            'Data_Period': period,
            'Days_Analyzed': len(data),
            'Date': data.index[-1].strftime('%Y-%m-%d')
        }
        
        return result
        
    except Exception as e:
        return {"error": f"Error analyzing {ticker}: {str(e)}"}
    
def display_results(results):
    """Display results in a nice format"""
    if 'error' in results:
        print(f"❌ {results['error']}")
        return
    
    print(f"✅ Ticker: {results['Ticker']}")
    print(f"📈 Current Price: ${results['Current_Price']}")
    print(f"📊 ADR%: {results['ADR%']}%")
    print(f"📦 Avg Daily Volume: {results['Avg_Volume']:,}")
    print(f"💰 Avg Dollar Volume: ${results['Dollar_Volume']:,}")
    print(f"📅 Period Return: {results['Period_Return%']}%")
    print(f"🗓️  Analysis Period: {results['Data_Period']} ({results['Days_Analyzed']} trading days)")
    print(f"📆 Last Updated: {results['Date']}")
    
    # ADR% interpretation
    adr = results['ADR%']
    if adr > 10:
        print(f"🔥 High volatility stock (ADR% > 10%)")
    elif adr > 5:
        print(f"⚡ Good volatility for day trading (ADR% > 5%)")
    elif adr > 3:
        print(f"📊 Moderate volatility (ADR% 3-5%)")
    else:
        print(f"😴 Low volatility stock (ADR% < 3%)")

def analyze_multiple_stocks(tickers, period='3mo'):
    """Analyze multiple stocks and return comparison table"""
    results = []
    
    print(f"Analyzing {len(tickers)} stocks for ADR%...")
    print("=" * 50)
    
    for ticker in tickers:
        result = calculate_adr_percentage(ticker.upper().strip(), period)
        if 'error' not in result:
            results.append(result)
            display_results(result)
        else:
            print(f"❌ {result['error']}")
    
    if results:
        print("\n" + "=" * 80)
        print("COMPARISON TABLE")
        print("=" * 80)
        
        df = pd.DataFrame(results)
        df = df.sort_values('ADR%', ascending=False)
        
        # Display formatted table
        print(df[['Ticker', 'Current_Price', 'ADR%', 'Dollar_Volume', 'Period_Return%']].to_string(index=False))
        
        print(f"\n📊 SUMMARY:")
        print(f"Highest ADR%: {df['ADR%'].max()}% ({df.loc[df['ADR%'].idxmax(), 'Ticker']})")
        print(f"Average ADR%: {df['ADR%'].mean():.2f}%")
        print(f"Stocks with ADR% > 5%: {len(df[df['ADR%'] > 5])}")
        
        return df
    else:
        print("No valid stocks analyzed.")
        return None

def main():
    """Main interactive function"""
    print("🎯 ADR% Calculator - Stock Volatility Analyzer")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Analyze single stock")
        print("2. Analyze multiple stocks")
        print("3. Quit")
        
        choice = input("\nChoose option (1/2/3): ").strip()
        
        if choice == '1':
            ticker = input("Enter stock ticker (e.g., AAPL): ").strip().upper()
            period = input("Enter period (1mo/3mo/6mo/1y) [default: 3mo]: ").strip() or '3mo'
            
            result = calculate_adr_percentage(ticker, period)
            display_results(result)
            
        elif choice == '2':
            print("Enter stock tickers separated by commas (e.g., AAPL,MSFT,GOOGL):")
            tickers_input = input("Tickers: ").strip()
            tickers = [t.strip() for t in tickers_input.split(',') if t.strip()]
            
            if not tickers:
                print("No valid tickers entered.")
                continue
            
            period = input("Enter period (1mo/3mo/6mo/1y) [default: 3mo]: ").strip() or '3mo'
            
            df = analyze_multiple_stocks(tickers, period)
            
            
        elif choice == '3':
            print("Thanks for using ADR% Calculator!")
            break
        
        else:
            print("Invalid option. Please choose 1, 2, or 3.")

if __name__ == "__main__":
    main()