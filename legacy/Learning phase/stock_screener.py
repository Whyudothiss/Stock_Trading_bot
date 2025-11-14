import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

class StockScreener:
    def __init__(self):
        self.results = []
        
    def get_sp500_tickers(self):
        """Get S&P 500 tickers from Wikipedia"""
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url)
            df = tables[0]
            return df['Symbol'].tolist()
        except:
            # Fallback list of major tickers if Wikipedia fails
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK-B', 'UNH', 'JNJ']
    
    def get_nasdaq_100_tickers(self):
        """Get NASDAQ 100 tickers"""
        try:
            url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
            tables = pd.read_html(url)
            # Usually the first table contains the companies
            df = tables[4]  # The companies table is typically index 4
            return df['Ticker'].tolist()
        except:
            return []
    
    def get_russell_1000_sample(self):
        """Sample of Russell 1000 stocks (would need paid API for complete list)"""
        # This is a representative sample - for complete Russell 1000 you'd need a paid data source
        additional_tickers = [
            'AMD', 'CRM', 'ADBE', 'PYPL', 'INTC', 'CSCO', 'PEP', 'KO', 'WMT', 'HD', 
            'BAC', 'JPM', 'V', 'MA', 'NKE', 'MCD', 'IBM', 'GE', 'F', 'T', 'VZ',
            'UBER', 'LYFT', 'SNAP', 'TWTR', 'SQ', 'SHOP', 'ZM', 'PTON', 'RBLX',
            'COIN', 'HOOD', 'SOFI', 'PLTR', 'SNOW', 'CRWD', 'ZS', 'OKTA', 'DDOG',
            'MDB', 'SPLK', 'WDAY', 'VEEV', 'TEAM', 'ATLASSIAN', 'NOW', 'PANW',
            'FTNT', 'CHKP', 'CYBR', 'RPD', 'TENB', 'PING', 'SAIL', 'ALRM'
        ]
        return additional_tickers
    
    def calculate_adr_percentage(self, data):
        """Calculate Average Daily Range as percentage"""
        try:
            daily_range = (data['High'] - data['Low']) / data['Close'] * 100
            return daily_range.mean()
        except:
            return 0
    
    def calculate_moving_averages(self, data):
        """Calculate moving averages"""
        try:
            data['MA_10'] = data['Close'].rolling(window=10).mean()
            data['MA_20'] = data['Close'].rolling(window=20).mean()
            data['MA_50'] = data['Close'].rolling(window=50).mean()
            data['MA_200'] = data['Close'].rolling(window=200).mean()
            return data
        except:
            return None
    
    def check_ma_alignment(self, data):
        """Check if 10MA > 20MA > 50MA > 200MA"""
        try:
            latest = data.iloc[-1]
            return (latest['MA_10'] > latest['MA_20'] > 
                   latest['MA_50'] > latest['MA_200'])
        except:
            return False
    
    def calculate_dollar_volume(self, data):
        """Calculate average daily dollar volume"""
        try:
            dollar_volume = (data['Close'] * data['Volume']).tail(20).mean()  # 20-day average
            return dollar_volume
        except:
            return 0
    
    def screen_stock(self, ticker):
        """Screen individual stock against all criteria"""
        try:
            print(f"Screening {ticker}...")
            
            # Download 1 year of data
            stock = yf.Ticker(ticker)
            data = stock.history(period='1y')
            
            if len(data) < 200:  # Need enough data for 200MA
                return None
            
            # Basic filters
            current_price = data['Close'].iloc[-1]
            if current_price <= 1:
                return None
            
            # Calculate ADR%
            adr_pct = self.calculate_adr_percentage(data)
            if adr_pct <= 5:
                return None
            
            # Calculate dollar volume
            dollar_volume = self.calculate_dollar_volume(data)
            if dollar_volume <= 300000:
                return None
            
            # Calculate moving averages
            data = self.calculate_moving_averages(data)
            if data is None:
                return None
            
            # Check MA alignment
            if not self.check_ma_alignment(data):
                return None
            
            # If we get here, stock passes all filters
            latest = data.iloc[-1]
            result = {
                'Ticker': ticker,
                'Price': round(current_price, 2),
                'ADR%': round(adr_pct, 2),
                'Dollar_Volume': int(dollar_volume),
                'MA_10': round(latest['MA_10'], 2),
                'MA_20': round(latest['MA_20'], 2),
                'MA_50': round(latest['MA_50'], 2),
                'MA_200': round(latest['MA_200'], 2),
                'Date': latest.name.strftime('%Y-%m-%d')
            }
            
            print(f"✓ {ticker} PASSED all filters!")
            return result
            
        except Exception as e:
            print(f"Error screening {ticker}: {str(e)}")
            return None
    
    def run_screen(self, max_stocks=100):
        """Run the complete screening process"""
        print("=== US Stock Screener Started ===")
        print(f"Screening criteria:")
        print(f"- Price > $1")
        print(f"- ADR% > 5%")
        print(f"- Dollar volume > $300,000")
        print(f"- Moving average alignment: 10MA > 20MA > 50MA > 200MA")
        print(f"- Screening top {max_stocks} stocks...")
        print()
        
        # Get stock list
        sp500_tickers = self.get_sp500_tickers()
        additional_tickers = self.get_additional_large_caps()
        all_tickers = list(set(sp500_tickers + additional_tickers))[:max_stocks]
        
        print(f"Total stocks to screen: {len(all_tickers)}")
        print()
        
        self.results = []
        total_stocks = len(all_tickers)
        
        for i, ticker in enumerate(all_tickers, 1):
            try:
                result = self.screen_stock(ticker)
                if result:
                    self.results.append(result)
                
                # Progress update
                if i % 10 == 0:
                    print(f"Progress: {i}/{total_stocks} ({i/total_stocks*100:.1f}%)")
                    print(f"Stocks passed so far: {len(self.results)}")
                    print()
                
                # Rate limiting - be nice to Yahoo Finance
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error with {ticker}: {str(e)}")
                continue
        
        print("=== SCREENING COMPLETE ===")
        print(f"Total stocks screened: {total_stocks}")
        print(f"Stocks that passed all filters: {len(self.results)}")
        print()
        
        return self.results
    
    def display_results(self):
        """Display results in a formatted table"""
        if not self.results:
            print("No stocks passed the screening criteria.")
            return
        
        df = pd.DataFrame(self.results)
        df = df.sort_values('Dollar_Volume', ascending=False)
        
        print("=== SCREENING RESULTS ===")
        print(df.to_string(index=False))
        print()
        
        # Summary statistics
        print("=== SUMMARY STATISTICS ===")
        print(f"Average Price: ${df['Price'].mean():.2f}")
        print(f"Average ADR%: {df['ADR%'].mean():.2f}%")
        print(f"Average Dollar Volume: ${df['Dollar_Volume'].mean():,.0f}")
        print()
        
        return df
    
    def save_results(self, filename='stock_screen_results.csv'):
        """Save results to CSV file"""
        if self.results:
            df = pd.DataFrame(self.results)
            df = df.sort_values('Dollar_Volume', ascending=False)
            df.to_csv(filename, index=False)
            print(f"Results saved to {filename}")

def main():
    screener = StockScreener()
    
    # Run the screen (limit to 50 stocks for demo - increase for full scan)
    results = screener.run_screen(max_stocks=50)
    
    # Display and save results
    df = screener.display_results()
    screener.save_results()
    
    print("\n=== NEXT STEPS ===")
    print("1. Review the filtered stocks above")
    print("2. Check the saved CSV file for detailed data")
    print("3. Research individual stocks on Webull/Moomoo")
    print("4. Consider additional fundamental analysis")

if __name__ == "__main__":
    main()