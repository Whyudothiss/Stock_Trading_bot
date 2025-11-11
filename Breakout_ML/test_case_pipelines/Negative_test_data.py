import pandas as pd
import numpy as np
import yfinance as yf # We need this to fetch the data

# --- Configuration Parameters for the Scan ---
# IMPORTANT: Adjust these variables for your specific analysis.

# List of tickers to scan. Start with a smaller list (e.g., S&P 500 stocks or major indices).
TICKER_LIST = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'GOOGL', 'AMZN', 'META', 'SPY', 'QQQ']
# Start date for historical data retrieval
START_DATE = '2023-01-01'
# End date for historical data retrieval (using today's date)
END_DATE = pd.to_datetime('today').strftime('%Y-%m-%d')

# --- SAR Strategy Parameters ---
FLAGPOLE_DAYS = 30 # Number of days to look back for the "Flagpole" (prior momentum)
CONSOLIDATION_DAYS = 10 # Number of days defining the consolidation window for range calculation
RESOLUTION_DAYS = 10 # Number of days to look ahead to determine if a breakdown/breakout occurred


# --- Data Loading Function (Uses yfinance) ---

def load_historical_data_yfinance(tickers, start_date, end_date):
    """
    Downloads historical OHLCV data for multiple tickers using yfinance
    and reshapes it into the long format required by the analysis logic.
    """
    print(f"--- Downloading data for {len(tickers)} tickers from {start_date} to {end_date} ---")

    # Download data for multiple tickers
    # auto_adjust=True handles stock splits/dividends for clean prices
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        interval='1d',
        auto_adjust=True,
        progress=True,
        threads=True
    )

    # Handle the MultiIndex column structure returned by yf.download
    if data.columns.nlevels > 1:
        # We need to drop the 'Adj Close' column if it's there, as 'Close' is already adjusted
        data = data.stack(level=1).rename_axis(['Date', 'Ticker'])
        # Rename columns to match our script's expected names
        data = data.rename(columns={'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'})
        data.reset_index(inplace=True)

        # Ensure we have the necessary columns and drop any symbols that failed to load
        required_cols = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
        data = data[required_cols].dropna()
    else:
        # Handle the case of only one ticker being downloaded (yfinance returns a simple DataFrame)
        data.reset_index(inplace=True)
        data['Ticker'] = tickers[0]
        data.rename(columns={'Adj Close': 'Close'}, inplace=True)
        data = data[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]

    print(f"--- Download Complete. Total {len(data)} rows of data loaded. ---")
    return data.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)

# --- Core ML Labeling Logic (From previous response, ensuring all dependencies are met) ---

def calculate_technical_indicators(df):
    """Calculates necessary moving averages and True Range."""
    # Ensure numerical data types
    for col in ['Close', 'High', 'Low', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Calculate Simple Moving Averages (SMAs)
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    # Calculate Average Daily Volume (ADV)
    df['ADV_50'] = df['Volume'].rolling(window=50).mean()
    df['ADV_5'] = df['Volume'].rolling(window=5).mean()

    # Calculate True Range (TR) for volatility analysis
    df['Close_prev'] = df['Close'].shift(1)
    df['TR'] = np.maximum.reduce([
        df['High'] - df['Low'],
        np.abs(df['High'] - df['Close_prev']),
        np.abs(df['Low'] - df['Close_prev'])
    ])
    df.drop(columns=['Close_prev'], inplace=True)

    df['ATR_5'] = df['TR'].rolling(window=5).mean()
    df['ATR_10'] = df['TR'].rolling(window=10).mean()

    # Drop intermediate columns and NaN values created by rolling windows
    df.dropna(inplace=True)
    return df

def check_consolidation_criteria(data, i):
    """
    Checks if the stock on day i meets the SAR consolidation setup criteria.
    """
    # 1. Prior Momentum (Flagpole) - up 30%+ over FLAGPOLE_DAYS
    # Check if we have enough historical data for the lookback
    if i - FLAGPOLE_DAYS < 0:
        return False

    price_x_days_ago = data.iloc[i - FLAGPOLE_DAYS]['Close']
    current_price = data.iloc[i]['Close']
    momentum_pct = (current_price / price_x_days_ago) - 1

    if momentum_pct < 0.30:
        return False

    # 2. SMA Alignment (Trend Health)
    if data.iloc[i]['SMA_10'] < data.iloc[i]['SMA_20']:
        return False

    # Calculate SMA slope over the last 5 days (10 SMA must be inclining)
    # Check if we have enough historical data for the lookback
    if i - 5 < 0:
        return False

    sma_slope = (data.iloc[i]['SMA_10'] - data.iloc[i - 5]['SMA_10']) / data.iloc[i - 5]['SMA_10']
    if sma_slope < 0:
        return False

    # 3. Price near 10-SMA (Pullback Position) - within 5%
    sma_distance = np.abs(data.iloc[i]['Close'] - data.iloc[i]['SMA_10']) / data.iloc[i]['SMA_10']
    if sma_distance > 0.05:
        return False

    # 4. Volume Drying Up (ADV_5 < 60% of ADV_50)
    if data.iloc[i]['ADV_5'] > (0.60 * data.iloc[i]['ADV_50']):
        return False

    # 5. Volatility Contraction (ATR_5 < ATR_10)
    if data.iloc[i]['ATR_5'] >= data.iloc[i]['ATR_10']:
        return False

    # If all checks pass, it is a POTENTIAL SETUP
    return True

def get_resolution_label(data, i):
    """
    Checks the next RESOLUTION_DAYS to determine if the setup led to failure or success.
    Returns: 'NEGATIVE' if breakdown occurs first, or None if breakout/sideways.
    """
    # Check if we have enough forward-looking data
    if i + RESOLUTION_DAYS >= len(data):
        return 'NEUTRAL'

    # Define the consolidation range based on the past CONSOLIDATION_DAYS
    consolidation_window = data.iloc[i - CONSOLIDATION_DAYS + 1 : i + 1]
    consolidation_low = consolidation_window['Low'].min()
    consolidation_high = consolidation_window['High'].max()

    # Look ahead for resolution
    resolution_window = data.iloc[i + 1 : i + RESOLUTION_DAYS + 1]

    breakout_triggered = False
    breakdown_triggered = False

    for day in range(len(resolution_window)):
        current_day_data = resolution_window.iloc[day]

        # 1. Check for Breakout (The desirable outcome)
        if current_day_data['Close'] > consolidation_high:
            breakout_triggered = True
            break

        # 2. Check for Breakdown (The NEGATIVE label condition)
        # Failure occurs if price closes below the defined failure threshold (Consolidation Low)
        if current_day_data['Close'] < consolidation_low:
            breakdown_triggered = True
            break

    # --- Labeling Logic ---

    # If a breakdown occurred before a breakout, this is our NEGATIVE sample.
    if breakdown_triggered and not breakout_triggered:
        # P1: NEGATIVE - It looked like a setup but failed downward.
        return 'NEGATIVE'

    # If a breakout occurred, it's a POSITIVE sample (already covered by your 700 trades)
    # or an ambiguous case we don't need for the NEGATIVE list.
    return None


def find_negative_samples(all_data):
    """
    Main function to iterate through data and label P1 NEGATIVE samples.
    """
    negative_samples = []
    # Ensure enough data for the longest indicator (SMA_50) + flagpole lookback
    min_start_index = max(FLAGPOLE_DAYS, 50)

    print(f"Starting scan for negative samples across {len(all_data['Ticker'].unique())} tickers...")

    for ticker in all_data['Ticker'].unique():
        # Filter data for one ticker and calculate indicators
        df_ticker = all_data[all_data['Ticker'] == ticker].copy().reset_index(drop=True)

        # Skip tickers without enough data
        if len(df_ticker) < min_start_index + RESOLUTION_DAYS:
            # print(f"Skipping {ticker}: Not enough data.")
            continue

        df_ticker = calculate_technical_indicators(df_ticker)

        # Iterate through the history starting where all indicators are valid
        for i in range(len(df_ticker) - RESOLUTION_DAYS):
            if i < min_start_index: # Skip until we have enough lookback data
                continue

            # Step 1: Check if the current day meets the consolidation criteria
            if check_consolidation_criteria(df_ticker, i):
                # Step 2: If it's a potential setup, check how it resolved
                label = get_resolution_label(df_ticker, i)

                if label == 'NEGATIVE':
                    # Log the negative sample
                    consolidation_window = df_ticker.iloc[i - CONSOLIDATION_DAYS + 1 : i + 1]
                    negative_samples.append({
                        'Ticker': ticker,
                        'Date': df_ticker.iloc[i]['Date'],
                        'P1_Label': 'NEGATIVE',
                        'Setup_Close': df_ticker.iloc[i]['Close'],
                        'Consolidation_Low': consolidation_window['Low'].min(),
                        'Consolidation_High': consolidation_window['High'].max(),
                    })
        
        # Optional: Add progress reporting
        if len(negative_samples) % 50 == 0 and len(negative_samples) > 0:
            print(f"Found {len(negative_samples)} negative samples so far...")


    return pd.DataFrame(negative_samples)


if __name__ == '__main__':
    # 1. Load historical data using yfinance
    historical_data = load_historical_data_yfinance(
        tickers=TICKER_LIST,
        start_date=START_DATE,
        end_date=END_DATE
    )

    # 2. Run the programmatic labeling script
    negative_samples_df = find_negative_samples(historical_data)

    # 3. Review and save the generated negative samples
    print("\n--- Generated Negative Samples (P1: FAILED CONSOLIDATION) ---")
    print(f"Total P1 Negative Samples Found: {len(negative_samples_df)}")

    if not negative_samples_df.empty:
        # Display the first few rows
        print(negative_samples_df.head())
        # Save the data to a CSV file
        negative_samples_df.to_csv('p1_negative_consolidation_samples.csv', index=False)
        print("\nSaved negative samples to 'p1_negative_consolidation_samples.csv'")
    else:
        print("No negative samples found based on the current list of tickers and date range.")

    print("\nNext Steps:")
    print("1. Combine these negative samples with your 700 positive trade entries.")
    print("2. Begin feature engineering (creating SMA slopes, volume ratios, etc.) on the combined dataset.")
    print("3. Train Pipeline 1 (Classification: POSITIVE vs. NEGATIVE).")