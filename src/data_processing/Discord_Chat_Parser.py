import json
import re
from datetime import datetime

# --- CONFIGURATION ---
# *** IMPORTANT: Replace this with the actual path to your raw Discord export JSON file ***
RAW_DISCORD_FILEPATH = 'Json files/discord_chat_history.json' 
OUTPUT_FILEPATH = 'Json files/parsed_trade_entries_initial.json'

# Regex pattern to capture the required trade details from a single message.
# It looks for: $TICKER, entry price (@ $X), stop loss price (Stop loss @ $Y), and risk percentage (Risking Z%).
TRADE_PATTERN = re.compile(
    # Entry: Look for a keyword, then $TICKER (G1), followed by an optional '@', optional '$', and the Entry Price (G2)
    r'(?:Adding|Buying|Entry).*?\$([A-Z]+).*?(?:\@ )?\$?(\d{1,3}(?:,\d{3})*\.?\d*).*?' 
    # Stop Loss: Look for a Stop Loss keyword, then optional 'at', optional '$', and the SL Price (G3)
    r'(?:[Ss]top [Ll]oss|[Ss][Ll] at|[Ss][Ll]).*?\$?(\d{1,3}(?:,\d{3})*\.?\d*).*?'  
    # Risk: Look for a Risk keyword, then the Risk Percentage (G4) followed by an optional '%'
    r'(?:[Rr]isking|[Rr]isk).*?(\d+\.?\d*)%?',                                     
    re.DOTALL | re.IGNORECASE
)

def parse_discord_history(filepath):
    """
    Reads a raw Discord JSON export and extracts structured trade entries 
    based on the 'Adding $TICKER' message pattern.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Raw Discord file not found at '{filepath}'. Please check the path.")
        return []
    except json.JSONDecodeError:
        print("Error: Could not decode JSON. Ensure the file is valid.")
        return []

    # The actual chat history is typically nested under the 'messages' key
    messages = raw_data.get('messages', [])
    parsed_trades = []
    trade_counter = 0

    print(f"Found {len(messages)} messages in the history. Starting trade extraction...")

    for message in messages:
        content = message.get('content', '')
        timestamp = message.get('timestamp', '')

        # Use the regex to find the trade entry pattern
        match = TRADE_PATTERN.search(content)

        if match:
            trade_counter += 1
            
            ticker = match.group(1).upper()
            entry_price = float(match.group(2))
            initial_stop_loss_price = float(match.group(3))
            portfolio_risk_percent = float(match.group(4))
            
            # Calculate the price distance (risk per share)
            price_risk_per_share = round(abs(entry_price - initial_stop_loss_price), 2)
            
            # Convert Discord timestamp to clean ISO format
            try:
                # The provided Discord timestamps end with +08:00, use fromisoformat
                dt_object = datetime.fromisoformat(timestamp)
                clean_timestamp = dt_object.isoformat()
            except ValueError:
                clean_timestamp = timestamp # Use original if parsing fails

            trade_entry = {
                "ticker": ticker,
                "entry_date": clean_timestamp,
                "entry_price": entry_price,
                "initial_stop_loss_price": initial_stop_loss_price,
                "portfolio_risk_percent": portfolio_risk_percent,
                "price_risk_per_share": price_risk_per_share,
                "exit_date": None,
                "exit_price": None,
                "final_pnl_usd": None,
                "outcome_category": "Open_Unknown"
            }
            parsed_trades.append(trade_entry)
            print(f"-> Extracted Trade {trade_counter}: ${ticker} @ ${entry_price}")
    
    print(f"\nCompleted extraction. Total trades found: {trade_counter}")
    return parsed_trades

if __name__ == '__main__':
    extracted_data = parse_discord_history(RAW_DISCORD_FILEPATH)
    
    if extracted_data:
        # Save the structured data
        with open(OUTPUT_FILEPATH, 'w') as f:
            json.dump(extracted_data, f, indent=4)
        print(f"Successfully saved {len(extracted_data)} trades to {OUTPUT_FILEPATH}")