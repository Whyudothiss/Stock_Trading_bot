import json
from datetime import datetime, timedelta
import os
import pandas as pd
import yfinance as yf 
import time
import argparse
from pathlib import Path
import numpy as np
import logging
import sys

# Logging setup
def setup_logger(log_file=None, level=logging.INFO):
    logger = logging.getLogger('trade_analyzer')
    logger.setLevel(level)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        fmt = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setLevel(level)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger


# Load trade entries from the repository's JSON file (if present).
def load_trade_entries(filename='parsed_trade_entries_initial.json'):
    """Attempt to load trade entries from the data/raw folder.

    Returns a list of trade dicts or an empty list if the file is missing or invalid.
    """
    # Go up to project root and look in data/raw
    base_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(base_dir, '..', '..'))
    json_path = os.path.join(project_root, 'data', 'raw', filename)
    try:
        with open(json_path, 'r') as fh:
            data = json.load(fh)
            if isinstance(data, list):
                return data
            # If the file contains a dict with a key like 'parsed_trade', try to extract it.
            if isinstance(data, dict):
                # common key names we might expect
                for key in ('parsed_trade', 'trades', 'entries', 'parsed_trade_entries'):
                    if key in data:
                        return data[key]
            print(f"Warning: unexpected format in {json_path}; expected list or dict containing trades.")
    except FileNotFoundError:
        print(f"Warning: trade entries file not found at {json_path}. TRADE_ENTRIES will be empty.")
    except Exception as e:
        print(f"Warning: failed to read trade entries from {json_path}: {e}")
    return []


# Populate TRADE_ENTRIES at import time so the script can run standalone.
# Use the repository's `parsed_trade_entries_initial.json` file (contains the parsed trade details).
TRADE_ENTRIES = load_trade_entries('parsed_trade_entries_initial.json')

def fetch_historical_data(ticker, start_date_str, end_date_override=None):
    """
    Fetches OHLC data and calculates the 10-period Simple Moving Average (SMA).
    
    Args:
        ticker (str): The stock ticker symbol.
        start_date_str (str or datetime): The date to start fetching data (e.g., 'YYYY-MM-DD').
        
    Returns:
        list: A list of daily price bars as dictionaries.
    """
    try:
        # Handle both string and datetime inputs
        if isinstance(start_date_str, datetime):
            start_date_obj = start_date_str
            start_date_str = start_date_obj.strftime('%Y-%m-%d')
        elif isinstance(start_date_str, str):
            start_date_obj = datetime.strptime(start_date_str, '%Y-%m-%d')
        else:
            start_date_obj = pd.to_datetime(start_date_str).to_pydatetime()
            start_date_str = start_date_obj.strftime('%Y-%m-%d')
            
        # use stdlib timedelta to avoid dtype issues when subtracting from datetime
        lookback_date = (start_date_obj - timedelta(days=30)).strftime('%Y-%m-%d')

        # First try to load cached data (prefetched) if available
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        cache_dir = os.environ.get('TRADE_ANALYZER_CACHE_DIR', os.path.join(project_root, 'data', 'cache'))
        cache_path = os.path.join(cache_dir, f"{ticker}.parquet")
        if os.path.exists(cache_path):
            try:
                data = pd.read_parquet(cache_path)
            except Exception:
                # if reading cache fails, fall back to live download
                data = None
        else:
            data = None

        # If cache exists, check if it covers the required date range
        if data is not None:
            try:
                data.index = pd.to_datetime(data.index)
                min_cached = data.index.min()
                max_cached = data.index.max()
                
                # Parse the start date we need (entry date, not lookback)
                entry_dt = pd.to_datetime(start_date_str)
                
                # If cache doesn't cover the entry date (cache is too recent), ignore it and re-fetch
                if min_cached > entry_dt:
                    data = None
                # If cache covers start but not the requested end date, attempt to extend
                elif end_date_override is not None:
                    ed = pd.to_datetime(end_date_override)
                    if max_cached < ed:
                        # need to fetch additional data up to end_date_override
                        try:
                            # Use auto_adjust=True for split-adjusted prices
                            more = yf.download(ticker, start=lookback_date, end=(ed + timedelta(days=1)).strftime('%Y-%m-%d'), progress=False, auto_adjust=True)
                            if isinstance(more, pd.DataFrame) and not more.empty:
                                more.index = pd.to_datetime(more.index)
                                # combine and dedupe
                                data = pd.concat([data, more])
                                data = data[~data.index.duplicated(keep='first')].sort_index()
                                # attempt to update cache
                                try:
                                    Path(cache_dir).mkdir(parents=True, exist_ok=True)
                                    data.to_parquet(cache_path)
                                except Exception:
                                    pass
                        except Exception:
                            pass
            except Exception:
                # If anything goes wrong reading or extending cache, we'll fall back to downloading below
                data = None

        if data is None:
            # Download data using yfinance (fetch from lookback to now)
            end_dt = datetime.now().strftime('%Y-%m-%d')
            # if user provided an end_date_override and it's earlier than today, use that
            if end_date_override is not None:
                try:
                    ed = pd.to_datetime(end_date_override)
                    end_dt = ed.strftime('%Y-%m-%d')
                except Exception:
                    end_dt = datetime.now().strftime('%Y-%m-%d')
            # Use auto_adjust=True to get split-adjusted prices
            data = yf.download(ticker, start=lookback_date, end=end_dt, progress=False, auto_adjust=True)

        # Defensive: ensure data is a DataFrame. yfinance can sometimes return unexpected types.
        if isinstance(data, pd.Series):
            # single-row series -> convert to DataFrame
            data = data.to_frame().T
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"Unexpected data type returned for {ticker}: {type(data)}")

        if data.empty:
            print(f"Warning: No data found for {ticker} starting from {lookback_date}")
            return []

        # Calculate 10-period Simple Moving Average
        data['10sma'] = data['Close'].rolling(window=10).mean()

        # Filter to only include days from the trade entry date onwards
        # Normalize the comparison by converting the entry date to a Timestamp
        data = data[data.index >= pd.to_datetime(start_date_str)]

        # Prepare data in the format expected by the simulator
        price_bars = []
        for index, row in data.iterrows():
            # Coerce possible array/series values to scalars safely
            def _first_scalar(v):
                if isinstance(v, (pd.Series, np.ndarray, list, tuple)):
                    try:
                        return v.iloc[0] if isinstance(v, pd.Series) else v[0]
                    except Exception:
                        return None
                return v

            open_v = _first_scalar(row.get('Open'))
            high_v = _first_scalar(row.get('High'))
            low_v = _first_scalar(row.get('Low'))
            close_v = _first_scalar(row.get('Close'))
            sma_v = _first_scalar(row.get('10sma'))

            price_bars.append({
                "date": index.strftime('%Y-%m-%d'),
                "Open": round(float(open_v), 2) if open_v is not None and not pd.isna(open_v) else None,
                "High": round(float(high_v), 2) if high_v is not None and not pd.isna(high_v) else None,
                "Low": round(float(low_v), 2) if low_v is not None and not pd.isna(low_v) else None,
                "Close": round(float(close_v), 2) if close_v is not None and not pd.isna(close_v) else None,
                "10sma": round(float(sma_v), 2) if sma_v is not None and not pd.isna(sma_v) else None
            })

        return price_bars

    except Exception as e:
        import traceback
        print(f"Error fetching data for {ticker}: {e}")
        try:
            print("Data variable type:", type(data))
        except Exception:
            pass
        print(traceback.format_exc())
        return []


def prefetch_trade_data(trades, cache_dir=None, sleep_between=1.0, force=False, logger=None):
    """Prefetch historical data for all unique tickers in `trades` and cache to disk.

    Args:
        trades (list): list of trade dicts
        cache_dir (str|Path): directory to store cached parquet files
        sleep_between (float): seconds to sleep between downloads
        force (bool): if True, re-download even when cache exists
        logger (logging.Logger): optional logger for progress messages
    Returns:
        dict: ticker -> path to cached file
    """
    if cache_dir is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        cache_dir = Path(os.path.join(project_root, 'data', 'cache'))
    else:
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Determine earliest entry date per ticker
    ticker_earliest = {}
    for t in trades:
        tk = t.get('ticker')
        ent = str(t.get('entry_date', '')).split('T')[0]
        if not tk or not ent:
            continue
        try:
            d = datetime.strptime(ent, '%Y-%m-%d')
        except Exception:
            continue
        if tk not in ticker_earliest or d < ticker_earliest[tk]:
            ticker_earliest[tk] = d

    cached = {}
    total = len(ticker_earliest)
    i = 0
    for tk, earliest in ticker_earliest.items():
        i += 1
        path = cache_dir / f"{tk}.parquet"
        if path.exists() and not force:
            cached[tk] = str(path)
            if logger:
                logger.debug(f"Cache hit for {tk}")
            continue

        # Fetch data starting 30 days before earliest entry
        lookback = (earliest - timedelta(days=30)).strftime('%Y-%m-%d')
        if logger:
            logger.info(f"Prefetching {tk} ({i}/{total}) from {lookback}...")
        else:
            print(f"Prefetching {tk} ({i}/{total}) from {lookback}...")
        try:
            # Use a 3 month period for prefetch to limit download size while covering near-term entries
            # This is faster than period='max' and aligns with your request.
            # Use auto_adjust=True for split-adjusted prices
            df = yf.download(tk, period='3mo', progress=False, auto_adjust=True)
            if df is not None and not df.empty:
                # Save as parquet for fast reload
                try:
                    df.to_parquet(path)
                    cached[tk] = str(path)
                except Exception as e:
                    if logger:
                        logger.warning(f"Could not write cache for {tk}: {e}")
                    else:
                        print(f"Warning: could not write cache for {tk}: {e}")
            else:
                if logger:
                    logger.warning(f"No data for {tk} when prefetching.")
                else:
                    print(f"Warning: no data for {tk} when prefetching.")
        except Exception as e:
            if logger:
                logger.error(f"Prefetch error for {tk}: {e}")
            else:
                print(f"Prefetch error for {tk}: {e}")

    # Sleep with jitter to reduce burstiness
    to_sleep = sleep_between + (0.1 * (i % 3))
    time.sleep(to_sleep)

    # Expose cache_dir to fetch_historical_data via env var for this process
    os.environ['TRADE_ANALYZER_CACHE_DIR'] = str(cache_dir)
    return cached


def run_simulation(trades, dry_run_n=None, cache_dir=None, logger=None, partial_filename='simulation_results_partial.jsonl', prefetch=True):
    """Run simulation for the provided trades. Optionally prefetch and run only first N trades.
    Supports resume by appending to a JSON-line partial file and producing a final JSON list on completion.
    Returns the list of results.
    """
    if dry_run_n is not None:
        trades = trades[:dry_run_n]

    # Prefetch data for unique tickers in the subset
    if prefetch:
        if logger:
            logger.info(f"Prefetching historical data for {len(trades)} trades (unique tickers)...")
        else:
            print(f"Prefetching historical data for {len(trades)} trades (unique tickers)...")
        prefetch_trade_data(trades, cache_dir=cache_dir, logger=logger)

    partial_path = Path(partial_filename)
    start_index = 0
    if partial_path.exists():
        # count existing lines to resume
        with open(partial_path, 'r') as pf:
            for _ in pf:
                start_index += 1
        if logger:
            logger.info(f"Resuming simulation from trade #{start_index} (found {start_index} partial results)")
        else:
            print(f"Resuming simulation from trade #{start_index} (found {start_index} partial results)")

    # Open partial file in append mode and write one JSON object per completed trade
    with open(partial_path, 'a') as pf:
        for idx, trade in enumerate(trades[start_index:], start=start_index):
            time.sleep(0.25)  # small throttle
            try:
                res = simulate_trade_outcome(trade)
            except Exception as e:
                res = trade.copy()
                res['simulation_result'] = 'SIM_ERROR'
                res['notes'] = f'Error during simulation: {e}'
                if logger:
                    logger.error(f"Error simulating trade idx {idx}: {e}")
                else:
                    print(f"Error simulating trade idx {idx}: {e}")
            pf.write(json.dumps(res) + "\n")
            pf.flush()
            if logger:
                logger.debug(f"Simulated trade {idx+1}/{len(trades)}: {res.get('simulation_result')}")

    # Consolidate partial file into final JSON
    results = []
    with open(partial_path, 'r') as pf:
        for line in pf:
            try:
                results.append(json.loads(line))
            except Exception:
                continue

    output_filepath = 'simulation_results_final.json'
    with open(output_filepath, 'w') as f:
        json.dump(results, f, indent=4)

    if logger:
        logger.info(f"Saved {len(results)} simulation results to {output_filepath}")
    else:
        print(f"Saved {len(results)} simulation results to {output_filepath}")
    return results

# --- CORE SIMULATION LOGIC (Unchanged) ---
def simulate_trade_outcome(trade, r_multiplier=3):
    """Label trades for ML: determine whether trade ever hits r_multiplier*R before being stopped.

    Changes for ML labeling:
    - Do NOT exit immediately when 3R is first hit; continue scanning days.
    - A stop-out occurs only when day's Low < stop_loss (strictly less than).
    - Final ML label values: 'HIT_3R', 'STOPPED_BEFORE_3R', 'NO_3R', 'AMBIGUOUS_BAR', 'DATA_ERROR', 'INVALID_*'.

    The function still returns the trade dict but adds an 'ml_label' key and some helper fields.
    """
    # Ensure expected keys exist and provide safe defaults where appropriate
    ticker = trade.get('ticker')
    entry_price = trade.get('entry_price')
    stop_loss = trade.get('initial_stop_loss_price')
    risk_price = trade.get('price_risk_per_share')
    entry_date_str = str(trade.get('entry_date', '')).split('T')[0]
    trade.setdefault('notes', '')

    # Basic validation
    if not ticker or entry_price is None or not entry_date_str:
        trade['ml_label'] = 'INVALID_TRADE_DATA'
        trade['notes'] += ' - Missing ticker, entry_price, or entry_date.'
        return trade

    if risk_price in (None, 0):
        trade['ml_label'] = 'INVALID_RISK'
        trade['notes'] += ' - price_risk_per_share missing or zero.'
        return trade

    # Calculate target based on requested multiplier (e.g. 3R or 5R)
    try:
        mul = float(r_multiplier)
    except Exception:
        mul = 3.0
    target_price = round(entry_price + (mul * risk_price), 2)
    trade[f'ml_target_{int(mul)}R'] = target_price

    # Fetch historical bars for up to 3 months after entry (user request).
    # We pass an end_date_override so the fetcher will attempt to extend cached data up to that date.
    try:
        entry_dt = datetime.strptime(entry_date_str, '%Y-%m-%d')
        end_dt = (entry_dt + timedelta(days=90)).strftime('%Y-%m-%d')
    except Exception:
        end_dt = None
    historical_bars = fetch_historical_data(ticker, entry_date_str, end_date_override=end_dt)
    if not historical_bars:
        trade['ml_label'] = 'DATA_ERROR'
        trade['notes'] += ' - Could not fetch historical data.'
        return trade

    target_ever_hit = False
    # record the first date target was hit (if any)
    trade.pop('ml_target_hit_date', None)

    for bar in historical_bars:
        trade_date = bar.get('date')
        high = bar.get('High')
        low = bar.get('Low')
        close = bar.get('Close')
        sma = bar.get('10sma')

        day_target = (high is not None) and (high >= target_price)
        # strict stop: low < stop_loss
        day_stopped = (stop_loss is not None) and (low is not None) and (low < stop_loss)
        # SMA-based close
        day_sma_close = (sma is not None) and (close is not None) and (close < sma)

        # record if target was seen
        if day_target:
            # capture first hit date
            if not target_ever_hit:
                trade['ml_target_hit_date'] = trade_date
            target_ever_hit = True

        # Both on same day -> ambiguous
        if day_target and day_stopped:
            trade['ml_label'] = 'AMBIGUOUS_BAR'
            trade['ml_target_ever_hit'] = True
            # ensure target hit date present
            if 'ml_target_hit_date' not in trade:
                trade['ml_target_hit_date'] = trade_date
            trade['ml_stop_date'] = trade_date
            return trade

        # Stop-out occurs before considering SMA close (stop is terminal)
        if day_stopped:
            trade['ml_stop_date'] = trade_date
            trade['ml_stop_price'] = low if low is not None else stop_loss
            if target_ever_hit:
                trade['ml_label'] = f'HIT_{int(mul)}R'
            else:
                # stopped before target
                trade['ml_label'] = 'LOSS_1R'
            return trade

        # SMA close: treat as a closing event
        if day_sma_close:
            trade['ml_close_date'] = trade_date
            trade['ml_close_price'] = close
            if target_ever_hit:
                trade['ml_label'] = f'HIT_{int(mul)}R'
            else:
                # If close < entry => LOSS_1R, else STOPPED_BEFORE_3R
                try:
                    if close < entry_price:
                        trade['ml_label'] = 'LOSS_1R'
                    else:
                        trade['ml_label'] = 'STOPPED_BEFORE_3R'
                except Exception:
                    trade['ml_label'] = 'NO_3R'
            return trade

    # finished scanning without stop or SMA close
    if target_ever_hit:
        trade['ml_label'] = f'HIT_{int(mul)}R'
        # no stop occurred - make field explicit
        trade['ml_stop_date'] = None
    else:
        trade['ml_label'] = 'NO_3R'
        trade['ml_stop_date'] = None

    return trade

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate trade outcomes using historical data (with prefetch/caching).')
    parser.add_argument('--dry-run', type=int, default=0, help='Run only the first N trades (0 = all)')
    parser.add_argument('--ml-labels', action='store_true', help='Generate ML labels (HIT_3R / STOPPED_BEFORE_3R / NO_3R) and exit')
    parser.add_argument('--r-multiplier', type=int, default=3, help='R multiplier to use for ML labels (e.g. 3 or 5)')
    parser.add_argument('--ml-5r', action='store_true', help='Generate ML labels specifically for 5R and write ml_5R_labels.jsonl')
    parser.add_argument('--cache-dir', type=str, default=None, help='Directory to write/read cached ticker data')
    parser.add_argument('--no-prefetch', action='store_true', help='Skip prefetch step and fetch on demand')
    parser.add_argument('--force', action='store_true', help='Force re-download of cached tickers')
    parser.add_argument('--sleep', type=float, default=1.0, help='Seconds to sleep between prefetch downloads')
    parser.add_argument('--partial-file', type=str, default='simulation_results_partial.jsonl', help='Partial JSONL file used for resume')
    parser.add_argument('--log-file', type=str, default=None, help='Path to write log file (default: cache_dir/logs/trade_analyzer.log)')
    args = parser.parse_args()

    entries = TRADE_ENTRIES
    if not entries:
        print('No trades loaded into TRADE_ENTRIES. Exiting.')
        sys.exit(0)

    # Determine cache_dir
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    cache_dir = args.cache_dir or os.path.join(project_root, 'data', 'cache')
    # Setup logs
    logs_dir = os.path.join(cache_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    log_file = args.log_file or os.path.join(logs_dir, 'trade_analyzer.log')
    logger = setup_logger(log_file=log_file)

    dry_n = args.dry_run if args.dry_run > 0 else None

    # If user requested ML labels only, run labeling and exit
    if args.ml_labels or args.ml_5r:
        # Choose multiplier and output filenames
        mult = 5 if args.ml_5r else int(args.r_multiplier or 3)
        ml_out = 'ml_5R_labels.jsonl' if args.ml_5r else 'ml_labels.jsonl'
        summary_out = 'ml_5R_summary.json' if args.ml_5r else 'ml_summary.json'
        logger.info(f'Starting ML labeling mode (R={mult}) -- writing {ml_out} and {summary_out}')
        # Ensure cache_dir is set in env for fetch_historical_data
        os.environ['TRADE_ANALYZER_CACHE_DIR'] = cache_dir
        counts = {}
        written = 0
        with open(ml_out, 'w') as outf:
            iter_entries = entries[:dry_n] if dry_n else entries
            for t in iter_entries:
                try:
                    res = simulate_trade_outcome(t.copy(), r_multiplier=mult)
                except Exception as e:
                    res = t.copy()
                    res['ml_label'] = 'SIM_ERROR'
                    res['notes'] = f'Simulation error: {e}'
                lbl = res.get('ml_label', 'UNKNOWN')
                counts[lbl] = counts.get(lbl, 0) + 1
                # dynamic target key name
                target_key = f'ml_target_{int(mult)}R'
                outf.write(json.dumps({
                    'ticker': res.get('ticker'),
                    'entry_date': res.get('entry_date'),
                    'entry_price': res.get('entry_price'),
                    'initial_stop_loss_price': res.get('initial_stop_loss_price'),
                    'price_risk_per_share': res.get('price_risk_per_share'),
                    'portfolio_risk_percent': res.get('portfolio_risk_percent'),
                    'ml_label': lbl,
                    'ml_target': res.get(target_key),
                    'ml_stop_date': res.get('ml_stop_date'),
                    'ml_stop_price': res.get('ml_stop_price'),
                    'notes': res.get('notes','')
                }) + '\n')
                written += 1

        summary = {
            'total': written,
            'counts': counts
        }
        with open(summary_out, 'w') as sf:
            json.dump(summary, sf, indent=2)
        logger.info(f'Wrote {written} labels to {ml_out}; summary -> {summary_out}')
        sys.exit(0)

    # Prefetch step (unless disabled) - supports resume via existing parquet files
    if not args.no_prefetch:
        logger.info('Starting prefetch phase (will skip tickers with existing cache files)')
        prefetch_trade_data(entries, cache_dir=cache_dir, sleep_between=args.sleep, force=args.force, logger=logger)
    else:
        logger.info('Skipping prefetch as requested (--no-prefetch)')

    # Run simulation with resume support (partial JSONL)
    run_simulation(entries, dry_run_n=dry_n, cache_dir=cache_dir, logger=logger, partial_filename=args.partial_file, prefetch=False)

    logger.info('--- SIMULATION COMPLETE ---')