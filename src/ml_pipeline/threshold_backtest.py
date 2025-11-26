#!/usr/bin/env python3
"""
Threshold backtest for 5R predictor models.

Loads a trained model and features file, sweeps probability thresholds,
selects historical trades (from labels file) that the model would have taken,
and runs the `RealisticPortfolioSimulator` on the selected trades to compute
realized P&L. Prints and saves a summary CSV with metrics per threshold and
reports the best threshold by total P&L.

Usage:
  python3 src/ml_pipeline/threshold_backtest.py --model-file models/5R_predictor_1min_twelve.pkl \
        --features-file data/processed/features_5R_1min_twelve.jsonl

"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import shutil

from portfolio_simulation.portfolio_simulator_realistic import RealisticPortfolioSimulator


def load_labels(labels_path):
    trades = []
    with open(labels_path, 'r') as f:
        for line in f:
            if line.strip():
                trades.append(json.loads(line))
    return trades


def match_trade(feature_row, labels_df):
    """Try to match a feature row to a label row (exact timestamp preferred).
    Returns the matched label dict or None."""
    ticker = feature_row['ticker']
    feat_ts = str(feature_row['entry_date'])

    # Exact string match first
    match = labels_df[(labels_df['ticker'] == ticker) & (labels_df['entry_date'] == feat_ts)]
    if len(match) == 1:
        return match.iloc[0].to_dict()

    # Try startswith match (feature may have date-only, label full iso)
    if 'entry_date' in labels_df.columns:
        try:
            match = labels_df[(labels_df['ticker'] == ticker) & labels_df['entry_date'].str.startswith(feat_ts[:10])]
        except Exception:
            match = labels_df[(labels_df['ticker'] == ticker) & (labels_df['entry_date'] == feat_ts)]
        if len(match) >= 1:
            return match.iloc[0].to_dict()

    return None


def run_threshold_backtest(args):
    model = joblib.load(args.model_file)
    print(f"Loaded model: {args.model_file}")

    feats = pd.read_json(args.features_file, lines=True)
    print(f"Loaded features: {len(feats)} rows from {args.features_file}")

    labels = load_labels(args.labels_file)
    labels_df = pd.DataFrame(labels)
    print(f"Loaded labels: {len(labels_df)} rows from {args.labels_file}")

    # Prepare feature matrix (same logic as training script)
    exclude_cols = set(['label', 'ticker', 'entry_date', 'timeframe', 'prob'])
    feature_cols = [c for c in feats.columns if c not in exclude_cols]
    X = feats[feature_cols].fillna(0)

    # Predict probabilities
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)[:, 1]
    else:
        # Fallback to decision_function scaled to 0-1
        try:
            scores = model.decision_function(X)
            probs = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        except Exception:
            # As a last resort, use model.predict
            probs = model.predict(X)

    feats = feats.copy()
    feats['prob'] = probs

    thresholds = np.arange(0.0, 1.0001, args.threshold_step)
    results = []

    tmp_dir = Path('data/tmp/threshold_backtest')
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for thr in thresholds:
        selected = feats[feats['prob'] >= thr]
        # Match to label trades
        matched_trades = []
        matched_keys = set()
        for _, row in selected.iterrows():
            m = match_trade(row, labels_df)
            if m is None:
                continue
            key = (m.get('ticker'), m.get('entry_date'))
            if key in matched_keys:
                continue
            matched_keys.add(key)
            matched_trades.append(m)

        n_trades = len(matched_trades)
        if n_trades < args.min_trades:
            results.append({'threshold': thr, 'n_trades': n_trades, 'total_pnl': 0.0, 'ending_capital': args.starting_capital})
            continue

        # Write matched trades to a temp JSONL
        tmp_file = tmp_dir / f'trades_thr_{int(thr*1000)}.jsonl'
        with open(tmp_file, 'w') as f:
            for t in matched_trades:
                f.write(json.dumps(t) + '\n')

        # Run simulator
        sim = RealisticPortfolioSimulator(
            starting_capital=args.starting_capital,
            risk_per_trade_pct=args.risk_pct,
            max_position_pct=args.max_position_pct,
            max_portfolio_for_scaling=args.portfolio_cap,
            slippage_pct=args.slippage,
            use_trade_risk=args.use_trade_risk
        )

        sim.run_simulation(str(tmp_file))
        try:
            ending = sim.get_portfolio_value()
        except AttributeError:
            # try alternative property
            ending = getattr(sim, 'portfolio_value', args.starting_capital)
        total_pnl = ending - args.starting_capital

        results.append({'threshold': thr, 'n_trades': n_trades, 'total_pnl': total_pnl, 'ending_capital': ending})

    # Save results CSV
    out_df = pd.DataFrame(results)
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"Saved threshold sweep results to: {out_csv}")

    # Report best threshold by total_pnl
    best = out_df.loc[out_df['total_pnl'].idxmax()]
    print("\nBest threshold summary:")
    print(best.to_string())


def build_parser():
    p = argparse.ArgumentParser(description='Threshold backtest for trained 5R model')
    p.add_argument('--model-file', required=True, help='Path to trained model (joblib .pkl)')
    p.add_argument('--features-file', required=True, help='Features JSONL used for prediction')
    p.add_argument('--labels-file', default='data/processed/ml_5R_labels_filtered.jsonl', help='Historical trades JSONL (labels)')
    p.add_argument('--threshold-step', type=float, default=0.01, help='Step for threshold sweep')
    p.add_argument('--min-trades', type=int, default=10, help='Minimum trades required to evaluate a threshold')
    p.add_argument('--starting-capital', type=float, default=10000.0)
    p.add_argument('--risk-pct', type=float, default=1.0)
    p.add_argument('--max-position-pct', type=float, default=40.0)
    p.add_argument('--portfolio-cap', type=float, default=None)
    p.add_argument('--slippage', type=float, default=0.1)
    p.add_argument('--use-trade-risk', dest='use_trade_risk', action='store_true')
    p.add_argument('--no-use-trade-risk', dest='use_trade_risk', action='store_false')
    p.set_defaults(use_trade_risk=True)
    p.add_argument('--output-csv', default='data/processed/threshold_sweep_results.csv')
    return p


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    run_threshold_backtest(args)
