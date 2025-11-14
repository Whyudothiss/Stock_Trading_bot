import argparse
import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, r2_score, mean_absolute_error


def fetch_history(ticker: str, period: str = '2y') -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    if data is None or len(data) == 0:
        raise ValueError(f"No data for {ticker}")
    return data


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['ret_1d'] = out['Close'].pct_change()
    out['ret_5d'] = out['Close'].pct_change(5)
    out['ret_10d'] = out['Close'].pct_change(10)

    out['sma_10'] = out['Close'].rolling(10).mean()
    out['sma_20'] = out['Close'].rolling(20).mean()
    out['sma_50'] = out['Close'].rolling(50).mean()

    out['ema_12'] = out['Close'].ewm(span=12).mean()
    out['ema_26'] = out['Close'].ewm(span=26).mean()

    out['macd'] = out['ema_12'] - out['ema_26']
    out['macd_signal'] = out['macd'].ewm(span=9).mean()
    out['macd_hist'] = out['macd'] - out['macd_signal']

    out['rsi'] = _rsi(out['Close'], window=14)

    out['bb_mid'] = out['Close'].rolling(20).mean()
    bb_std = out['Close'].rolling(20).std()
    out['bb_up'] = out['bb_mid'] + 2 * bb_std
    out['bb_lo'] = out['bb_mid'] - 2 * bb_std

    out['vol_sma_20'] = out['Volume'].rolling(20).mean()
    out['vol_ratio'] = out['Volume'] / out['vol_sma_20']

    out['price_vs_sma20'] = out['Close'] / out['sma_20']
    out['price_vs_sma50'] = out['Close'] / out['sma_50']
    out['sma10_vs_sma20'] = out['sma_10'] / out['sma_20']
    out['sma20_vs_sma50'] = out['sma_20'] / out['sma_50']

    out['adr_pct'] = ((out['High'] - out['Low']) / out['Close']) * 100
    out['adr20'] = out['adr_pct'].rolling(20).mean()

    return out


def _rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=window, min_periods=window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=window, min_periods=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def make_classification_labels(df: pd.DataFrame, horizon_days: int = 10, threshold_pct: float = 5.0) -> pd.Series:
    future_price = df['Close'].shift(-horizon_days)
    future_return_pct = (future_price - df['Close']) / df['Close'] * 100
    label = (future_return_pct >= threshold_pct).astype(int)
    return label


def make_regression_target(df: pd.DataFrame, horizon_days: int = 10) -> pd.Series:
    future_price = df['Close'].shift(-horizon_days)
    future_return_pct = (future_price - df['Close']) / df['Close'] * 100
    return future_return_pct


def build_dataset(ticker: str, period: str = '5y', horizon_days: int = 10, threshold_pct: float = 5.0, mode: str = 'classify') -> tuple[pd.DataFrame, pd.Series]:
    df = fetch_history(ticker, period)
    df = add_features(df)

    feature_cols = [
        'ret_1d','ret_5d','ret_10d',
        'sma_10','sma_20','sma_50',
        'ema_12','ema_26','macd','macd_signal','macd_hist',
        'rsi','bb_mid','bb_up','bb_lo',
        'vol_sma_20','vol_ratio',
        'price_vs_sma20','price_vs_sma50','sma10_vs_sma20','sma20_vs_sma50',
        'adr20'
    ]

    if mode == 'classify':
        y = make_classification_labels(df, horizon_days=horizon_days, threshold_pct=threshold_pct)
    else:
        y = make_regression_target(df, horizon_days=horizon_days)

    X = df[feature_cols].copy()
    valid = (~X.isna().any(axis=1)) & (~y.isna())
    X = X[valid]
    y = y[valid]

    return X, y


def select_top_k_features(model, X: pd.DataFrame, top_k: int) -> list:
    importances = getattr(model, 'feature_importances_', None)
    if importances is None:
        # Fallback: variance ranking
        variances = X.var().sort_values(ascending=False)
        return list(variances.head(top_k).index)
    order = np.argsort(importances)[::-1]
    chosen = [X.columns[i] for i in order[:top_k]]
    return chosen


def train_evaluate_classification(X: pd.DataFrame, y: pd.Series, top_k: int, random_state: int = 42):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, shuffle=False
    )

    clf_all = RandomForestClassifier(
        n_estimators=400,
        random_state=random_state,
        class_weight='balanced_subsample'
    )
    clf_all.fit(X_train, y_train)

    # Feature selection on original (unscaled) columns by importance
    top_features = select_top_k_features(clf_all, pd.DataFrame(X, columns=X.columns, index=y.index), top_k)

    # Retrain using only top-K
    X_top = pd.DataFrame(X[top_features])
    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        scaler.fit_transform(X_top), y, test_size=0.25, shuffle=False
    )

    clf = RandomForestClassifier(
        n_estimators=400,
        random_state=random_state,
        class_weight='balanced_subsample'
    )
    clf.fit(X_train2, y_train2)

    y_pred = clf.predict(X_test2)
    y_prob = clf.predict_proba(X_test2)[:, 1]

    report = classification_report(y_test2, y_pred, digits=3)
    try:
        auc = roc_auc_score(y_test2, y_prob)
    except Exception:
        auc = float('nan')

    metrics = {
        'test_samples': int(len(y_test2)),
        'pos_rate_test': float(y_test2.mean()) if len(y_test2) else 0.0,
        'roc_auc': float(auc) if not pd.isna(auc) else None,
        'selected_features': top_features
    }

    return clf, scaler, metrics, report, top_features


def train_evaluate_regression(X: pd.DataFrame, y: pd.Series, top_k: int, random_state: int = 42):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, shuffle=False
    )

    reg_all = RandomForestRegressor(
        n_estimators=400,
        random_state=random_state
    )
    reg_all.fit(X_train, y_train)

    top_features = select_top_k_features(reg_all, pd.DataFrame(X, columns=X.columns, index=y.index), top_k)

    X_top = pd.DataFrame(X[top_features])
    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        scaler.fit_transform(X_top), y, test_size=0.25, shuffle=False
    )

    reg = RandomForestRegressor(
        n_estimators=400,
        random_state=random_state
    )
    reg.fit(X_train2, y_train2)

    y_pred = reg.predict(X_test2)

    metrics = {
        'test_samples': int(len(y_test2)),
        'r2': float(r2_score(y_test2, y_pred)) if len(y_test2) else None,
        'mae': float(mean_absolute_error(y_test2, y_pred)) if len(y_test2) else None,
        'selected_features': top_features
    }

    report = None
    return reg, scaler, metrics, report, top_features


def predict_tickers_classification(model, scaler, feature_columns: list[str], tickers: list[str]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        try:
            df = fetch_history(t, period='1y')
            df_feat = add_features(df)
            X = df_feat[feature_columns].tail(1)
            if X.isna().any(axis=1).iloc[0]:
                raise ValueError('NaN features')
            Xs = scaler.transform(X)
            prob = model.predict_proba(Xs)[0, 1]
            rows.append({'Ticker': t, 'Prob_Growth': prob, 'Date': df.index[-1].strftime('%Y-%m-%d')})
        except Exception as e:
            rows.append({'Ticker': t, 'Prob_Growth': None, 'Error': str(e)})
    return pd.DataFrame(rows)


def predict_tickers_regression(model, scaler, feature_columns: list[str], tickers: list[str], horizon_days: int) -> pd.DataFrame:
    rows = []
    for t in tickers:
        try:
            df = fetch_history(t, period='1y')
            df_feat = add_features(df)
            X = df_feat[feature_columns].tail(1)
            if X.isna().any(axis=1).iloc[0]:
                raise ValueError('NaN features')
            Xs = scaler.transform(X)
            pred_return_pct = float(model.predict(Xs)[0])
            last_close = float(df['Close'].iloc[-1])
            est_price = last_close * (1 + pred_return_pct / 100.0)
            rows.append({
                'Ticker': t,
                'Pred_Return_%': pred_return_pct,
                'Last_Close': last_close,
                f'Est_Close_+{horizon_days}d': est_price,
                'Date': df.index[-1].strftime('%Y-%m-%d')
            })
        except Exception as e:
            rows.append({'Ticker': t, 'Pred_Return_%': None, 'Error': str(e)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description='Stock predictor with top-K feature selection')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Single ticker to train on and evaluate')
    parser.add_argument('--period', type=str, default='5y', help='History period for training, e.g., 2y, 5y, max')
    parser.add_argument('--horizon', type=int, default=10, help='Prediction horizon in trading days')
    parser.add_argument('--threshold', type=float, default=5.0, help='Growth threshold in % over horizon (classification)')
    parser.add_argument('--mode', type=str, default='classify', choices=['classify','regress'], help='Task: classify (>=threshold) or regress (predict % return)')
    parser.add_argument('--top_k', type=int, default=8, help='Select top-K indicators by importance')
    parser.add_argument('--score', type=str, default='', help='Comma-separated tickers to score after training')

    args = parser.parse_args()

    print(f"Training on {args.ticker} ({args.period}) → horizon={args.horizon}d mode={args.mode} top_k={args.top_k}")

    X, y = build_dataset(args.ticker, period=args.period, horizon_days=args.horizon, threshold_pct=args.threshold, mode=args.mode)
    if len(X) < 200:
        raise ValueError('Not enough samples after feature prep. Try longer period or different ticker.')

    if args.mode == 'classify':
        model, scaler, metrics, report, top_features = train_evaluate_classification(X, y, top_k=args.top_k)
        print('Test metrics:')
        print(metrics)
        if report:
            print(report)
        feature_columns = top_features
        if args.score:
            tickers = [t.strip().upper() for t in args.score.split(',') if t.strip()]
            scores = predict_tickers_classification(model, scaler, feature_columns, tickers)
            print('\nScores:')
            print(scores.to_string(index=False))
    else:
        model, scaler, metrics, report, top_features = train_evaluate_regression(X, y, top_k=args.top_k)
        print('Test metrics:')
        print(metrics)
        feature_columns = top_features
        if args.score:
            tickers = [t.strip().upper() for t in args.score.split(',') if t.strip()]
            scores = predict_tickers_regression(model, scaler, feature_columns, tickers, horizon_days=args.horizon)
            print('\nScores:')
            print(scores.to_string(index=False))


if __name__ == '__main__':
    main()
