"""
Train 5R Predictor Model (Pipeline 2)

Trains models on 1-min or 5-min breakout features to predict which trades will hit 5R target.
Handles severe class imbalance (15.7% win rate) using balanced weights.

Usage:
    python3 src/ml_pipeline/train_5R_model.py --timeframe 1min
    python3 src/ml_pipeline/train_5R_model.py --timeframe 5min
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    precision_recall_curve,
    roc_curve
)
import joblib

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available - will skip XGBoost model")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  LightGBM not available - will skip LightGBM model")


def load_features(filepath):
    """Load features from JSONL file"""
    features = []
    with open(filepath, 'r') as f:
        for line in f:
            features.append(json.loads(line))
    
    df = pd.DataFrame(features)
    return df


def prepare_data(df):
    """Prepare features and labels for training"""
    # Separate features and labels
    feature_cols = [col for col in df.columns if col not in ['label', 'ticker', 'entry_date', 'timeframe']]
    
    X = df[feature_cols]
    y = df['label']
    
    # Check for any NaN or infinite values
    if X.isnull().any().any():
        print(f"⚠️  Warning: {X.isnull().sum().sum()} NaN values found, filling with 0")
        X = X.fillna(0)
    
    if np.isinf(X.values).any():
        print(f"⚠️  Warning: Infinite values found, replacing with large numbers")
        X = X.replace([np.inf, -np.inf], [1e10, -1e10])
    
    return X, y, feature_cols


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest with balanced class weights"""
    print("\n" + "="*80)
    print("Training Random Forest Classifier...")
    print("="*80)
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight='balanced',  # Handle class imbalance
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    print(f"\nTest Set Performance:")
    print(classification_report(y_test, y_pred, target_names=['LOSS', 'HIT_5R']))
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"                Predicted LOSS  Predicted HIT_5R")
    print(f"Actual LOSS          {cm[0][0]:4d}           {cm[0][1]:4d}")
    print(f"Actual HIT_5R        {cm[1][0]:4d}           {cm[1][1]:4d}")
    
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"\nROC-AUC Score: {roc_auc:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
    print(f"5-Fold CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    return model, roc_auc


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost with scale_pos_weight for imbalance"""
    if not XGBOOST_AVAILABLE:
        return None, 0
    
    print("\n" + "="*80)
    print("Training XGBoost Classifier...")
    print("="*80)
    
    # Calculate scale_pos_weight for class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Using scale_pos_weight: {scale_pos_weight:.2f}")
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    print(f"\nTest Set Performance:")
    print(classification_report(y_test, y_pred, target_names=['LOSS', 'HIT_5R']))
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"                Predicted LOSS  Predicted HIT_5R")
    print(f"Actual LOSS          {cm[0][0]:4d}           {cm[0][1]:4d}")
    print(f"Actual HIT_5R        {cm[1][0]:4d}           {cm[1][1]:4d}")
    
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"\nROC-AUC Score: {roc_auc:.4f}")
    
    return model, roc_auc


def train_lightgbm(X_train, y_train, X_test, y_test):
    """Train LightGBM with balanced class weights"""
    if not LIGHTGBM_AVAILABLE:
        return None, 0
    
    print("\n" + "="*80)
    print("Training LightGBM Classifier...")
    print("="*80)
    
    model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    print(f"\nTest Set Performance:")
    print(classification_report(y_test, y_pred, target_names=['LOSS', 'HIT_5R']))
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"                Predicted LOSS  Predicted HIT_5R")
    print(f"Actual LOSS          {cm[0][0]:4d}           {cm[0][1]:4d}")
    print(f"Actual HIT_5R        {cm[1][0]:4d}           {cm[1][1]:4d}")
    
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"\nROC-AUC Score: {roc_auc:.4f}")
    
    return model, roc_auc


def analyze_feature_importance(model, feature_cols, top_n=15):
    """Display top feature importances"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_

        # Normalize LightGBM importances (which are split counts)
        if isinstance(model, lgb.LGBMClassifier):
            importances = importances / importances.sum()

        indices = np.argsort(importances)[::-1]
        
        print(f"\nTop {top_n} Most Important Features:")
        print("-" * 50)
        for i in range(min(top_n, len(feature_cols))):
            idx = indices[i]
            print(f"{i+1:2d}. {feature_cols[idx]:30s} {importances[idx]:.4f}")


def find_optimal_threshold(model, X_test, y_test):
    """Find optimal probability threshold for classification"""
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    
    # Find threshold that maximizes F1 score
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    print(f"\nOptimal Classification Threshold: {best_threshold:.3f}")
    print(f"  At this threshold:")
    print(f"    Precision: {precisions[best_idx]:.3f}")
    print(f"    Recall: {recalls[best_idx]:.3f}")
    print(f"    F1-Score: {f1_scores[best_idx]:.3f}")
    
    return best_threshold


def main():
    parser = argparse.ArgumentParser(description='Train 5R Predictor Model')
    parser.add_argument('--timeframe', type=str, required=True, choices=['1min', '5min'],
                        help='Timeframe to train on (1min or 5min)')
    parser.add_argument('--features-file', type=str, default=None,
                        help='Optional: path to features JSONL file (overrides timeframe-based default)')
    parser.add_argument('--model-file', type=str, default=None,
                        help='Optional: path to save trained model (overrides default name)')
    args = parser.parse_args()
    
    timeframe = args.timeframe
    
    # Paths
    project_root = Path(__file__).parent.parent.parent
    features_file = project_root / "data" / "processed" / f"features_5R_{timeframe}.jsonl"
    # Allow overriding features file via CLI
    if args.features_file:
        features_file = Path(args.features_file)
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print(f"Training 5R Predictor - {timeframe.upper()} Timeframe")
    print("="*80)
    
    # Load data
    print(f"\nLoading features from: {features_file}")
    df = load_features(features_file)
    print(f"✅ Loaded {len(df)} trades")
    
    # Show class distribution
    pos_count = (df['label'] == 1).sum()
    neg_count = (df['label'] == 0).sum()
    print(f"\nClass Distribution:")
    print(f"  Positive (HIT_5R): {pos_count} ({pos_count/len(df)*100:.1f}%)")
    print(f"  Negative (LOSS): {neg_count} ({neg_count/len(df)*100:.1f}%)")
    print(f"  Class Imbalance Ratio: 1:{neg_count/pos_count:.1f}")
    
    # Prepare data
    X, y, feature_cols = prepare_data(df)
    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Samples: {len(X)}")
    
    # Group-aware time split: use the last 20% of unique entry days as test set
    # This keeps all entries from the same ticker+day together and preserves chronology.
    print("\nPerforming group-aware time split (last 20% of days as test set)...")
    df['entry_day'] = df['entry_date'].astype(str).str[:10]
    unique_days = sorted(df['entry_day'].unique())
    n_test_days = max(1, int(len(unique_days) * 0.20))
    test_days = set(unique_days[-n_test_days:])

    train_mask = ~df['entry_day'].isin(test_days)
    test_mask = df['entry_day'].isin(test_days)

    X_train = X[train_mask].reset_index(drop=True)
    X_test = X[test_mask].reset_index(drop=True)
    y_train = y[train_mask].reset_index(drop=True)
    y_test = y[test_mask].reset_index(drop=True)

    # If class distribution ends up with 0 positives in train or test (edge case), fall back to stratified split
    if (y_train.sum() == 0) or (y_test.sum() == 0):
        print("⚠️  Group-aware split produced empty positive class in train/test. Falling back to stratified random split.")
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except Exception:
            # As a last resort, do a random split without stratify
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=None
            )

    print(f"\nTrain set: {len(X_train)} samples ({(y_train==1).sum()} positive)")
    print(f"Test set:  {len(X_test)} samples ({(y_test==1).sum()} positive)")
    
    # Train models
    results = {}
    
    # Random Forest (always available)
    rf_model, rf_score = train_random_forest(X_train, y_train, X_test, y_test)
    results['RandomForest'] = (rf_model, rf_score)
    analyze_feature_importance(rf_model, feature_cols)
    find_optimal_threshold(rf_model, X_test, y_test)
    
    # XGBoost (if available)
    if XGBOOST_AVAILABLE:
        xgb_model, xgb_score = train_xgboost(X_train, y_train, X_test, y_test)
        if xgb_model:
            results['XGBoost'] = (xgb_model, xgb_score)
            analyze_feature_importance(xgb_model, feature_cols)
    
    # LightGBM (if available)
    if LIGHTGBM_AVAILABLE:
        lgb_model, lgb_score = train_lightgbm(X_train, y_train, X_test, y_test)
        if lgb_model:
            results['LightGBM'] = (lgb_model, lgb_score)
            analyze_feature_importance(lgb_model, feature_cols)
    
    # Select best model
    best_name = max(results.items(), key=lambda x: x[1][1])[0]
    best_model, best_score = results[best_name]
    
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    for name, (model, score) in results.items():
        marker = " ⭐ BEST" if name == best_name else ""
        print(f"{name:20s} ROC-AUC: {score:.4f}{marker}")
    
    # Save best model (allow CLI override)
    if args.model_file:
        model_file = Path(args.model_file)
        model_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        model_file = models_dir / f"5R_predictor_{timeframe}.pkl"

    joblib.dump(best_model, model_file)
    print(f"\n✅ Saved best model ({best_name}) to: {model_file}")
    
    # Save feature names
    feature_file = models_dir / f"5R_predictor_{timeframe}_features.txt"
    with open(feature_file, 'w') as f:
        f.write('\n'.join(feature_cols))
    print(f"✅ Saved feature names to: {feature_file}")
    
    print("\n" + "="*80)
    print("🎉 TRAINING COMPLETE!")
    print("="*80)
    print(f"Best Model: {best_name}")
    print(f"ROC-AUC Score: {best_score:.4f}")
    print(f"Timeframe: {timeframe}")
    print(f"\nNext Steps:")
    print(f"1. Compare with other timeframe")
    print(f"2. Review feature importances to understand what predicts success")
    print(f"3. Use model for live predictions: model = joblib.load('{model_file}')")


if __name__ == "__main__":
    main()
