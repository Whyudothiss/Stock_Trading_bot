#!/usr/bin/env python3
"""
Compare Pipeline 2 Models: Alpaca vs TwelveData

Compares 5R_predictor_5min.pkl (trained on Alpaca data) vs 
5R_predictor_5min_twelve.pkl (trained on TwelveData) to determine
which model performs better.

Shows:
- Accuracy, Precision, Recall, F1, ROC-AUC for both models
- Confusion matrices side-by-side
- Feature importance comparison
- Recommendation on which to use
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)


def load_features(filepath):
    """Load features from JSONL file"""
    features = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                features.append(json.loads(line))
    return pd.DataFrame(features)


def prepare_data(df, feature_cols):
    """Prepare features for prediction"""
    X = df[feature_cols].values
    y = df['label'].values
    return X, y


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model and return metrics dict"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    return metrics


def print_metrics_comparison(metrics_alpaca, metrics_twelve):
    """Print side-by-side metrics comparison"""
    print("\n" + "=" * 80)
    print("METRICS COMPARISON")
    print("=" * 80)
    
    print(f"\n{'Metric':<20s} {'Alpaca Model':<20s} {'TwelveData Model':<20s} {'Winner':<15s}")
    print("-" * 80)
    
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    metric_names = {
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1': 'F1 Score',
        'roc_auc': 'ROC-AUC'
    }
    
    wins = {'alpaca': 0, 'twelve': 0}
    
    for metric_key in metrics_to_compare:
        alpaca_val = metrics_alpaca[metric_key]
        twelve_val = metrics_twelve[metric_key]
        
        if alpaca_val > twelve_val:
            winner = "🏆 Alpaca"
            wins['alpaca'] += 1
        elif twelve_val > alpaca_val:
            winner = "🏆 TwelveData"
            wins['twelve'] += 1
        else:
            winner = "Tie"
        
        print(f"{metric_names[metric_key]:<20s} {alpaca_val:>6.4f} ({alpaca_val*100:>5.2f}%)  "
              f"{twelve_val:>6.4f} ({twelve_val*100:>5.2f}%)  {winner:<15s}")
    
    print("-" * 80)
    print(f"{'TOTAL WINS':<20s} {wins['alpaca']:<20d} {wins['twelve']:<20d}")
    
    return wins


def print_confusion_matrices(metrics_alpaca, metrics_twelve):
    """Print confusion matrices side-by-side"""
    print("\n" + "=" * 80)
    print("CONFUSION MATRICES")
    print("=" * 80)
    
    cm_a = metrics_alpaca['confusion_matrix']
    cm_t = metrics_twelve['confusion_matrix']
    
    tn_a, fp_a, fn_a, tp_a = cm_a.ravel()
    tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
    
    print("\n" + "-" * 40 + " ALPACA MODEL " + "-" * 40)
    print(f"                Predicted LOSS  Predicted HIT_5R")
    print(f"Actual LOSS          {tn_a:6d}           {fp_a:6d}")
    print(f"Actual HIT_5R        {fn_a:6d}           {tp_a:6d}")
    print(f"\nTrue Negatives:  {tn_a:4d} (correctly predicted losses)")
    print(f"False Positives: {fp_a:4d} (incorrectly predicted 5R hits)")
    print(f"False Negatives: {fn_a:4d} (missed 5R hits)")
    print(f"True Positives:  {tp_a:4d} (correctly predicted 5R hits)")
    
    print("\n" + "-" * 38 + " TWELVEDATA MODEL " + "-" * 38)
    print(f"                Predicted LOSS  Predicted HIT_5R")
    print(f"Actual LOSS          {tn_t:6d}           {fp_t:6d}")
    print(f"Actual HIT_5R        {fn_t:6d}           {tp_t:6d}")
    print(f"\nTrue Negatives:  {tn_t:4d} (correctly predicted losses)")
    print(f"False Positives: {fp_t:4d} (incorrectly predicted 5R hits)")
    print(f"False Negatives: {fn_t:4d} (missed 5R hits)")
    print(f"True Positives:  {tp_t:4d} (correctly predicted 5R hits)")
    
    # Compare key metrics
    print("\n" + "=" * 80)
    print("KEY DIFFERENCES")
    print("=" * 80)
    
    print(f"\nFalse Positives (bad trades approved):")
    print(f"  Alpaca: {fp_a} | TwelveData: {fp_t} | Difference: {abs(fp_a - fp_t)} "
          f"({'TwelveData better' if fp_t < fp_a else 'Alpaca better' if fp_a < fp_t else 'Tie'})")
    
    print(f"\nFalse Negatives (good trades missed):")
    print(f"  Alpaca: {fn_a} | TwelveData: {fn_t} | Difference: {abs(fn_a - fn_t)} "
          f"({'TwelveData better' if fn_t < fn_a else 'Alpaca better' if fn_a < fn_t else 'Tie'})")
    
    print(f"\nTrue Positives (good trades caught):")
    print(f"  Alpaca: {tp_a} | TwelveData: {tp_t} | Difference: {abs(tp_a - tp_t)} "
          f"({'TwelveData better' if tp_t > tp_a else 'Alpaca better' if tp_a > tp_t else 'Tie'})")


def compare_feature_importance(model_alpaca, model_twelve, feature_cols, top_n=10):
    """Compare feature importances between models"""
    print("\n" + "=" * 80)
    print(f"TOP {top_n} FEATURE IMPORTANCE COMPARISON")
    print("=" * 80)
    
    if not hasattr(model_alpaca, 'feature_importances_') or not hasattr(model_twelve, 'feature_importances_'):
        print("⚠️  Models don't have feature_importances_ attribute")
        return
    
    imp_a = model_alpaca.feature_importances_
    imp_t = model_twelve.feature_importances_
    
    # Get top features from each model
    top_indices_a = np.argsort(imp_a)[::-1][:top_n]
    top_indices_t = np.argsort(imp_t)[::-1][:top_n]
    
    # Combine unique features
    all_top_indices = sorted(set(list(top_indices_a) + list(top_indices_t)))
    
    print(f"\n{'Feature':<35s} {'Alpaca':<12s} {'TwelveData':<12s} {'Diff':<10s}")
    print("-" * 80)
    
    for idx in all_top_indices[:top_n]:
        feature_name = feature_cols[idx]
        diff = imp_t[idx] - imp_a[idx]
        diff_str = f"{diff:+.4f}"
        
        print(f"{feature_name:<35s} {imp_a[idx]:>8.4f}     {imp_t[idx]:>8.4f}     {diff_str:>10s}")


def analyze_prediction_agreement(metrics_alpaca, metrics_twelve, y_test):
    """Analyze where models agree/disagree"""
    print("\n" + "=" * 80)
    print("PREDICTION AGREEMENT ANALYSIS")
    print("=" * 80)
    
    y_pred_a = metrics_alpaca['y_pred']
    y_pred_t = metrics_twelve['y_pred']
    
    # Agreement
    agree_mask = (y_pred_a == y_pred_t)
    agree_correct = agree_mask & (y_pred_a == y_test)
    agree_incorrect = agree_mask & (y_pred_a != y_test)
    
    # Disagreement
    disagree_mask = (y_pred_a != y_pred_t)
    alpaca_correct = disagree_mask & (y_pred_a == y_test)
    twelve_correct = disagree_mask & (y_pred_t == y_test)
    both_wrong = disagree_mask & (y_pred_a != y_test) & (y_pred_t != y_test)
    
    total = len(y_test)
    
    print(f"\nAgreement (both models predict same):")
    print(f"  Both Correct:   {agree_correct.sum():4d} ({agree_correct.sum()/total*100:5.2f}%)")
    print(f"  Both Incorrect: {agree_incorrect.sum():4d} ({agree_incorrect.sum()/total*100:5.2f}%)")
    print(f"  Total Agreement: {agree_mask.sum():4d} ({agree_mask.sum()/total*100:5.2f}%)")
    
    print(f"\nDisagreement (models predict differently):")
    print(f"  Alpaca Correct, TwelveData Wrong:  {alpaca_correct.sum():4d} ({alpaca_correct.sum()/total*100:5.2f}%)")
    print(f"  TwelveData Correct, Alpaca Wrong:  {twelve_correct.sum():4d} ({twelve_correct.sum()/total*100:5.2f}%)")
    print(f"  Both Wrong:                        {both_wrong.sum():4d} ({both_wrong.sum()/total*100:5.2f}%)")
    print(f"  Total Disagreement: {disagree_mask.sum():4d} ({disagree_mask.sum()/total*100:5.2f}%)")


def main():
    """Compare both Pipeline 2 models on their respective data sources"""
    print("=" * 80)
    print("PIPELINE 2 MODEL COMPARISON: Alpaca vs TwelveData")
    print("=" * 80)
    print("\n⚠️  IMPORTANT: Each model will be tested on its own data source")
    print("   - Alpaca model tested on Alpaca features")
    print("   - TwelveData model tested on TwelveData features")
    print("   This ensures fair comparison (no data source mismatch)")
    
    # Paths
    model_alpaca_path = Path('models/5R_predictor_5min.pkl')
    model_twelve_path = Path('models/5R_predictor_5min_twelve.pkl')
    features_alpaca_path = Path('data/processed/features_5R_5min.jsonl')
    features_twelve_path = Path('data/processed/features_5R_5min_twelve.jsonl')
    
    # Check if both models and datasets exist
    if not model_alpaca_path.exists():
        print(f"\n❌ Alpaca model not found: {model_alpaca_path}")
        return
    
    if not model_twelve_path.exists():
        print(f"\n❌ TwelveData model not found: {model_twelve_path}")
        return
    
    if not features_alpaca_path.exists():
        print(f"\n❌ Alpaca features not found: {features_alpaca_path}")
        return
    
    if not features_twelve_path.exists():
        print(f"\n❌ TwelveData features not found: {features_twelve_path}")
        return
    
    # Load models
    print("\n" + "=" * 80)
    print("LOADING MODELS")
    print("=" * 80)
    model_alpaca = joblib.load(model_alpaca_path)
    model_twelve = joblib.load(model_twelve_path)
    print("✅ Both models loaded")
    
    # ========== EVALUATE ALPACA MODEL ON ALPACA DATA ==========
    print("\n" + "=" * 80)
    print("EVALUATING ALPACA MODEL ON ALPACA DATA")
    print("=" * 80)
    
    print(f"\nLoading Alpaca features from: {features_alpaca_path}")
    df_alpaca = load_features(features_alpaca_path)
    print(f"✅ Loaded {len(df_alpaca)} Alpaca trades")
    
    # Show class distribution
    pos_alpaca = (df_alpaca['label'] == 1).sum()
    neg_alpaca = (df_alpaca['label'] == 0).sum()
    print(f"\nAlpaca Class Distribution:")
    print(f"  HIT_5R (positive): {pos_alpaca} ({pos_alpaca/len(df_alpaca)*100:.1f}%)")
    print(f"  LOSS (negative):   {neg_alpaca} ({neg_alpaca/len(df_alpaca)*100:.1f}%)")
    
    # Get feature columns
    metadata_cols = ['ticker', 'entry_date', 'label', 'timeframe']
    feature_cols_alpaca = [col for col in df_alpaca.columns if col not in metadata_cols]
    print(f"\nAlpaca features: {len(feature_cols_alpaca)}")
    
    # Temporal split (last 20% of days for test)
    df_alpaca['entry_day'] = df_alpaca['entry_date'].astype(str).str[:10]
    unique_days_alpaca = sorted(df_alpaca['entry_day'].unique())
    n_test_days_alpaca = max(1, int(len(unique_days_alpaca) * 0.20))
    test_days_alpaca = set(unique_days_alpaca[-n_test_days_alpaca:])
    
    test_mask_alpaca = df_alpaca['entry_day'].isin(test_days_alpaca)
    df_test_alpaca = df_alpaca[test_mask_alpaca].reset_index(drop=True)
    
    X_test_alpaca = df_test_alpaca[feature_cols_alpaca].values
    y_test_alpaca = df_test_alpaca['label'].values
    
    print(f"\nAlpaca test set: {len(X_test_alpaca)} trades ({(y_test_alpaca==1).sum()} winners)")
    print(f"Test period: {df_test_alpaca['entry_day'].min()} to {df_test_alpaca['entry_day'].max()}")
    
    # Evaluate Alpaca model on Alpaca data
    print("\nEvaluating Alpaca model on Alpaca test set...")
    metrics_alpaca = evaluate_model(model_alpaca, X_test_alpaca, y_test_alpaca, "Alpaca")
    
    # ========== EVALUATE TWELVEDATA MODEL ON TWELVEDATA DATA ==========
    print("\n" + "=" * 80)
    print("EVALUATING TWELVEDATA MODEL ON TWELVEDATA DATA")
    print("=" * 80)
    
    print(f"\nLoading TwelveData features from: {features_twelve_path}")
    df_twelve = load_features(features_twelve_path)
    print(f"✅ Loaded {len(df_twelve)} TwelveData trades")
    
    # Show class distribution
    pos_twelve = (df_twelve['label'] == 1).sum()
    neg_twelve = (df_twelve['label'] == 0).sum()
    print(f"\nTwelveData Class Distribution:")
    print(f"  HIT_5R (positive): {pos_twelve} ({pos_twelve/len(df_twelve)*100:.1f}%)")
    print(f"  LOSS (negative):   {neg_twelve} ({neg_twelve/len(df_twelve)*100:.1f}%)")
    
    # Get feature columns
    feature_cols_twelve = [col for col in df_twelve.columns if col not in metadata_cols]
    print(f"\nTwelveData features: {len(feature_cols_twelve)}")
    
    # Temporal split (last 20% of days for test)
    df_twelve['entry_day'] = df_twelve['entry_date'].astype(str).str[:10]
    unique_days_twelve = sorted(df_twelve['entry_day'].unique())
    n_test_days_twelve = max(1, int(len(unique_days_twelve) * 0.20))
    test_days_twelve = set(unique_days_twelve[-n_test_days_twelve:])
    
    test_mask_twelve = df_twelve['entry_day'].isin(test_days_twelve)
    df_test_twelve = df_twelve[test_mask_twelve].reset_index(drop=True)
    
    X_test_twelve = df_test_twelve[feature_cols_twelve].values
    y_test_twelve = df_test_twelve['label'].values
    
    print(f"\nTwelveData test set: {len(X_test_twelve)} trades ({(y_test_twelve==1).sum()} winners)")
    print(f"Test period: {df_test_twelve['entry_day'].min()} to {df_test_twelve['entry_day'].max()}")
    
    # Evaluate TwelveData model on TwelveData data
    print("\nEvaluating TwelveData model on TwelveData test set...")
    metrics_twelve = evaluate_model(model_twelve, X_test_twelve, y_test_twelve, "TwelveData")
    
    # ========== PRINT COMPARISONS ==========
    wins = print_metrics_comparison(metrics_alpaca, metrics_twelve)
    print_confusion_matrices(metrics_alpaca, metrics_twelve)
    
    # Compare feature importance (use Alpaca features since both should have same features)
    compare_feature_importance(model_alpaca, model_twelve, feature_cols_alpaca, top_n=15)
    
    # Can't do prediction agreement analysis since test sets are different
    print("\n" + "=" * 80)
    print("PREDICTION AGREEMENT ANALYSIS")
    print("=" * 80)
    print("\n⚠️  Skipping prediction agreement analysis")
    print("   Reason: Models tested on different data sources (Alpaca vs TwelveData)")
    print("   Cannot compare predictions when test sets differ")
    
    # Final recommendation
    print("\n" + "=" * 80)
    print("🎯 FINAL RECOMMENDATION")
    print("=" * 80)
    
    if wins['twelve'] > wins['alpaca']:
        winner = "TwelveData"
        winner_model = model_twelve_path
        winner_metrics = metrics_twelve
    elif wins['alpaca'] > wins['twelve']:
        winner = "Alpaca"
        winner_model = model_alpaca_path
        winner_metrics = metrics_alpaca
    else:
        # If tied on wins, use ROC-AUC as tiebreaker
        if metrics_twelve['roc_auc'] > metrics_alpaca['roc_auc']:
            winner = "TwelveData"
            winner_model = model_twelve_path
            winner_metrics = metrics_twelve
        else:
            winner = "Alpaca"
            winner_model = model_alpaca_path
            winner_metrics = metrics_alpaca
    
    print(f"\n🏆 Winner: {winner} Model")
    print(f"\nModel path: {winner_model}")
    print(f"\nPerformance on its own data source:")
    print(f"  Accuracy:  {winner_metrics['accuracy']:.4f} ({winner_metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {winner_metrics['precision']:.4f} ({winner_metrics['precision']*100:.2f}%)")
    print(f"  Recall:    {winner_metrics['recall']:.4f} ({winner_metrics['recall']*100:.2f}%)")
    print(f"  F1 Score:  {winner_metrics['f1']:.4f} ({winner_metrics['f1']*100:.2f}%)")
    print(f"  ROC-AUC:   {winner_metrics['roc_auc']:.4f} ({winner_metrics['roc_auc']*100:.2f}%)")
    
    print(f"\n💡 Recommendation: Use the {winner} model ({winner_model.name})")
    print(f"\n📊 Key Insight:")
    if winner == "TwelveData":
        print(f"   - TwelveData model performs better on TwelveData features")
        print(f"   - Since you're using TwelveData for intraday downloads in production,")
        print(f"     the TwelveData model will see data that matches its training distribution")
        print(f"   - ROC-AUC: {metrics_twelve['roc_auc']*100:.2f}% (TwelveData) vs {metrics_alpaca['roc_auc']*100:.2f}% (Alpaca)")
    else:
        print(f"   - Alpaca model performs better on Alpaca features")
        print(f"   - However, you're using TwelveData for intraday downloads in production")
        print(f"   - Consider: There may be a data distribution mismatch in live trading")
        print(f"   - ROC-AUC: {metrics_alpaca['roc_auc']*100:.2f}% (Alpaca) vs {metrics_twelve['roc_auc']*100:.2f}% (TwelveData)")
    
    print("\n" + "=" * 80)
    print("⚠️  CRITICAL CONSIDERATION FOR PRODUCTION")
    print("=" * 80)
    print("\nYour simulation uses TwelveData API for downloading intraday data.")
    print("Therefore, features extracted in production will match TwelveData's data format.")
    print("\n✅ STRONGLY RECOMMEND: Use TwelveData model for production")
    print("   Even if Alpaca model has higher ROC-AUC on Alpaca data,")
    print("   the TwelveData model will generalize better to production data")
    print("   since it was trained on the same data source you use in live trading.")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
