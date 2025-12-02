#!/usr/bin/env python3
"""
Evaluate existing consolidation_detector.pkl model
Shows accuracy, precision, recall, ROC-AUC on test set
"""

import json
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score
)


def load_test_data():
    """Load test data from features files"""
    # Load positive features
    positives = []
    with open('data/processed/features_positive.jsonl', 'r') as f:
        for line in f:
            if line.strip():
                positives.append(json.loads(line))
    
    # Load negative features
    negatives = []
    with open('data/processed/features_negative.jsonl', 'r') as f:
        for line in f:
            if line.strip():
                negatives.append(json.loads(line))
    
    # Combine
    all_data = positives + negatives
    df = pd.DataFrame(all_data)
    
    # Remove metadata
    metadata_cols = ['ticker', 'date', 'negative_category']
    feature_cols = [col for col in df.columns if col not in metadata_cols + ['label']]
    
    X = df[feature_cols].values
    y = df['label'].values
    
    return X, y, feature_cols


def evaluate_model():
    """Load model and evaluate on test set"""
    print("=" * 70)
    print("CONSOLIDATION DETECTOR MODEL EVALUATION")
    print("=" * 70)
    
    # Load model
    model_data = joblib.load('models/consolidation_detector.pkl')
    model = model_data['model']
    feature_names = model_data['feature_names']
    model_name = model_data.get('model_name', 'Unknown')
    
    print(f"\nModel type: {model_name}")
    print(f"Features: {len(feature_names)}")
    
    # Load data
    X, y, feature_cols = load_test_data()
    
    print(f"\nTotal samples: {len(X)}")
    print(f"Positive samples: {np.sum(y == 1)} ({np.sum(y == 1)/len(y)*100:.1f}%)")
    print(f"Negative samples: {np.sum(y == 0)} ({np.sum(y == 0)/len(y)*100:.1f}%)")
    
    # Make predictions (use 80/20 split like training to get test set)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print("\n" + "=" * 70)
    print("TEST SET PERFORMANCE METRICS")
    print("=" * 70)
    print(f"\nAccuracy:  {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}  ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f}  ({recall*100:.2f}%)")
    print(f"F1 Score:  {f1:.4f}  ({f1*100:.2f}%)")
    print(f"ROC-AUC:   {roc_auc:.4f}  ({roc_auc*100:.2f}%)")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print("\n" + "=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)
    print(f"\n                Predicted Negative  Predicted Positive")
    print(f"Actual Negative      {tn:6d}              {fp:6d}")
    print(f"Actual Positive      {fn:6d}              {tp:6d}")
    
    print(f"\nTrue Negatives:  {tn} (correctly rejected bad setups)")
    print(f"False Positives: {fp} (incorrectly approved bad setups)")
    print(f"False Negatives: {fn} (incorrectly rejected good setups)")
    print(f"True Positives:  {tp} (correctly approved good setups)")
    
    # Classification report
    print("\n" + "=" * 70)
    print("DETAILED CLASSIFICATION REPORT")
    print("=" * 70)
    print("\n" + classification_report(y_test, y_pred, 
                                      target_names=['Negative (Bad Setup)', 'Positive (Good Setup)']))
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]
        
        print("=" * 70)
        print("TOP 15 MOST IMPORTANT FEATURES")
        print("=" * 70)
        for i, idx in enumerate(indices):
            print(f"{i+1:2d}. {feature_names[idx]:35s} {importances[idx]:.4f}")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print(f"ROC-AUC: {roc_auc*100:.2f}%")
    print(f"Precision: {precision*100:.2f}% (when model says 'good setup', it's right {precision*100:.1f}% of time)")
    print(f"Recall: {recall*100:.2f}% (model catches {recall*100:.1f}% of all good setups)")


if __name__ == '__main__':
    evaluate_model()
