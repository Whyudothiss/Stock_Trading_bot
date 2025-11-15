#!/usr/bin/env python3
"""
Train Consolidation Detection Model (Pipeline 1)

Trains a binary classifier to detect valid consolidation patterns
ready to break out using features extracted from 10-day lookback.

Models tested:
- Random Forest
- XGBoost
- LightGBM

Outputs:
- Trained model (.pkl)
- Performance metrics
- Feature importance
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, precision_recall_curve, roc_curve
)
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt


class ConsolidationModelTrainer:
    """Train and evaluate consolidation detection models."""
    
    def __init__(self, positive_features_file: str, negative_features_file: str):
        """
        Args:
            positive_features_file: Path to positive features JSONL
            negative_features_file: Path to negative features JSONL
        """
        self.positive_file = positive_features_file
        self.negative_file = negative_features_file
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.models = {}
    
    def load_data(self):
        """Load and prepare training data."""
        print("=" * 70)
        print("LOADING TRAINING DATA")
        print("=" * 70)
        
        # Load positive features
        positives = []
        with open(self.positive_file, 'r') as f:
            for line in f:
                if line.strip():
                    positives.append(json.loads(line))
        
        # Load negative features
        negatives = []
        with open(self.negative_file, 'r') as f:
            for line in f:
                if line.strip():
                    negatives.append(json.loads(line))
        
        print(f"\nPositive examples: {len(positives)}")
        print(f"Negative examples: {len(negatives)}")
        
        # Combine and create DataFrame
        all_data = positives + negatives
        df = pd.DataFrame(all_data)
        
        # Remove metadata columns
        metadata_cols = ['ticker', 'date', 'negative_category']
        feature_cols = [col for col in df.columns if col not in metadata_cols + ['label']]
        
        X = df[feature_cols].values
        y = df['label'].values
        
        self.feature_names = feature_cols
        
        print(f"\nTotal examples: {len(X)}")
        print(f"Features: {len(feature_cols)}")
        print(f"Class balance: {np.sum(y == 1)} positive ({np.sum(y == 1)/len(y)*100:.1f}%), "
              f"{np.sum(y == 0)} negative ({np.sum(y == 0)/len(y)*100:.1f}%)")
        
        # Train/test split (80/20)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTrain set: {len(self.X_train)} examples")
        print(f"Test set: {len(self.X_test)} examples")
        
        return df
    
    def train_random_forest(self):
        """Train Random Forest classifier."""
        print("\n" + "=" * 70)
        print("TRAINING RANDOM FOREST")
        print("=" * 70)
        
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'  # Handle imbalance
        )
        
        rf.fit(self.X_train, self.y_train)
        self.models['random_forest'] = rf
        
        # Evaluate
        train_score = rf.score(self.X_train, self.y_train)
        test_score = rf.score(self.X_test, self.y_test)
        
        print(f"\nTrain accuracy: {train_score:.4f}")
        print(f"Test accuracy: {test_score:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(rf, self.X_train, self.y_train, cv=5, scoring='roc_auc')
        print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return rf
    
    def train_xgboost(self):
        """Train XGBoost classifier."""
        print("\n" + "=" * 70)
        print("TRAINING XGBOOST")
        print("=" * 70)
        
        # Calculate scale_pos_weight for imbalance
        scale_pos_weight = np.sum(self.y_train == 0) / np.sum(self.y_train == 1)
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1
        )
        
        xgb_model.fit(self.X_train, self.y_train)
        self.models['xgboost'] = xgb_model
        
        # Evaluate
        train_score = xgb_model.score(self.X_train, self.y_train)
        test_score = xgb_model.score(self.X_test, self.y_test)
        
        print(f"\nTrain accuracy: {train_score:.4f}")
        print(f"Test accuracy: {test_score:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(xgb_model, self.X_train, self.y_train, cv=5, scoring='roc_auc')
        print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return xgb_model
    
    def train_lightgbm(self):
        """Train LightGBM classifier."""
        print("\n" + "=" * 70)
        print("TRAINING LIGHTGBM")
        print("=" * 70)
        
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        lgb_model.fit(self.X_train, self.y_train)
        self.models['lightgbm'] = lgb_model
        
        # Evaluate
        train_score = lgb_model.score(self.X_train, self.y_train)
        test_score = lgb_model.score(self.X_test, self.y_test)
        
        print(f"\nTrain accuracy: {train_score:.4f}")
        print(f"Test accuracy: {test_score:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(lgb_model, self.X_train, self.y_train, cv=5, scoring='roc_auc')
        print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return lgb_model
    
    def evaluate_model(self, model_name: str):
        """Detailed evaluation of a trained model."""
        print("\n" + "=" * 70)
        print(f"DETAILED EVALUATION: {model_name.upper()}")
        print("=" * 70)
        
        model = self.models[model_name]
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, target_names=['Negative', 'Positive']))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print("\nConfusion Matrix:")
        print(f"                Predicted Neg  Predicted Pos")
        print(f"Actual Neg      {cm[0,0]:6d}         {cm[0,1]:6d}")
        print(f"Actual Pos      {cm[1,0]:6d}         {cm[1,1]:6d}")
        
        # ROC-AUC
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        print(f"\nROC-AUC Score: {roc_auc:.4f}")
        
        # Feature importance (top 15)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:15]
            
            print("\nTop 15 Most Important Features:")
            for i, idx in enumerate(indices):
                print(f"  {i+1:2d}. {self.feature_names[idx]:30s} {importances[idx]:.4f}")
        
        return {
            'model_name': model_name,
            'test_accuracy': model.score(self.X_test, self.y_test),
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(self.y_test, y_pred, output_dict=True)
        }
    
    def compare_models(self):
        """Compare all trained models."""
        print("\n" + "=" * 70)
        print("MODEL COMPARISON")
        print("=" * 70)
        
        results = []
        for model_name in self.models.keys():
            model = self.models[model_name]
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            test_acc = model.score(self.X_test, self.y_test)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            results.append({
                'Model': model_name,
                'Test Accuracy': test_acc,
                'ROC-AUC': roc_auc
            })
        
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('ROC-AUC', ascending=False)
        
        print("\n", df_results.to_string(index=False))
        
        # Best model
        best_model_name = df_results.iloc[0]['Model']
        print(f"\n🏆 Best model: {best_model_name.upper()}")
        
        return best_model_name
    
    def save_best_model(self, model_name: str, output_file: str):
        """Save the best model to disk."""
        model = self.models[model_name]
        
        model_data = {
            'model': model,
            'feature_names': self.feature_names,
            'model_name': model_name,
            'trained_date': datetime.now().isoformat()
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n💾 Saved {model_name} model to: {output_file}")


if __name__ == '__main__':
    # Initialize trainer
    trainer = ConsolidationModelTrainer(
        positive_features_file='data/processed/features_positive.jsonl',
        negative_features_file='data/processed/features_negative.jsonl'
    )
    
    # Load data
    df = trainer.load_data()
    
    # Train all models
    trainer.train_random_forest()
    trainer.train_xgboost()
    trainer.train_lightgbm()
    
    # Compare models
    best_model = trainer.compare_models()
    
    # Detailed evaluation of best model
    eval_results = trainer.evaluate_model(best_model)
    
    # Save best model
    trainer.save_best_model(
        model_name=best_model,
        output_file='models/consolidation_detector.pkl'
    )
    
    print("\n" + "=" * 70)
    print("✨ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nBest model: {best_model}")
    print(f"Test accuracy: {eval_results['test_accuracy']:.4f}")
    print(f"ROC-AUC: {eval_results['roc_auc']:.4f}")
    print(f"\nModel saved to: models/consolidation_detector.pkl")
    print(f"\nNext step: Use this model to scan stocks in real-time")
