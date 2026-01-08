#!/usr/bin/env python3
"""
Neural Network Stacking for Pipeline 2

Approach: Use Random Forest predictions as an additional feature
for a neural network to learn when RF is over/under-confident.

Steps:
1. Generate out-of-fold RF predictions using TimeSeriesSplit
2. Add RF probability as 18th feature
3. Train simple 2-3 layer NN on augmented features
4. Compare performance: RF alone vs NN stacking
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import json
from datetime import datetime

# Neural network imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
    
    class SimpleStackingNN(nn.Module):
        """
        Simple 3-layer neural network for stacking on top of RF
        
        Architecture:
        - Input: 18 features (17 original + 1 RF probability)
        - Hidden 1: 32 neurons + ReLU + Dropout(0.3)
        - Hidden 2: 16 neurons + ReLU + Dropout(0.2)
        - Output: 1 neuron + Sigmoid (probability)
        """
        def __init__(self, input_size=18):
            super(SimpleStackingNN, self).__init__()
            self.fc1 = nn.Linear(input_size, 32)
            self.dropout1 = nn.Dropout(0.3)
            self.fc2 = nn.Linear(32, 16)
            self.dropout2 = nn.Dropout(0.2)
            self.fc3 = nn.Linear(16, 1)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.dropout1(x)
            x = torch.relu(self.fc2(x))
            x = self.dropout2(x)
            x = torch.sigmoid(self.fc3(x))
            return x
            
except ImportError:
    TORCH_AVAILABLE = False
    SimpleStackingNN = None
def generate_oof_predictions(X, y, dates, rf_model, feature_names):
    """
    Generate out-of-fold predictions using TimeSeriesSplit
    
    This ensures no data leakage - each prediction uses only past data
    """
    print("Generating out-of-fold RF predictions...")
    
    # Sort by date to ensure temporal order
    sort_idx = np.argsort(dates)
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]
    dates_sorted = dates[sort_idx]
    
    oof_predictions = np.zeros(len(X))
    
    # 5-fold time series split
    tscv = TimeSeriesSplit(n_splits=5)
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_sorted)):
        print(f"   Fold {fold + 1}/5...")
        
        X_train_fold = X_sorted[train_idx]
        y_train_fold = y_sorted[train_idx]
        X_val_fold = X_sorted[val_idx]
        
        # Train RF on this fold
        from sklearn.ensemble import RandomForestClassifier
        fold_rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=42
        )
        fold_rf.fit(pd.DataFrame(X_train_fold, columns=feature_names), y_train_fold)
        
        # Predict on validation fold
        val_probs = fold_rf.predict_proba(pd.DataFrame(X_val_fold, columns=feature_names))[:, 1]
        oof_predictions[sort_idx[val_idx]] = val_probs
    
    print(f"   ✅ Generated OOF predictions for {len(oof_predictions)} samples")
    return oof_predictions


def train_stacking_nn(X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    """
    Train neural network with early stopping
    """
    if not TORCH_AVAILABLE:
        print("❌ PyTorch not installed")
        return None
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = SimpleStackingNN(input_size=X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop with early stopping
    best_auc = 0
    patience = 10
    patience_counter = 0
    best_model_state = None
    
    print(f"\nTraining neural network...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor).numpy()
            test_auc = roc_auc_score(y_test, test_outputs)
        
        avg_train_loss = train_loss / len(train_loader)
        
        if (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f} - Test AUC: {test_auc:.4f}")
        
        # Early stopping
        if test_auc > best_auc:
            best_auc = test_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"   Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model, best_auc


def main():
    """
    Main stacking pipeline
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-source', choices=['alpaca', 'twelve'], default='twelve',
                       help='Data source to train on (alpaca or twelve). Default: twelve')
    args = parser.parse_args()
    
    print("="*80)
    print("NEURAL NETWORK STACKING FOR PIPELINE 2")
    print("="*80)
    print(f"\nData source: {args.data_source.upper()}")
    
    if not TORCH_AVAILABLE:
        print("\n❌ PyTorch not installed. Install with: pip install torch")
        return
    
    # Load data
    print("\n1. Loading training data...")
    if args.data_source == 'twelve':
        data_file = Path('data/processed/features_5R_5min_twelve.jsonl')
        model_file = Path('models/5R_predictor_5min_twelve.pkl')
    else:
        data_file = Path('data/processed/features_5R_5min.jsonl')
        model_file = Path('models/5R_predictor_5min.pkl')
    
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        if args.data_source == 'twelve':
            print("   Run: python3 src/ml_pipeline/extract_5R_features_batch.py --use-twelve")
        return
    
    records = []
    with open(data_file, 'r') as f:
        for line in f:
            records.append(json.loads(line))
    
    df = pd.DataFrame(records)
    print(f"   Loaded {len(df)} samples from {args.data_source.upper()} data")
    
    # Check which column name is used for target
    if 'hit_5R' in df.columns:
        target_col = 'hit_5R'
    elif 'label' in df.columns:
        target_col = 'label'
    elif 'outcome' in df.columns:
        target_col = 'outcome'
    else:
        print(f"❌ Could not find target column. Available columns: {list(df.columns)}")
        return
    
    print(f"   Positive class: {df[target_col].sum()} ({df[target_col].mean()*100:.1f}%)")
    
    # Load existing RF model
    print(f"\n2. Loading existing Random Forest model ({args.data_source.upper()})...")
    if not model_file.exists():
        print(f"❌ Model file not found: {model_file}")
        if args.data_source == 'twelve':
            print("   Run: python3 src/ml_pipeline/train_5R_model.py --timeframe 5min --use-twelve")
        return
        
    rf_model = joblib.load(model_file)
    with open('models/5R_predictor_5min_features.txt', 'r') as f:
        feature_names = [line.strip() for line in f]
    
    # Prepare features
    X = df[feature_names].values
    y = df[target_col].values
    dates = pd.to_datetime(df['entry_date']).values
    
    # Generate OOF predictions
    print("\n3. Generating out-of-fold RF predictions...")
    oof_rf_probs = generate_oof_predictions(X, y, dates, rf_model, feature_names)
    
    # Augment features with RF probability
    X_augmented = np.column_stack([X, oof_rf_probs])
    augmented_feature_names = feature_names + ['rf_probability']
    
    print(f"   Augmented features: {X_augmented.shape[1]} ({len(feature_names)} original + 1 RF prob)")
    
    # Temporal split (last 20% for test)
    print("\n4. Splitting data (temporal split - last 20% for test)...")
    sort_idx = np.argsort(dates)
    n_train = int(len(sort_idx) * 0.8)
    
    train_idx = sort_idx[:n_train]
    test_idx = sort_idx[n_train:]
    
    X_train = X_augmented[train_idx]
    y_train = y[train_idx]
    X_test = X_augmented[test_idx]
    y_test = y[test_idx]
    
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")
    
    # Baseline: RF performance on test set
    print("\n5. Baseline RF performance on test set...")
    rf_test_probs = rf_model.predict_proba(pd.DataFrame(X[test_idx], columns=feature_names))[:, 1]
    rf_auc = roc_auc_score(y_test, rf_test_probs)
    print(f"   RF ROC-AUC: {rf_auc:.4f}")
    
    # Train stacking NN
    print("\n6. Training stacking neural network...")
    nn_model, nn_auc = train_stacking_nn(X_train, y_train, X_test, y_test, epochs=100)
    
    # Compare results
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)
    print(f"\nRandom Forest (baseline):     {rf_auc:.4f}")
    print(f"Neural Network (stacking):    {nn_auc:.4f}")
    print(f"Improvement:                  {(nn_auc - rf_auc):.4f} ({(nn_auc - rf_auc)/rf_auc*100:+.2f}%)")
    
    if nn_auc > rf_auc:
        print("\n✅ Neural network stacking improves performance!")
        
        # Save model
        suffix = '_twelve' if args.data_source == 'twelve' else ''
        save_path = Path(f'models/5R_predictor_5min_nn_stacked{suffix}.pt')
        torch.save({
            'model_state_dict': nn_model.state_dict(),
            'feature_names': augmented_feature_names,
            'test_auc': nn_auc,
            'data_source': args.data_source
        }, save_path)
        print(f"   Saved to: {save_path}")
        
        # Save feature names
        feature_file = Path(f'models/5R_predictor_5min_nn_stacked{suffix}_features.txt')
        with open(feature_file, 'w') as f:
            for fname in augmented_feature_names:
                f.write(fname + '\n')
        print(f"   Features saved to: {feature_file}")
    else:
        print("\n⚠️  Neural network did not improve over RF. Try:")
        print("   - Different architecture (more layers, different sizes)")
        print("   - More training data")
        print("   - Different features")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
