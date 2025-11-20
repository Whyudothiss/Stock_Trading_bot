#!/usr/bin/env python3
"""
Visualize a single decision tree from the Random Forest model.
"""

import pickle
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load the trained model
print("Loading model...")
with open('models/consolidation_detector.pkl', 'rb') as f:
    saved_data = pickle.load(f)

# Extract the actual model
model = saved_data['model']
feature_names_saved = saved_data.get('feature_names', None)

# Get one tree from the forest
print(f"\nRandom Forest has {len(model.estimators_)} trees")
print("Visualizing tree #0 (first tree)...\n")

# Get feature names (we need to reconstruct them)
feature_names = [
    'range_pct', 'position_in_range', 'price_change_pct', 'volatility',
    'avg_daily_range_pct', 'volume_trend', 'relative_volume', 'volume_volatility',
    'volume_spike_count', 'avg_volume', 'dist_from_sma10_pct', 'dist_from_sma20_pct',
    'dist_from_sma50_pct', 'sma_alignment', 'above_sma10', 'above_sma20',
    'above_sma50', 'days_hugging_10sma', 'avg_dist_from_10sma_pct', 'max_dist_from_10sma_pct',
    'days_testing_10sma', 'pct_above_10sma', 'roc_pct', 'consecutive_up_days',
    'consecutive_down_days', 'up_down_ratio', 'max_single_day_gain', 'max_single_day_loss',
    'days_near_high', 'days_near_low', 'range_contraction', 'consolidation_tightness',
    'atr_trend', 'body_to_range_ratio', 'momentum_strength'
]

# Plot the tree
plt.figure(figsize=(40, 20))
plot_tree(
    model.estimators_[0],  # First tree
    feature_names=feature_names,
    class_names=['Not Consolidation', 'Consolidation'],
    filled=True,
    rounded=True,
    fontsize=10,
    max_depth=4  # Only show top 4 levels (otherwise too complex)
)

plt.title('Decision Tree #0 from Random Forest\n(Showing top 4 levels only)', fontsize=20)
plt.tight_layout()
plt.savefig('results/decision_tree_visualization.png', dpi=150, bbox_inches='tight')
print("✅ Saved visualization to: results/decision_tree_visualization.png")
print("\nHow to read it:")
print("- Each box is a decision node")
print("- Top line: The decision rule (e.g., 'dist_from_sma50_pct <= 5.2')")
print("- 'gini': Impurity (lower = more pure)")
print("- 'samples': How many training examples reach this node")
print("- 'value': [negatives, positives] at this node")
print("- 'class': Final prediction if this was a leaf")
print("- Color: Orange = predicts Consolidation, Blue = predicts Not Consolidation")
print("\nNote: Full tree goes much deeper than 4 levels (limited for readability)")
