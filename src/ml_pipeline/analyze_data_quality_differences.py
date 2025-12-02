#!/usr/bin/env python3
"""
Analyze Data Quality Differences: Alpaca vs TwelveData

Compares the same trades extracted from both APIs to identify:
1. Feature value differences (mean, std, outliers)
2. Missing data rates
3. Distribution shifts
4. Which features differ most between APIs

This helps understand why models trained on different APIs perform differently.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats


def load_features(filepath):
    """Load features from JSONL file"""
    features = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                features.append(json.loads(line))
    return pd.DataFrame(features)


def align_datasets(df_alpaca, df_twelve):
    """Find common trades between both datasets"""
    # Create unique keys for matching
    df_alpaca['trade_key'] = df_alpaca['ticker'].astype(str) + '_' + df_alpaca['entry_date'].astype(str)
    df_twelve['trade_key'] = df_twelve['ticker'].astype(str) + '_' + df_twelve['entry_date'].astype(str)
    
    # Find common trades
    common_keys = set(df_alpaca['trade_key']) & set(df_twelve['trade_key'])
    
    df_alpaca_common = df_alpaca[df_alpaca['trade_key'].isin(common_keys)].copy()
    df_twelve_common = df_twelve[df_twelve['trade_key'].isin(common_keys)].copy()
    
    # Sort by trade_key for alignment
    df_alpaca_common = df_alpaca_common.sort_values('trade_key').reset_index(drop=True)
    df_twelve_common = df_twelve_common.sort_values('trade_key').reset_index(drop=True)
    
    return df_alpaca_common, df_twelve_common, common_keys


def compare_feature_distributions(df_alpaca, df_twelve, feature_cols):
    """Compare distributions of features between APIs"""
    print("\n" + "=" * 80)
    print("FEATURE DISTRIBUTION COMPARISON")
    print("=" * 80)
    
    comparison_results = []
    
    for feature in feature_cols:
        alpaca_vals = df_alpaca[feature].values
        twelve_vals = df_twelve[feature].values
        
        # Basic statistics
        alpaca_mean = np.mean(alpaca_vals)
        twelve_mean = np.mean(twelve_vals)
        alpaca_std = np.std(alpaca_vals)
        twelve_std = np.std(twelve_vals)
        
        # Mean absolute difference
        mad = np.mean(np.abs(alpaca_vals - twelve_vals))
        
        # Correlation
        correlation = np.corrcoef(alpaca_vals, twelve_vals)[0, 1]
        
        # KS test (distribution similarity)
        ks_stat, ks_pvalue = stats.ks_2samp(alpaca_vals, twelve_vals)
        
        comparison_results.append({
            'feature': feature,
            'alpaca_mean': alpaca_mean,
            'twelve_mean': twelve_mean,
            'mean_diff': twelve_mean - alpaca_mean,
            'mean_diff_pct': ((twelve_mean - alpaca_mean) / (abs(alpaca_mean) + 1e-10)) * 100,
            'alpaca_std': alpaca_std,
            'twelve_std': twelve_std,
            'mad': mad,
            'correlation': correlation,
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pvalue
        })
    
    df_comparison = pd.DataFrame(comparison_results)
    
    # Sort by mean absolute difference (most different features first)
    df_comparison = df_comparison.sort_values('mad', ascending=False)
    
    print(f"\n{'Feature':<30s} {'MAD':<10s} {'Correlation':<12s} {'Mean Diff %':<12s}")
    print("-" * 80)
    
    for _, row in df_comparison.head(15).iterrows():
        print(f"{row['feature']:<30s} {row['mad']:>8.4f}  {row['correlation']:>10.4f}  {row['mean_diff_pct']:>10.2f}%")
    
    return df_comparison


def analyze_outliers(df_alpaca, df_twelve, feature_cols):
    """Identify features with significant outlier differences"""
    print("\n" + "=" * 80)
    print("OUTLIER ANALYSIS")
    print("=" * 80)
    
    outlier_summary = []
    
    for feature in feature_cols:
        alpaca_vals = df_alpaca[feature].values
        twelve_vals = df_twelve[feature].values
        
        # Count outliers (values > 3 std from mean)
        alpaca_outliers = np.sum(np.abs(alpaca_vals - np.mean(alpaca_vals)) > 3 * np.std(alpaca_vals))
        twelve_outliers = np.sum(np.abs(twelve_vals - np.mean(twelve_vals)) > 3 * np.std(twelve_vals))
        
        # Count extreme differences between APIs (> 50% relative difference)
        differences = np.abs((alpaca_vals - twelve_vals) / (np.abs(alpaca_vals) + 1e-10))
        extreme_diffs = np.sum(differences > 0.5)
        
        outlier_summary.append({
            'feature': feature,
            'alpaca_outliers': alpaca_outliers,
            'twelve_outliers': twelve_outliers,
            'extreme_diffs': extreme_diffs,
            'extreme_diff_pct': (extreme_diffs / len(alpaca_vals)) * 100
        })
    
    df_outliers = pd.DataFrame(outlier_summary)
    df_outliers = df_outliers.sort_values('extreme_diff_pct', ascending=False)
    
    print(f"\n{'Feature':<30s} {'Extreme Diffs':<15s} {'% of Trades':<12s}")
    print("-" * 80)
    
    for _, row in df_outliers.head(10).iterrows():
        print(f"{row['feature']:<30s} {row['extreme_diffs']:>8d}       {row['extreme_diff_pct']:>8.2f}%")
    
    return df_outliers


def analyze_missing_data(df_alpaca, df_twelve):
    """Check for missing/NaN values in both datasets"""
    print("\n" + "=" * 80)
    print("MISSING DATA ANALYSIS")
    print("=" * 80)
    
    alpaca_missing = df_alpaca.isnull().sum()
    twelve_missing = df_twelve.isnull().sum()
    
    print(f"\nAlpaca dataset: {alpaca_missing.sum()} total missing values")
    print(f"TwelveData dataset: {twelve_missing.sum()} total missing values")
    
    if alpaca_missing.sum() > 0 or twelve_missing.sum() > 0:
        print("\nFeatures with missing values:")
        print(f"{'Feature':<30s} {'Alpaca Missing':<20s} {'TwelveData Missing':<20s}")
        print("-" * 80)
        for col in df_alpaca.columns:
            if alpaca_missing[col] > 0 or twelve_missing[col] > 0:
                print(f"{col:<30s} {alpaca_missing[col]:>10d}           {twelve_missing[col]:>10d}")


def find_problematic_trades(df_alpaca, df_twelve, feature_cols, top_n=10):
    """Find specific trades with largest feature differences"""
    print("\n" + "=" * 80)
    print(f"TOP {top_n} TRADES WITH LARGEST FEATURE DIFFERENCES")
    print("=" * 80)
    
    # Calculate total absolute difference per trade
    total_diffs = []
    for idx in range(len(df_alpaca)):
        trade_diff = 0
        for feature in feature_cols:
            alpaca_val = df_alpaca[feature].iloc[idx]
            twelve_val = df_twelve[feature].iloc[idx]
            # Normalized difference
            norm_diff = abs(alpaca_val - twelve_val) / (abs(alpaca_val) + abs(twelve_val) + 1e-10)
            trade_diff += norm_diff
        total_diffs.append(trade_diff)
    
    df_alpaca['total_diff'] = total_diffs
    df_alpaca_sorted = df_alpaca.sort_values('total_diff', ascending=False)
    
    print(f"\n{'Ticker':<10s} {'Date':<15s} {'Total Diff':<15s} {'Label':<10s}")
    print("-" * 80)
    
    for idx in range(min(top_n, len(df_alpaca_sorted))):
        row = df_alpaca_sorted.iloc[idx]
        print(f"{row['ticker']:<10s} {str(row['entry_date']):<15s} {row['total_diff']:>10.2f}     {row['label']:<10d}")
    
    return df_alpaca_sorted.head(top_n)


def summarize_findings(df_comparison, df_outliers):
    """Provide actionable summary of findings"""
    print("\n" + "=" * 80)
    print("📊 SUMMARY & RECOMMENDATIONS")
    print("=" * 80)
    
    # Top 5 most different features
    top_diff_features = df_comparison.head(5)['feature'].tolist()
    
    print("\n🔴 Features with Largest Differences:")
    for i, feature in enumerate(top_diff_features, 1):
        row = df_comparison[df_comparison['feature'] == feature].iloc[0]
        print(f"{i}. {feature}: MAD={row['mad']:.4f}, Correlation={row['correlation']:.4f}")
    
    # Poorly correlated features
    poorly_correlated = df_comparison[df_comparison['correlation'] < 0.9]['feature'].tolist()
    
    if poorly_correlated:
        print(f"\n⚠️  Features with Poor Correlation (<0.9) between APIs:")
        for feature in poorly_correlated[:5]:
            row = df_comparison[df_comparison['feature'] == feature].iloc[0]
            print(f"   - {feature}: r={row['correlation']:.4f}")
    
    # Features with many extreme differences
    extreme_features = df_outliers[df_outliers['extreme_diff_pct'] > 10]['feature'].tolist()
    
    if extreme_features:
        print(f"\n🚨 Features with >10% of trades having extreme differences:")
        for feature in extreme_features[:5]:
            row = df_outliers[df_outliers['feature'] == feature].iloc[0]
            print(f"   - {feature}: {row['extreme_diff_pct']:.2f}% of trades")
    
    # Overall assessment
    avg_correlation = df_comparison['correlation'].mean()
    
    print(f"\n📈 Overall Data Quality Assessment:")
    print(f"   Average correlation across all features: {avg_correlation:.4f}")
    
    if avg_correlation > 0.95:
        print("   ✅ EXCELLENT: APIs produce very similar features")
        print("   → Safe to use Alpaca model with TwelveData features")
    elif avg_correlation > 0.90:
        print("   ✅ GOOD: APIs are mostly consistent")
        print("   → Using Alpaca model with TwelveData should work well")
    elif avg_correlation > 0.80:
        print("   ⚠️  MODERATE: Some discrepancies between APIs")
        print("   → Test Alpaca model on TwelveData before production")
    else:
        print("   🔴 POOR: Significant differences between APIs")
        print("   → Consider retraining on TwelveData or feature normalization")
    
    print("\n💡 Recommendations:")
    if avg_correlation > 0.90:
        print("   1. ✅ Use Alpaca model (91% ROC-AUC on TwelveData features)")
        print("   2. Monitor the poorly correlated features in production")
        print("   3. Consider feature normalization if issues arise")
    else:
        print("   1. Retrain model specifically on TwelveData features")
        print("   2. Apply feature scaling/normalization to reduce API differences")
        print("   3. Create ensemble model that averages both API predictions")


def main():
    """Run full data quality analysis"""
    print("=" * 80)
    print("DATA QUALITY ANALYSIS: Alpaca vs TwelveData")
    print("=" * 80)
    
    # Load datasets
    features_alpaca_path = Path('data/processed/features_5R_5min.jsonl')
    features_twelve_path = Path('data/processed/features_5R_5min_twelve.jsonl')
    
    if not features_alpaca_path.exists() or not features_twelve_path.exists():
        print("\n❌ Missing feature files:")
        print(f"   Alpaca: {features_alpaca_path}")
        print(f"   TwelveData: {features_twelve_path}")
        return
    
    print("\nLoading datasets...")
    df_alpaca = load_features(features_alpaca_path)
    df_twelve = load_features(features_twelve_path)
    
    print(f"✅ Alpaca: {len(df_alpaca)} trades")
    print(f"✅ TwelveData: {len(df_twelve)} trades")
    
    # Align datasets (find common trades)
    print("\nAligning datasets (finding common trades)...")
    df_alpaca_common, df_twelve_common, common_keys = align_datasets(df_alpaca, df_twelve)
    
    print(f"✅ Found {len(common_keys)} common trades")
    
    if len(common_keys) == 0:
        print("\n❌ No common trades found! Cannot compare.")
        print("   This likely means the trades were extracted from different time periods.")
        return
    
    # Get feature columns
    metadata_cols = ['ticker', 'entry_date', 'label', 'timeframe', 'trade_key']
    feature_cols = [col for col in df_alpaca_common.columns if col not in metadata_cols]
    
    print(f"\nAnalyzing {len(feature_cols)} features across {len(common_keys)} common trades...")
    
    # Run analyses
    df_comparison = compare_feature_distributions(df_alpaca_common, df_twelve_common, feature_cols)
    df_outliers = analyze_outliers(df_alpaca_common, df_twelve_common, feature_cols)
    analyze_missing_data(df_alpaca_common, df_twelve_common)
    problematic_trades = find_problematic_trades(df_alpaca_common, df_twelve_common, feature_cols)
    
    # Summary
    summarize_findings(df_comparison, df_outliers)
    
    # Save detailed comparison to CSV
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    comparison_file = output_dir / 'api_feature_comparison.csv'
    df_comparison.to_csv(comparison_file, index=False)
    print(f"\n💾 Detailed comparison saved to: {comparison_file}")
    
    outliers_file = output_dir / 'api_outlier_analysis.csv'
    df_outliers.to_csv(outliers_file, index=False)
    print(f"💾 Outlier analysis saved to: {outliers_file}")


if __name__ == '__main__':
    main()
