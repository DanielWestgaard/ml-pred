"""
Quick Target Validation - Fast version for immediate insights
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from data.processing.cleaning import DataCleaner
from data.processing.validation import DataValidator
from data.features.enhanced_generation import ModelAwareFeatureGenerator
from data.features.target_hypothesis import TargetHypothesis
from utils import data_utils
from utils.logging_utils import get_logger

logger = get_logger()

def quick_validate_target(dataset_path: str, hypothesis: str = 'spread_compression_patterns_v3'):
    """
    Quick validation focusing on key metrics only
    """
    print(f"=== QUICK TARGET VALIDATION: {hypothesis} ===\n")
    
    # Load and clean data
    print("Loading and cleaning data...")
    base_df, _ = data_utils.check_and_return_df(dataset_path)
    
    cleaner = DataCleaner(base_df, save_cleaned_data=False)
    cleaned_df = cleaner.run()
    
    validator = DataValidator(cleaned_df)
    validator_results = validator.run()
    df = validator_results["validated_data"]
    
    print(f"Dataset: {len(df):,} rows after cleaning")
    
    # Generate minimal features needed for target
    print("Generating key features...")
    feature_generator = ModelAwareFeatureGenerator(df)
    
    # Generate in dependency order
    feature_generator.volume_ma_20_feature()
    feature_generator.volume_ratio_feature()
    feature_generator.enhanced_market_depth_features()
    feature_generator.institutional_footprint_detection()
    feature_generator.enhanced_microstructure_health()
    feature_generator.enhanced_predictive_signals()
    feature_generator.fx_trading_sessions()
    
    # Generate target
    print(f"Generating target: {hypothesis}")
    target_generator = TargetHypothesis(feature_generator.df, hypothesis)
    final_df = target_generator.run()
    
    # Analyze target distribution
    target_columns = [col for col in final_df.columns if col.startswith('target_')]
    
    print(f"\n=== TARGET DISTRIBUTION ANALYSIS ===")
    for target_col in target_columns:
        target_series = final_df[target_col]
        distribution = target_series.value_counts().sort_index()
        total = len(target_series)
        
        print(f"\n{target_col}:")
        for class_val, count in distribution.items():
            pct = count / total * 100
            print(f"  Class {class_val}: {count:,} samples ({pct:.2f}%)")
        
        # Key metrics
        if target_col.endswith('_directional'):
            buy_pct = distribution.get(1, 0) / total * 100
            sell_pct = distribution.get(2, 0) / total * 100
            signal_rate = buy_pct + sell_pct
            
            print(f"  📊 Signal Rate: {signal_rate:.2f}% (Target: >15%)")
            if signal_rate > 15:
                print(f"  ✅ SUCCESS: Signal rate achieved!")
            else:
                print(f"  ❌ BELOW TARGET: Need {15-signal_rate:.1f}% more signals")
                
            if buy_pct > 0 and sell_pct > 0:
                balance = min(buy_pct, sell_pct) / max(buy_pct, sell_pct)
                print(f"  ⚖️  Directional Balance: {balance:.3f} (1.0 = perfect)")
                if balance > 0.8:
                    print(f"  ✅ EXCELLENT: Well-balanced signals")
                elif balance > 0.6:
                    print(f"  ✅ GOOD: Reasonably balanced")
                else:
                    print(f"  ⚠️  IMBALANCED: Consider rebalancing")
    
    # Quick feature correlation (top 10 features only)
    print(f"\n=== FEATURE CORRELATION (Quick Analysis) ===")
    main_target = [col for col in target_columns if col.endswith('_directional')][0]
    
    # Get numeric features only
    numeric_features = final_df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_features if not col.startswith('target_')]
    
    if len(feature_cols) > 0:
        correlations = {}
        target_values = final_df[main_target]
        
        for feature in feature_cols:
            try:
                corr = abs(final_df[feature].corr(target_values))
                if not pd.isna(corr):
                    correlations[feature] = corr
            except:
                pass
        
        # Sort and show top 10
        sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:10]
        
        print(f"\nTop 10 Correlated Features with {main_target}:")
        for i, (feature, corr) in enumerate(sorted_corr, 1):
            print(f"  {i:2d}. {feature:<30} {corr:.4f}")
    
    # Quick profitability check (5-period forward returns)
    print(f"\n=== QUICK PROFITABILITY CHECK ===")
    if 'close' in final_df.columns:
        # Calculate 5-period forward returns
        final_df['forward_return_5'] = (final_df['close'].shift(-5) / final_df['close'] - 1) * 100
        
        target_series = final_df[main_target]
        returns_col = 'forward_return_5'
        
        print(f"5-period forward returns by signal class:")
        for class_val in sorted(target_series.unique()):
            mask = target_series == class_val
            class_returns = final_df.loc[mask, returns_col].dropna()
            
            if len(class_returns) > 10:  # Need sufficient samples
                mean_return = class_returns.mean()
                win_rate = (class_returns > 0).mean() * 100
                sample_count = len(class_returns)
                
                class_name = {0: "HOLD", 1: "BUY", 2: "SELL"}.get(class_val, f"Class {class_val}")
                print(f"  {class_name:<8}: {mean_return:+.4f}% avg return, {win_rate:.1f}% win rate ({sample_count:,} samples)")
                
                if class_val > 0:  # Signal classes
                    if mean_return > 0 and win_rate > 50:
                        print(f"            ✅ PROFITABLE: Good signal quality")
                    elif mean_return > 0:
                        print(f"            ⚠️  MARGINAL: Positive but low win rate")
                    else:
                        print(f"            ❌ UNPROFITABLE: Negative average return")
    
    print(f"\n=== SUMMARY ===")
    print(f"Dataset: {len(final_df):,} rows")
    print(f"Target Hypothesis: {hypothesis}")
    print(f"Generated Targets: {len(target_columns)}")
    if len(sorted_corr) > 0:
        print(f"Best Feature: {sorted_corr[0][0]} (correlation: {sorted_corr[0][1]:.4f})")
    
    return final_df, target_columns

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        hypothesis = sys.argv[2] if len(sys.argv) > 2 else 'spread_compression_patterns_v3'
        
        result_df, targets = quick_validate_target(data_path, hypothesis)
        print(f"\n✅ Quick validation completed!")
        print(f"Generated {len(targets)} target columns: {targets}")
    else:
        print("Usage: python quick_target_validation.py <data_path> [hypothesis_name]")