#!/usr/bin/env python3
"""
Target Analysis Diagnostic Tool
==============================

Comprehensive analysis to identify the root cause of 0% win rate in the ML model.
This script validates target label generation logic and alignment with actual price movements.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.logging_utils import get_logger
from data.features.target_hypothesis import TargetHypothesis

logger = get_logger()

def analyze_target_generation_logic():
    """
    PHASE 1: CRITICAL TARGET GENERATION ANALYSIS
    ============================================
    
    Examines the actual target generation logic to identify potential issues:
    1. Target label distribution and balance
    2. Forward-looking bias validation  
    3. Feature threshold analysis
    4. Directional signal validation
    """
    
    print("\n" + "="*80)
    print("🎯 PHASE 1: TARGET GENERATION LOGIC ANALYSIS")
    print("="*80)
    
    # Load the processed training dataset to analyze target generation  
    training_data_path = "storage/data/capital_com/GBPUSD/processed/1m/2024-01-01_2025-03-01.csv"
    
    if not Path(training_data_path).exists():
        print(f"❌ Training dataset not found: {training_data_path}")
        return None
    
    print(f"📊 Loading training dataset: {training_data_path}")
    df = pd.read_csv(training_data_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Check if target already exists or needs generation
    target_cols = [col for col in df.columns if col.startswith('target_spread_compression_v3')]
    
    if not target_cols:
        print("🔧 Target not found, generating target labels...")
        
        # Generate target using the same logic as the model
        target_generator = TargetHypothesis(df, 'spread_compression_patterns_v3')
        df_with_target = target_generator.run()
        target_cols = [col for col in df_with_target.columns if col.startswith('target_spread_compression_v3')]
    else:
        print(f"✅ Found existing targets: {target_cols}")
        df_with_target = df
    
    if not target_cols:
        print("❌ CRITICAL: No target columns generated!")
        return None
    
    # Analyze the main target column used for training
    main_target = 'target_spread_compression_v3_long_score'
    if main_target not in df_with_target.columns:
        print(f"❌ Main target column missing: {main_target}")
        return None
    
    target_data = df_with_target[main_target]
    
    print(f"\n🎯 TARGET ANALYSIS: {main_target}")
    print("-" * 50)
    
    # 1. Basic distribution analysis
    target_dist = target_data.value_counts().sort_index()
    print(f"Target distribution: {target_dist.to_dict()}")
    
    total_samples = len(target_data)
    for score, count in target_dist.items():
        percentage = count / total_samples * 100
        print(f"  Score {score}: {count:,} samples ({percentage:.2f}%)")
    
    # 2. Signal rate analysis
    signal_samples = (target_data > 0).sum()
    signal_rate = signal_samples / total_samples * 100
    print(f"\nSignal Rate: {signal_samples:,}/{total_samples:,} = {signal_rate:.2f}%")
    
    if signal_rate < 5:
        print("⚠️  WARNING: Very low signal rate (<5%) - Model may struggle to learn")
    elif signal_rate > 50:
        print("⚠️  WARNING: Very high signal rate (>50%) - May include noise")
    else:
        print("✅ Signal rate within reasonable range (5-50%)")
    
    # 3. Check for directional targets
    if 'target_spread_compression_v3_directional' in df_with_target.columns:
        directional_target = df_with_target['target_spread_compression_v3_directional']
        dir_dist = directional_target.value_counts().sort_index()
        
        print(f"\n🧭 DIRECTIONAL TARGET ANALYSIS:")
        print(f"Directional distribution: {dir_dist.to_dict()}")
        
        for direction, count in dir_dist.items():
            direction_name = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}.get(direction, f'Unknown({direction})')
            percentage = count / total_samples * 100
            print(f"  {direction_name}: {count:,} samples ({percentage:.2f}%)")
        
        # Check for balance between BUY/SELL
        buy_count = dir_dist.get(1, 0)
        sell_count = dir_dist.get(2, 0)
        
        if buy_count > 0 and sell_count > 0:
            balance_ratio = min(buy_count, sell_count) / max(buy_count, sell_count)
            print(f"\nBUY/SELL Balance Ratio: {balance_ratio:.3f} (1.0 = perfect)")
            
            if balance_ratio < 0.3:
                print("❌ CRITICAL: Severe directional imbalance detected!")
                dominant = "BUY" if buy_count > sell_count else "SELL"
                print(f"   {dominant} signals dominate - this could cause 0% win rate!")
            elif balance_ratio < 0.7:
                print("⚠️  WARNING: Moderate directional imbalance")
            else:
                print("✅ Good directional balance")
        else:
            print("❌ CRITICAL: Missing BUY or SELL signals entirely!")
    
    return df_with_target

def validate_price_movement_alignment(df_with_target):
    """
    PHASE 2: PRICE MOVEMENT VALIDATION
    ==================================
    
    Validates whether target labels actually align with future price movements.
    This is critical for identifying if the 0% win rate is due to incorrect targets.
    """
    
    print("\n" + "="*80)
    print("📈 PHASE 2: PRICE MOVEMENT ALIGNMENT VALIDATION")
    print("="*80)
    
    if df_with_target is None:
        print("❌ No target data available for validation")
        return
    
    # Ensure we have the required columns
    required_cols = ['close', 'target_spread_compression_v3_long_score']
    missing_cols = [col for col in required_cols if col not in df_with_target.columns]
    
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        return
    
    # Calculate actual future price movements
    print("🔍 Calculating actual price movements...")
    
    # Different time horizons for validation
    horizons = [5, 10, 15, 30]  # 5, 10, 15, 30 minute forward returns
    
    for horizon in horizons:
        future_return_col = f'future_return_{horizon}m'
        df_with_target[future_return_col] = df_with_target['close'].pct_change(periods=-horizon) * 100
    
    target_col = 'target_spread_compression_v3_long_score'
    
    print(f"\n📊 VALIDATION ANALYSIS:")
    print("-" * 40)
    
    # Check correlation between targets and actual future returns
    for horizon in horizons:
        future_return_col = f'future_return_{horizon}m'
        
        # Remove NaN values for correlation
        valid_data = df_with_target[[target_col, future_return_col]].dropna()
        
        if len(valid_data) < 100:
            print(f"⚠️  {horizon}m: Insufficient data for analysis")
            continue
        
        correlation = valid_data[target_col].corr(valid_data[future_return_col])
        print(f"{horizon}m correlation: {correlation:.4f}")
        
        # Analyze performance by target score
        print(f"\n{horizon}m Performance by Target Score:")
        
        for score in sorted(valid_data[target_col].unique()):
            if score == 0:
                continue  # Skip HOLD signals
            
            score_data = valid_data[valid_data[target_col] == score]
            
            if len(score_data) < 10:
                continue
            
            avg_return = score_data[future_return_col].mean()
            win_rate = (score_data[future_return_col] > 0).mean() * 100
            
            print(f"  Score {score}: {len(score_data):,} samples, "
                  f"Avg Return: {avg_return:.4f}%, Win Rate: {win_rate:.1f}%")
            
            if score > 0 and avg_return < 0:
                print(f"    ❌ PROBLEM: Positive target score but negative average return!")
            elif score > 0 and win_rate < 50:
                print(f"    ⚠️  WARNING: Win rate below 50% for positive signal")
    
    # Special analysis for directional targets if available
    if 'target_spread_compression_v3_directional' in df_with_target.columns:
        print(f"\n🧭 DIRECTIONAL TARGET VALIDATION:")
        print("-" * 40)
        
        dir_col = 'target_spread_compression_v3_directional'
        
        for horizon in [10, 15]:  # Focus on key horizons  
            future_return_col = f'future_return_{horizon}m'
            valid_data = df_with_target[[dir_col, future_return_col]].dropna()
            
            print(f"\n{horizon}m Directional Performance:")
            
            for direction in [1, 2]:  # BUY=1, SELL=2
                direction_name = {1: 'BUY', 2: 'SELL'}[direction]
                dir_data = valid_data[valid_data[dir_col] == direction]
                
                if len(dir_data) < 10:
                    print(f"  {direction_name}: Insufficient data ({len(dir_data)} samples)")
                    continue
                
                avg_return = dir_data[future_return_col].mean()
                win_rate = (dir_data[future_return_col] > 0).mean() * 100
                
                # For SELL signals, we want negative returns to be "wins"
                if direction == 2:  # SELL
                    correct_direction_rate = (dir_data[future_return_col] < 0).mean() * 100
                    print(f"  {direction_name}: {len(dir_data):,} signals, "
                          f"Avg Return: {avg_return:.4f}%, Correct Direction: {correct_direction_rate:.1f}%")
                    
                    if avg_return > 0:
                        print(f"    ❌ CRITICAL: SELL signals showing positive returns!")
                else:  # BUY
                    correct_direction_rate = (dir_data[future_return_col] > 0).mean() * 100
                    print(f"  {direction_name}: {len(dir_data):,} signals, "
                          f"Avg Return: {avg_return:.4f}%, Correct Direction: {correct_direction_rate:.1f}%")
                    
                    if avg_return < 0:
                        print(f"    ❌ CRITICAL: BUY signals showing negative returns!")
                
                if correct_direction_rate < 40:
                    print(f"    ❌ MAJOR ISSUE: Directional accuracy too low ({correct_direction_rate:.1f}%)")

def analyze_feature_target_relationships(df_with_target):
    """
    PHASE 3: FEATURE-TARGET CORRELATION ANALYSIS
    ============================================
    
    Analyzes the relationship between individual features and target labels
    to identify if features actually contain predictive power.
    """
    
    print("\n" + "="*80)
    print("🔗 PHASE 3: FEATURE-TARGET CORRELATION ANALYSIS")
    print("="*80)
    
    if df_with_target is None:
        print("❌ No data available for feature analysis")
        return
    
    # Key features to analyze (based on your model's top features)
    key_features = [
        'volume_ratio',
        'top_predictor_interaction',
        'microstructure_momentum_fusion',
        'institutional_footprint',
        'depth_stability'
    ]
    
    target_col = 'target_spread_compression_v3_long_score'
    
    # Check which features are available
    available_features = [f for f in key_features if f in df_with_target.columns]
    missing_features = [f for f in key_features if f not in df_with_target.columns]
    
    print(f"Available features: {available_features}")
    if missing_features:
        print(f"Missing features: {missing_features}")
    
    if not available_features:
        print("❌ No key features available for analysis")
        return
    
    print(f"\n📊 FEATURE-TARGET CORRELATIONS:")
    print("-" * 50)
    
    for feature in available_features:
        try:
            # Calculate correlation with target
            valid_data = df_with_target[[feature, target_col]].dropna()
            
            if len(valid_data) < 100:
                print(f"{feature}: Insufficient data")
                continue
            
            correlation = valid_data[feature].corr(valid_data[target_col])
            
            # Analyze feature distribution in different target classes
            target_0 = valid_data[valid_data[target_col] == 0][feature]
            target_pos = valid_data[valid_data[target_col] > 0][feature]
            
            if len(target_0) > 0 and len(target_pos) > 0:
                mean_diff = target_pos.mean() - target_0.mean()
                
                print(f"{feature}:")
                print(f"  Correlation: {correlation:.4f}")
                print(f"  Mean(target=0): {target_0.mean():.4f}")
                print(f"  Mean(target>0): {target_pos.mean():.4f}")
                print(f"  Difference: {mean_diff:.4f}")
                
                if abs(correlation) < 0.01:
                    print(f"    ❌ PROBLEM: No correlation with target!")
                elif abs(mean_diff) < 0.01:
                    print(f"    ❌ PROBLEM: No difference between signal/no-signal!")
                else:
                    print(f"    ✅ Feature shows predictive potential")
        except Exception as e:
            print(f"{feature}: Error - {e}")

def run_comprehensive_analysis():
    """
    Execute the complete target analysis to identify the root cause of 0% win rate
    """
    
    print("🚀 STARTING COMPREHENSIVE TARGET ANALYSIS")
    print("="*80)
    print("This analysis will identify why the model has 0% win rate despite")
    print("fixing the data corruption issues.")
    print("="*80)
    
    # Phase 1: Target generation logic
    df_with_target = analyze_target_generation_logic()
    
    # Phase 2: Price movement alignment  
    validate_price_movement_alignment(df_with_target)
    
    # Phase 3: Feature-target relationships
    analyze_feature_target_relationships(df_with_target)
    
    print("\n" + "="*80)
    print("🎯 ANALYSIS COMPLETE")
    print("="*80)
    print("Review the findings above to identify the root cause of 0% win rate.")
    print("Key areas to focus on:")
    print("1. Target label distribution and balance")
    print("2. Alignment between targets and actual price movements")
    print("3. Feature-target correlations")
    print("="*80)

if __name__ == "__main__":
    run_comprehensive_analysis()