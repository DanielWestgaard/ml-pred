#!/usr/bin/env python3
"""
Test Improved Target Generation
==============================

Tests the improved BUY signal logic with stronger thresholds to validate
if the fix improves directional accuracy.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.logging_utils import get_logger
from data.features.target_hypothesis import TargetHypothesis

logger = get_logger()

def test_improved_target_generation():
    """
    Test the improved target generation logic on a subset of data
    """
    
    print("🧪 TESTING IMPROVED TARGET GENERATION")
    print("="*60)
    
    # Load a subset of data for quick testing
    data_path = "storage/data/capital_com/GBPUSD/processed/1m/2024-01-01_2025-03-01.csv"
    
    if not Path(data_path).exists():
        print(f"❌ Dataset not found: {data_path}")
        return
    
    print(f"📊 Loading dataset: {data_path}")
    df = pd.read_csv(data_path)
    
    # Take a manageable subset for testing (last 50k rows)
    df_test = df.tail(50000).copy()
    print(f"Testing on {len(df_test):,} samples")
    
    # Remove existing targets to force regeneration
    target_cols = [col for col in df_test.columns if col.startswith('target_spread_compression_v3')]
    if target_cols:
        print(f"Removing existing targets: {target_cols}")
        df_test = df_test.drop(columns=target_cols)
    
    # Generate new targets with improved logic
    print("\n🔧 Generating targets with IMPROVED BUY signal logic...")
    target_generator = TargetHypothesis(df_test, 'spread_compression_patterns_v3')
    df_with_new_targets = target_generator.run()
    
    # Analyze the new directional target
    if 'target_spread_compression_v3_directional' not in df_with_new_targets.columns:
        print("❌ Failed to generate directional target")
        return
    
    directional_target = df_with_new_targets['target_spread_compression_v3_directional']
    
    # Calculate future returns for validation
    print("\n📈 Calculating future price movements...")
    for horizon in [10, 15]:
        future_return_col = f'future_return_{horizon}m'
        df_with_new_targets[future_return_col] = df_with_new_targets['close'].pct_change(periods=-horizon) * 100
    
    # Analyze NEW vs OLD performance
    print(f"\n🎯 NEW TARGET PERFORMANCE ANALYSIS")
    print("-" * 50)
    
    dir_dist = directional_target.value_counts().sort_index()
    total_samples = len(directional_target)
    
    print(f"Directional distribution: {dir_dist.to_dict()}")
    
    for direction in [1, 2]:  # BUY=1, SELL=2
        direction_name = {1: 'BUY', 2: 'SELL'}[direction]
        dir_data = df_with_new_targets[df_with_new_targets['target_spread_compression_v3_directional'] == direction]
        
        if len(dir_data) < 10:
            print(f"  {direction_name}: Insufficient data ({len(dir_data)} samples)")
            continue
        
        print(f"\n{direction_name} SIGNALS ({len(dir_data):,} samples):")
        
        for horizon in [10, 15]:
            future_return_col = f'future_return_{horizon}m'
            valid_data = dir_data[[future_return_col]].dropna()
            
            if len(valid_data) < 5:
                continue
            
            avg_return = valid_data[future_return_col].mean()
            
            if direction == 1:  # BUY signals
                correct_direction_rate = (valid_data[future_return_col] > 0).mean() * 100
                print(f"  {horizon}m: Avg Return: {avg_return:.4f}%, Accuracy: {correct_direction_rate:.1f}%")
                
                if correct_direction_rate > 45:
                    print(f"    ✅ IMPROVED: BUY accuracy {correct_direction_rate:.1f}% (vs 35.6% old)")
                else:
                    print(f"    ⚠️  Still low: BUY accuracy {correct_direction_rate:.1f}%")
                    
            else:  # SELL signals
                correct_direction_rate = (valid_data[future_return_col] < 0).mean() * 100
                print(f"  {horizon}m: Avg Return: {avg_return:.4f}%, Accuracy: {correct_direction_rate:.1f}%")
    
    # Compare signal counts
    buy_count = (directional_target == 1).sum()
    sell_count = (directional_target == 2).sum()
    hold_count = (directional_target == 0).sum()
    
    buy_pct = buy_count / total_samples * 100
    sell_pct = sell_count / total_samples * 100
    hold_pct = hold_count / total_samples * 100
    
    print(f"\n📊 SIGNAL DISTRIBUTION ANALYSIS:")
    print(f"  HOLD: {hold_count:,} ({hold_pct:.1f}%)")
    print(f"  BUY:  {buy_count:,} ({buy_pct:.1f}%)")  
    print(f"  SELL: {sell_count:,} ({sell_pct:.1f}%)")
    
    if buy_count > 0 and sell_count > 0:
        balance_ratio = min(buy_count, sell_count) / max(buy_count, sell_count)
        print(f"  Balance ratio: {balance_ratio:.3f}")
        
        if buy_pct < 3:
            print(f"⚠️  Very few BUY signals ({buy_pct:.1f}%) - Thresholds may be too strict")
        elif buy_pct > 15:
            print(f"⚠️  Too many BUY signals ({buy_pct:.1f}%) - May include noise")
        else:
            print(f"✅ Reasonable BUY signal rate: {buy_pct:.1f}%")

if __name__ == "__main__":
    test_improved_target_generation()