#!/usr/bin/env python3
"""
Balanced Target Distribution Validation
=====================================

Tests the new percentile-based target generation to validate that it produces
balanced class distribution instead of the extreme 269.5:1 imbalance.

ROOT CAUSE FIX VALIDATION:
- Original: HOLD: 99.22%, BUY: 0.37%, SELL: 0.41% (269.5:1 imbalance)
- Target:   HOLD: 60-80%, BUY: 10-20%, SELL: 10-20% (balanced for ML training)
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from data.features.target_hypothesis import TargetHypothesis
from utils.logging_utils import get_logger

logger = get_logger()

def validate_balanced_target_generation():
    """
    Test the new percentile-based target generation approach.
    """
    print("🚀 VALIDATING ROOT CAUSE FIX: BALANCED TARGET GENERATION")
    print("=" * 80)
    print("Testing percentile-based thresholds vs previous absolute thresholds")
    print("Expected: 10-20% BUY/SELL signals instead of <1%")
    print("=" * 80)
    
    # Use existing processed dataset for testing
    dataset_path = "storage/data/capital_com/GBPUSD/processed/1m/2025-03-02_2025-07-01.csv"
    
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("Please run data processing first with your hypothesis config.")
        return
    
    print(f"📊 Loading dataset: {dataset_path}")
    
    # Load and test target generation
    try:
        target_generator = TargetHypothesis(dataset_path, 'spread_compression_patterns_v3')
        
        print("\n🎯 TESTING NEW PERCENTILE-BASED TARGET GENERATION...")
        result_df = target_generator.run()
        
        if result_df is None or 'target_spread_compression_v3_directional' not in result_df.columns:
            print("❌ Target generation failed or target column not found")
            return
        
        # Analyze new target distribution
        target_column = 'target_spread_compression_v3_directional'
        target_dist = result_df[target_column].value_counts().sort_index()
        total_samples = len(result_df)
        
        print(f"\n📊 NEW BALANCED TARGET DISTRIBUTION RESULTS:")
        print(f"Total samples: {total_samples:,}")
        
        # Calculate percentages
        hold_count = target_dist.get(0, 0)
        buy_count = target_dist.get(1, 0)
        sell_count = target_dist.get(2, 0)
        
        hold_pct = hold_count / total_samples * 100
        buy_pct = buy_count / total_samples * 100
        sell_pct = sell_count / total_samples * 100
        
        print(f"  HOLD (Class 0): {hold_count:,} samples ({hold_pct:.2f}%)")
        print(f"  BUY  (Class 1): {buy_count:,} samples ({buy_pct:.2f}%)")
        print(f"  SELL (Class 2): {sell_count:,} samples ({sell_pct:.2f}%)")
        
        # ROOT CAUSE FIX SUCCESS VALIDATION
        print(f"\n🎯 ROOT CAUSE FIX VALIDATION:")
        signal_pct = buy_pct + sell_pct
        
        print(f"Original extreme imbalance: HOLD: 99.22%, BUY: 0.37%, SELL: 0.41%")
        print(f"New balanced distribution:  HOLD: {hold_pct:.2f}%, BUY: {buy_pct:.2f}%, SELL: {sell_pct:.2f}%")
        
        # Check improvement metrics
        original_buy_pct = 0.37
        original_sell_pct = 0.41
        original_signal_pct = original_buy_pct + original_sell_pct
        
        buy_improvement = buy_pct / original_buy_pct if original_buy_pct > 0 else float('inf')
        sell_improvement = sell_pct / original_sell_pct if original_sell_pct > 0 else float('inf')
        signal_improvement = signal_pct / original_signal_pct if original_signal_pct > 0 else float('inf')
        
        print(f"\n📈 IMPROVEMENT METRICS:")
        print(f"  BUY signals:  {buy_improvement:.1f}x increase ({original_buy_pct:.2f}% → {buy_pct:.2f}%)")
        print(f"  SELL signals: {sell_improvement:.1f}x increase ({original_sell_pct:.2f}% → {sell_pct:.2f}%)")
        print(f"  Total signals: {signal_improvement:.1f}x increase ({original_signal_pct:.2f}% → {signal_pct:.2f}%)")
        
        # Imbalance ratio check
        if buy_count > 0 and sell_count > 0:
            min_class = min(buy_count, sell_count)
            max_class = max(hold_count, buy_count, sell_count)
            new_imbalance_ratio = max_class / min_class
            original_imbalance_ratio = 269.5
            
            print(f"\n⚖️  CLASS BALANCE ANALYSIS:")
            print(f"  Original imbalance ratio: {original_imbalance_ratio:.1f}:1 (SEVERE)")
            print(f"  New imbalance ratio: {new_imbalance_ratio:.1f}:1")
            
            if new_imbalance_ratio < 10:
                print("  ✅ EXCELLENT: Balanced class distribution achieved!")
            elif new_imbalance_ratio < 20:
                print("  ✅ GOOD: Much improved class balance")
            elif new_imbalance_ratio < 50:
                print("  ⚠️  MODERATE: Some improvement but still imbalanced")
            else:
                print("  ❌ POOR: Still severely imbalanced")
            
            balance_improvement = original_imbalance_ratio / new_imbalance_ratio
            print(f"  Balance improvement: {balance_improvement:.1f}x better")
        
        # Directional balance check
        if buy_count > 0 and sell_count > 0:
            directional_balance = min(buy_count, sell_count) / max(buy_count, sell_count)
            print(f"\n📊 DIRECTIONAL BALANCE:")
            print(f"  BUY/SELL balance ratio: {directional_balance:.3f} (1.0 = perfect)")
            
            if directional_balance > 0.8:
                print("  ✅ EXCELLENT: Well-balanced BUY/SELL distribution")
            elif directional_balance > 0.6:
                print("  ✅ GOOD: Reasonable BUY/SELL balance")
            else:
                print(f"  ⚠️  IMBALANCED: {('BUY' if buy_count > sell_count else 'SELL')} signals dominate")
        
        # ML Training Suitability Assessment
        print(f"\n🤖 ML TRAINING SUITABILITY:")
        if signal_pct >= 20 and signal_pct <= 40:
            print(f"  ✅ OPTIMAL: {signal_pct:.1f}% signal rate perfect for ML training")
        elif signal_pct >= 10 and signal_pct < 20:
            print(f"  ✅ GOOD: {signal_pct:.1f}% signal rate suitable for ML training")
        elif signal_pct < 10:
            print(f"  ⚠️  LOW: {signal_pct:.1f}% signal rate may still cause confidence issues")
        else:
            print(f"  ⚠️  HIGH: {signal_pct:.1f}% signal rate may be too aggressive")
        
        # Confidence calibration prediction
        if signal_pct > 5:  # Much higher than original 0.78%
            print("  ✅ CONFIDENCE PREDICTION: Model should have meaningful confidence values")
            print("  ✅ BACKTESTING PREDICTION: Signals should pass confidence thresholds")
        else:
            print("  ⚠️  May still have confidence calibration issues")
        
        print(f"\n🎯 ROOT CAUSE FIX SUMMARY:")
        if signal_pct > 10 and new_imbalance_ratio < 20:
            print("  ✅ SUCCESS: ROOT CAUSE FIXED!")
            print("  ✅ Balanced class distribution achieved")
            print("  ✅ Should resolve 0% win rate and 0.0000 confidence issues")
            print("  ✅ Ready for model retraining and backtesting")
        else:
            print("  ⚠️  PARTIAL SUCCESS: Improved but may need further tuning")
        
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    validate_balanced_target_generation()