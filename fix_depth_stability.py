#!/usr/bin/env python3
"""
Fix for depth_stability calculation issue
========================================

The issue is in the enhanced_generation.py depth_stability calculation:
    depth_stability = 1 / (spread_range.rolling(window=20, min_periods=5).std() + 1e-8)
    self.df['depth_stability'] = np.log1p(depth_stability)

When spread volatility is very low, this creates extremely large values (up to 100M),
which after backtesting return calculations can cause the NaN and extreme negative returns.

This script shows the problematic calculation and provides a fix.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def demonstrate_issue():
    """Demonstrate the depth_stability calculation issue"""
    print("🔍 DEMONSTRATING DEPTH_STABILITY ISSUE")
    print("="*50)
    
    # Create sample data with very low spread volatility
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    
    # Simulate spread data with periods of very low volatility
    base_spread = 0.5
    noise_high = np.random.normal(0, 0.1, 500)  # Normal volatility
    noise_low = np.random.normal(0, 0.001, 500)  # Very low volatility
    noise = np.concatenate([noise_high, noise_low])
    
    spread_data = base_spread + noise
    spread_range = np.abs(np.random.normal(0, 0.05, 1000)) + 0.001  # Small positive spread range
    
    df = pd.DataFrame({
        'spread_close': spread_data,
        'spread_range': spread_range
    }, index=dates)
    
    print(f"Spread range statistics:")
    print(f"  Min: {spread_range.min():.6f}")
    print(f"  Max: {spread_range.max():.6f}")
    print(f"  Mean: {spread_range.mean():.6f}")
    print(f"  Std: {spread_range.std():.6f}")
    
    # Calculate rolling std
    rolling_std = pd.Series(spread_range).rolling(window=20, min_periods=5).std()
    print(f"\nRolling std statistics:")
    print(f"  Min: {rolling_std.min():.8f}")
    print(f"  Max: {rolling_std.max():.8f}")
    print(f"  Values near zero (< 1e-6): {(rolling_std < 1e-6).sum()}")
    
    # Current problematic calculation
    problematic_depth_stability = 1 / (rolling_std + 1e-8)
    final_problematic = np.log1p(problematic_depth_stability)
    
    print(f"\nProblematic calculation results:")
    print(f"  1/(std + 1e-8) min: {problematic_depth_stability.min():.2f}")
    print(f"  1/(std + 1e-8) max: {problematic_depth_stability.max():.2e}")
    print(f"  log1p() min: {final_problematic.min():.2f}")
    print(f"  log1p() max: {final_problematic.max():.2e}")
    print(f"  Extreme values (> 1M): {(final_problematic > 1e6).sum()}")
    
    return df, rolling_std, final_problematic

def show_fixed_calculation(df, rolling_std):
    """Show the fixed calculation"""
    print("\n🔧 FIXED CALCULATION")
    print("="*50)
    
    # Method 1: Cap the reciprocal before log transform
    max_stability = 1000  # Reasonable maximum for stability measure
    capped_depth_stability = np.minimum(1 / (rolling_std + 1e-6), max_stability)
    fixed_method1 = np.log1p(capped_depth_stability)
    
    print(f"Method 1 - Capped reciprocal:")
    print(f"  Before log1p() max: {capped_depth_stability.max():.2f}")
    print(f"  After log1p() max: {fixed_method1.max():.2f}")
    
    # Method 2: Use a more robust stability measure
    # Instead of 1/std, use std^(-0.5) which grows more slowly
    robust_depth_stability = np.power(rolling_std + 1e-6, -0.5)
    fixed_method2 = robust_depth_stability
    
    print(f"\nMethod 2 - Robust power transform:")
    print(f"  std^(-0.5) max: {fixed_method2.max():.2f}")
    print(f"  std^(-0.5) min: {fixed_method2.min():.2f}")
    
    # Method 3: Use log-scale approach
    # log(1 + 1/std) instead of log(1 + 1/std)
    log_scale_stability = np.log1p(1 / (rolling_std + 1e-4))
    
    print(f"\nMethod 3 - Direct log transform:")
    print(f"  log1p(1/std) max: {log_scale_stability.max():.2f}")
    print(f"  log1p(1/std) min: {log_scale_stability.min():.2f}")
    
    return fixed_method1, fixed_method2, log_scale_stability

def create_fix():
    """Create the actual fix for the enhanced_generation.py file"""
    print("\n🛠️  RECOMMENDED FIX")
    print("="*50)
    
    fix_code = '''
    # FIXED depth_stability calculation - Replace the problematic lines in enhanced_generation.py
    
    # Instead of:
    # depth_stability = 1 / (spread_range.rolling(window=20, min_periods=5).std() + 1e-8)
    # self.df['depth_stability'] = np.log1p(depth_stability)
    
    # Use this:
    rolling_std = spread_range.rolling(window=20, min_periods=5).std()
    
    # Method 1: Capped reciprocal approach (recommended)
    max_stability_raw = 1000  # Reasonable cap for stability measure
    depth_stability_raw = np.minimum(1 / (rolling_std + 1e-6), max_stability_raw)
    self.df['depth_stability'] = np.log1p(depth_stability_raw)
    
    # OR Method 2: More robust power transform (alternative)
    # self.df['depth_stability'] = np.power(rolling_std + 1e-6, -0.5)
    
    # OR Method 3: Direct log approach (simpler alternative)
    # self.df['depth_stability'] = np.log1p(1 / (rolling_std + 1e-4))
    
    # Ensure reasonable bounds
    self.df['depth_stability'] = np.clip(self.df['depth_stability'], 0, 10)
    '''
    
    print(fix_code)
    
    return fix_code

def main():
    """Run the demonstration and create fix"""
    # Demonstrate the issue
    df, rolling_std, problematic_values = demonstrate_issue()
    
    # Show fixed calculations
    fixed1, fixed2, fixed3 = show_fixed_calculation(df, rolling_std)
    
    # Create the fix
    fix_code = create_fix()
    
    print("\n📋 SUMMARY")
    print("="*50)
    print("1. The depth_stability calculation creates extreme values when spread volatility is very low")
    print("2. These extreme values (up to 100M) cause issues in downstream calculations")
    print("3. The fix involves capping the reciprocal and ensuring reasonable bounds")
    print("4. Apply the fix to enhanced_generation.py in the enhanced_market_depth_features method")
    
    print("\n🎯 NEXT STEPS")
    print("="*50)
    print("1. Apply the fix to data/features/enhanced_generation.py")
    print("2. Regenerate processed training and test datasets")
    print("3. Retrain the model with the corrected features")
    print("4. Run backtesting again to verify the fix")

if __name__ == "__main__":
    main()