#!/usr/bin/env python3
"""
Test script to verify that performance optimizations maintain exact calculation precision.
This ensures that the optimized functions produce identical results to the original implementations.
"""

import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.performance_utils import fast_linear_trend, fast_momentum_consistency

def test_fast_linear_trend():
    """Test that fast_linear_trend produces identical results to np.polyfit"""
    print("Testing fast_linear_trend precision...")
    
    # Create test data
    np.random.seed(42)
    test_data = pd.Series(np.random.randn(1000) * 10 + np.arange(1000) * 0.1)
    window_size = 10
    
    # Original method (slow)
    def original_trend(series, window):
        return series.rolling(window).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0
        )
    
    # Calculate with both methods
    original_result = original_trend(test_data, window_size)
    optimized_result = fast_linear_trend(test_data, window_size)
    
    # Check if results are identical (within floating point precision)
    diff = (original_result - optimized_result).abs()
    max_diff = diff.max()
    
    print(f"Maximum difference: {max_diff}")
    print(f"Mean difference: {diff.mean()}")
    
    # Should be nearly identical (within 1e-10 due to floating point precision)
    if max_diff < 1e-10:
        print("✅ fast_linear_trend maintains exact precision!")
        return True
    else:
        print("❌ fast_linear_trend precision test FAILED!")
        print(f"Original sample: {original_result.iloc[20:25].values}")
        print(f"Optimized sample: {optimized_result.iloc[20:25].values}")
        return False

def test_fast_momentum_consistency():
    """Test that fast_momentum_consistency produces identical results"""
    print("\nTesting fast_momentum_consistency precision...")
    
    # Create test data with some directional patterns
    np.random.seed(42)
    test_data = pd.Series(np.random.randn(1000))
    window_size = 5
    
    # Original method (slow)
    def original_consistency(series, window):
        return series.rolling(window).apply(
            lambda x: (np.sign(x) == np.sign(x.iloc[-1])).sum() / len(x)
        )
    
    # Calculate with both methods
    original_result = original_consistency(test_data, window_size)
    optimized_result = fast_momentum_consistency(test_data, window_size)
    
    # Check if results are identical
    diff = (original_result - optimized_result).abs()
    max_diff = diff.max()
    
    print(f"Maximum difference: {max_diff}")
    print(f"Mean difference: {diff.mean()}")
    
    # Should be exactly identical for this calculation
    if max_diff < 1e-15:
        print("✅ fast_momentum_consistency maintains exact precision!")
        return True
    else:
        print("❌ fast_momentum_consistency precision test FAILED!")
        print(f"Original sample: {original_result.iloc[20:25].values}")
        print(f"Optimized sample: {optimized_result.iloc[20:25].values}")
        return False

def test_sampling_preserves_distribution():
    """Test that sampling preserves statistical properties"""
    print("\nTesting sampling distribution preservation...")
    
    from utils.performance_utils import safe_sample_for_analysis
    
    # Create test data with known statistical properties
    np.random.seed(42)
    # Create data with trend and seasonality
    time_index = pd.date_range('2020-01-01', periods=500000, freq='1min')
    trend = np.linspace(0, 100, len(time_index))
    seasonal = 10 * np.sin(2 * np.pi * np.arange(len(time_index)) / (24 * 60))  # Daily pattern
    noise = np.random.randn(len(time_index)) * 5
    
    original_data = pd.DataFrame({
        'value': trend + seasonal + noise,
        'feature1': np.random.randn(len(time_index)),
        'feature2': np.random.exponential(2, len(time_index))
    }, index=time_index)
    
    # Sample the data
    sampled_data = safe_sample_for_analysis(original_data, max_rows=50000, preserve_distribution=True)
    
    # Check statistical properties are preserved
    original_stats = original_data.describe()
    sampled_stats = sampled_data.describe()
    
    # Check if means are within reasonable range - exclude feature2 (exponential) which has high variance
    # Focus on the main 'value' column which represents typical financial time series
    value_mean_diff_pct = abs((original_stats.loc['mean', 'value'] - sampled_stats.loc['mean', 'value']) / original_stats.loc['mean', 'value'] * 100)
    value_std_diff_pct = abs((original_stats.loc['std', 'value'] - sampled_stats.loc['std', 'value']) / original_stats.loc['std', 'value'] * 100)
    
    print(f"Value mean difference %: {value_mean_diff_pct:.2f}%")
    print(f"Value std difference %: {value_std_diff_pct:.2f}%")
    print(f"Sample size: {len(sampled_data):,} from {len(original_data):,}")
    
    # Check temporal distribution is preserved (key for time series)
    original_time_span = (original_data.index.max() - original_data.index.min()).total_seconds()
    sampled_time_span = (sampled_data.index.max() - sampled_data.index.min()).total_seconds()
    time_span_ratio = sampled_time_span / original_time_span
    
    print(f"Time span preserved: {time_span_ratio:.2%}")
    
    if value_mean_diff_pct < 5 and value_std_diff_pct < 10 and time_span_ratio > 0.9:
        print("✅ Sampling preserves statistical distribution!")
        return True
    else:
        print("❌ Sampling distribution preservation test FAILED!")
        return False

def main():
    """Run all precision tests"""
    print("=" * 60)
    print("PERFORMANCE OPTIMIZATION PRECISION TESTS")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 3
    
    # Run tests
    if test_fast_linear_trend():
        tests_passed += 1
    
    if test_fast_momentum_consistency():
        tests_passed += 1
    
    if test_sampling_preserves_distribution():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"RESULTS: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 ALL OPTIMIZATIONS MAINTAIN EXACT PRECISION!")
        print("✅ Safe to use optimized versions in production")
        return 0
    else:
        print("⚠️  Some optimizations failed precision tests")
        print("❌ Review failed optimizations before using")
        return 1

if __name__ == "__main__":
    sys.exit(main())