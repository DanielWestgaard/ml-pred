#!/usr/bin/env python3
"""
Data Pipeline Diagnostic Tool
============================

This script systematically tests the data pipeline to identify the root cause of 
NaN returns and extreme negative values in backtesting results.

Based on the analysis, the data looks clean, so the issue likely lies in:
1. Return calculation logic during backtesting
2. Price data inconsistencies between entry and exit
3. Division by zero in return calculations
4. Time alignment issues between signals and price data
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.logging_utils import get_logger

logger = get_logger()

class DataPipelineDiagnostic:
    def __init__(self):
        """Initialize diagnostic with processed datasets"""
        self.train_path = "storage/data/capital_com/GBPUSD/processed/1m/2024-01-01_2025-03-01.csv"
        self.test_path = "storage/data/capital_com/GBPUSD/processed/1m/2025-03-02_2025-07-01.csv"
        
        # Load datasets 
        print("Loading processed datasets...")
        self.train_df = pd.read_csv(self.train_path, index_col='date', parse_dates=True)
        self.test_df = pd.read_csv(self.test_path, index_col='date', parse_dates=True)
        
        print(f"Train dataset: {self.train_df.shape}")
        print(f"Test dataset: {self.test_df.shape}")
    
    def run_comprehensive_diagnostic(self):
        """Run all diagnostic tests"""
        print("\n" + "="*60)
        print("DATA PIPELINE DIAGNOSTIC ANALYSIS")
        print("="*60)
        
        # Test 1: Basic data integrity
        self.test_basic_data_integrity()
        
        # Test 2: Price consistency and relationships
        self.test_price_consistency()
        
        # Test 3: Signal generation simulation
        self.test_signal_generation()
        
        # Test 4: Return calculation simulation
        self.test_return_calculations()
        
        # Test 5: Edge cases that cause NaN
        self.test_edge_cases()
        
        # Test 6: Feature generation validation
        self.test_feature_values()
        
        print("\n" + "="*60)
        print("DIAGNOSTIC COMPLETE")
        print("="*60)
    
    def test_basic_data_integrity(self):
        """Test basic data integrity"""
        print("\n🔍 Test 1: Basic Data Integrity")
        print("-" * 40)
        
        for name, df in [("Train", self.train_df), ("Test", self.test_df)]:
            print(f"\n{name} Dataset:")
            
            # Check for NaN/inf values
            nan_count = df.isnull().sum().sum()
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            inf_count = np.isinf(df[numeric_cols]).sum().sum()
            
            print(f"  ✓ NaN values: {nan_count}")
            print(f"  ✓ Inf values: {inf_count}")
            
            # Check close_og (critical for return calculations)
            if 'close_og' in df.columns:
                close_stats = df['close_og'].describe()
                print(f"  ✓ close_og min: {close_stats['min']:.5f}")
                print(f"  ✓ close_og max: {close_stats['max']:.5f}")
                print(f"  ✓ close_og unique: {df['close_og'].nunique()}")
                
                # Check for zero or negative prices (would cause NaN in returns)
                zero_or_neg = (df['close_og'] <= 0).sum()
                print(f"  ⚠️  close_og <= 0: {zero_or_neg}")
                
            # Check index continuity
            if len(df) > 1:
                time_diffs = df.index.to_series().diff().dropna()
                mode_diff = time_diffs.mode().iloc[0] if not time_diffs.mode().empty else time_diffs.median()
                gaps = (time_diffs > mode_diff * 2).sum()
                print(f"  ✓ Time gaps (>2x normal): {gaps}")
    
    def test_price_consistency(self):
        """Test price data consistency between normalized and original"""
        print("\n🔍 Test 2: Price Consistency")
        print("-" * 40)
        
        for name, df in [("Train", self.train_df), ("Test", self.test_df)]:
            print(f"\n{name} Dataset:")
            
            if 'close_og' in df.columns and 'close' in df.columns:
                # Check if normalized close makes sense relative to close_og
                close_norm_min = df['close'].min()
                close_norm_max = df['close'].max()
                print(f"  ✓ Normalized close range: [{close_norm_min:.4f}, {close_norm_max:.4f}]")
                
                # Check correlation between close and close_og (should be high)
                correlation = df['close'].corr(df['close_og'])
                print(f"  ✓ close vs close_og correlation: {correlation:.6f}")
                
                # Check for constant price periods (could cause division by zero)
                price_changes = df['close_og'].diff()
                zero_changes = (price_changes == 0).sum()
                print(f"  ⚠️  Periods with no price change: {zero_changes}")
                
                # Check price change distribution
                non_zero_changes = price_changes[price_changes != 0]
                if len(non_zero_changes) > 0:
                    print(f"  ✓ Price change mean: {non_zero_changes.mean():.8f}")
                    print(f"  ✓ Price change std: {non_zero_changes.std():.8f}")
    
    def test_signal_generation(self):
        """Test signal generation patterns"""
        print("\n🔍 Test 3: Signal Generation Analysis")
        print("-" * 40)
        
        for name, df in [("Train", self.train_df), ("Test", self.test_df)]:
            print(f"\n{name} Dataset:")
            
            # Check target columns
            target_cols = [col for col in df.columns if col.startswith('target_')]
            print(f"  ✓ Target columns: {target_cols}")
            
            if 'target_spread_compression_v3_binary' in df.columns:
                signal_col = 'target_spread_compression_v3_binary'
                signals = df[signal_col]
                
                signal_counts = signals.value_counts().sort_index()
                print(f"  ✓ Signal distribution: {signal_counts.to_dict()}")
                
                # Check for signal clustering (too many consecutive signals)
                if 1 in signal_counts.index:
                    signal_periods = df[signals == 1]
                    if len(signal_periods) > 1:
                        signal_gaps = signal_periods.index.to_series().diff().dropna()
                        min_gap = signal_gaps.min()
                        mean_gap = signal_gaps.mean()
                        print(f"  ✓ Signal gaps - min: {min_gap}, mean: {mean_gap}")
    
    def test_return_calculations(self):
        """Simulate return calculations to identify NaN sources"""
        print("\n🔍 Test 4: Return Calculation Simulation")
        print("-" * 40)
        
        # Use test dataset for this simulation
        df = self.test_df.copy()
        
        if 'close_og' not in df.columns or 'target_spread_compression_v3_binary' not in df.columns:
            print("  ❌ Missing required columns for return calculation test")
            return
        
        print("  Simulating return calculations...")
        
        # Get signal positions
        signals = df['target_spread_compression_v3_binary']
        buy_signals = df[signals == 1].copy()
        
        if len(buy_signals) == 0:
            print("  ⚠️  No BUY signals found in test data")
            return
        
        print(f"  ✓ Found {len(buy_signals)} BUY signals")
        
        # Simulate return calculation for first 10 signals
        sample_signals = buy_signals.head(10)
        
        returns_list = []
        issues_found = []
        
        for i, (entry_time, entry_row) in enumerate(sample_signals.iterrows()):
            entry_price = entry_row['close_og']
            
            # Simulate 60-minute holding period
            exit_time = entry_time + pd.Timedelta(minutes=60)
            
            # Find actual exit price
            future_data = df[df.index > entry_time]
            if len(future_data) == 0:
                issues_found.append(f"Signal {i}: No future data available")
                continue
                
            # Get exit price (first available price after 60 minutes)
            exit_candidates = future_data[future_data.index >= exit_time]
            if len(exit_candidates) == 0:
                # Use last available price
                exit_price = future_data['close_og'].iloc[-1]
                actual_exit_time = future_data.index[-1]
            else:
                exit_price = exit_candidates['close_og'].iloc[0]
                actual_exit_time = exit_candidates.index[0]
            
            # Calculate return
            if entry_price <= 0:
                issues_found.append(f"Signal {i}: Entry price <= 0 ({entry_price})")
                continue
                
            if exit_price <= 0:
                issues_found.append(f"Signal {i}: Exit price <= 0 ({exit_price})")
                continue
            
            # Standard return calculation
            raw_return = (exit_price - entry_price) / entry_price
            bps_return = raw_return * 10000
            
            returns_list.append({
                'signal_id': i,
                'entry_time': entry_time,
                'exit_time': actual_exit_time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'raw_return': raw_return,
                'bps_return': bps_return
            })
            
            print(f"  Signal {i}: Entry={entry_price:.5f}, Exit={exit_price:.5f}, Return={bps_return:.2f} bps")
        
        # Analyze results
        if returns_list:
            returns_df = pd.DataFrame(returns_list)
            
            print(f"\n  📊 Return Statistics:")
            print(f"    Mean return: {returns_df['bps_return'].mean():.2f} bps")
            print(f"    Std return: {returns_df['bps_return'].std():.2f} bps")
            print(f"    Min return: {returns_df['bps_return'].min():.2f} bps") 
            print(f"    Max return: {returns_df['bps_return'].max():.2f} bps")
            
            # Check for NaN or extreme values
            nan_returns = returns_df['bps_return'].isnull().sum()
            inf_returns = np.isinf(returns_df['bps_return']).sum()
            extreme_negative = (returns_df['bps_return'] < -100000).sum()
            
            print(f"    NaN returns: {nan_returns}")
            print(f"    Inf returns: {inf_returns}")
            print(f"    Extreme negative (< -100k bps): {extreme_negative}")
        
        # Report issues
        if issues_found:
            print(f"\n  ⚠️  Issues found:")
            for issue in issues_found[:5]:  # Show first 5 issues
                print(f"    - {issue}")
    
    def test_edge_cases(self):
        """Test edge cases that could cause NaN values"""
        print("\n🔍 Test 5: Edge Case Analysis")
        print("-" * 40)
        
        df = self.test_df.copy()
        
        # Test 5.1: Zero price scenarios
        if 'close_og' in df.columns:
            zero_prices = (df['close_og'] == 0).sum()
            print(f"  ✓ Zero prices: {zero_prices}")
            
            # Check for very small prices that might cause precision issues
            tiny_prices = (df['close_og'] < 0.00001).sum()
            print(f"  ✓ Tiny prices (< 0.00001): {tiny_prices}")
        
        # Test 5.2: Duplicate timestamps
        duplicate_indices = df.index.duplicated().sum()
        print(f"  ✓ Duplicate timestamps: {duplicate_indices}")
        
        # Test 5.3: Missing data gaps during signal periods
        if 'target_spread_compression_v3_binary' in df.columns:
            signal_times = df[df['target_spread_compression_v3_binary'] == 1].index
            
            gap_issues = 0
            for signal_time in signal_times[:10]:  # Check first 10 signals
                # Check if there's data in the next hour
                future_window = df[(df.index > signal_time) & 
                                 (df.index <= signal_time + pd.Timedelta(hours=1))]
                if len(future_window) == 0:
                    gap_issues += 1
            
            print(f"  ✓ Signals with no future data (1hr window): {gap_issues}")
        
        # Test 5.4: Extreme feature values that might indicate corruption
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        extreme_values = {}
        
        for col in numeric_cols:
            if col != 'close_og':  # Skip price columns
                col_data = df[col]
                extreme_positive = (col_data > 1e6).sum()
                extreme_negative = (col_data < -1e6).sum()
                if extreme_positive > 0 or extreme_negative > 0:
                    extreme_values[col] = {'pos': extreme_positive, 'neg': extreme_negative}
        
        if extreme_values:
            print(f"  ⚠️  Extreme values found:")
            for col, counts in extreme_values.items():
                print(f"    {col}: +{counts['pos']}, -{counts['neg']}")
        else:
            print(f"  ✓ No extreme values in features")
    
    def test_feature_values(self):
        """Test feature generation correctness"""
        print("\n🔍 Test 6: Feature Value Analysis")
        print("-" * 40)
        
        df = self.test_df.copy()
        
        # Check key features used in the model
        key_features = [
            'microstructure_momentum_fusion',
            'depth_stability', 
            'volume_ratio',
            'institutional_footprint',
            'top_predictor_interaction'
        ]
        
        for feature in key_features:
            if feature in df.columns:
                feature_data = df[feature]
                
                print(f"\n  {feature}:")
                print(f"    Range: [{feature_data.min():.4f}, {feature_data.max():.4f}]")
                print(f"    Mean: {feature_data.mean():.4f}")
                print(f"    Std: {feature_data.std():.4f}")
                
                # Check for suspicious patterns
                nan_count = feature_data.isnull().sum()
                inf_count = np.isinf(feature_data).sum()
                zero_count = (feature_data == 0).sum()
                
                print(f"    NaN: {nan_count}, Inf: {inf_count}, Zeros: {zero_count}")
                
                # Check if feature is constant (would indicate calculation error)
                unique_values = feature_data.nunique()
                if unique_values < 10:
                    print(f"    ⚠️  Only {unique_values} unique values (might be constant)")

def main():
    """Run the diagnostic"""
    try:
        diagnostic = DataPipelineDiagnostic()
        diagnostic.run_comprehensive_diagnostic()
        
        print("\n💡 RECOMMENDATIONS:")
        print("1. Check return calculation logic in backtesting module")
        print("2. Verify price data alignment between signal and execution times")
        print("3. Look for division by zero in return calculations")
        print("4. Check if signals are generated at times with valid future price data")
        print("5. Verify stop-loss/take-profit calculations don't create extreme values")
        
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()