#!/usr/bin/env python3
"""
Test Both Pipeline Approaches - Ensure Consistency
=================================================

This script validates that both the engineer_pipeline (for manual training data) 
and the live data manager (for strategy execution) produce consistent results
using the unified feature generation system.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.features.enhanced_generation import ModelAwareFeatureGenerator
from utils.logging_utils import get_logger

logger = get_logger()


def create_test_data(n_rows=300):
    """Create realistic test OHLCV data"""
    dates = pd.date_range(start='2024-01-01', periods=n_rows, freq='h')
    
    base_price = 1.2500
    price_changes = np.cumsum(np.random.normal(0, 0.0001, n_rows))
    close_prices = base_price + price_changes
    
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = base_price
    
    high_noise = np.abs(np.random.normal(0, 0.00005, n_rows))
    low_noise = np.abs(np.random.normal(0, 0.00005, n_rows))
    
    high_prices = np.maximum(open_prices, close_prices) + high_noise
    low_prices = np.minimum(open_prices, close_prices) - low_noise
    
    volumes = np.random.randint(100, 1000, n_rows)
    
    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }, index=dates)
    
    return df


def test_strategy_approach():
    """Test the approach used by Strategy component (via LiveDataManager)"""
    test_data = create_test_data()
    return test_strategy_approach_with_data(test_data)

def test_strategy_approach_with_data(test_data):
    """Test the approach used by Strategy component with provided data"""
    print("=" * 60)
    print("TESTING STRATEGY APPROACH (Model-Aware Generation)")
    print("=" * 60)
    
    print(f"Test data shape: {test_data.shape}")
    
    # This mirrors what LiveDataManager does
    try:
        feature_generator = ModelAwareFeatureGenerator(test_data.copy())
        
        # Test with specific features (instead of model metadata which may not exist)
        strategy_features = [
            'atr',                    # Volatility indicator
            'hour_sin', 'hour_cos',   # Time features
            'sma_20', 'ema_20',      # Moving averages
            'asian_session',          # Trading sessions
            'returns',               # Market regime base
            'rsi_14',                # Oscillator
            'macd_line'              # Momentum
        ]
        
        df_with_features = feature_generator.generate_selected(strategy_features)
        
        print(f"Result shape: {df_with_features.shape}")
        
        # Check which features were generated
        generated = [f for f in strategy_features if f in df_with_features.columns]
        missing = [f for f in strategy_features if f not in df_with_features.columns]
        
        print(f"✅ Generated features: {generated}")
        if missing:
            print(f"❌ Missing features: {missing}")
        
        return df_with_features, len(missing) == 0
        
    except Exception as e:
        print(f"❌ Error in strategy approach: {e}")
        import traceback
        traceback.print_exc()
        return None, False


def test_training_pipeline_approach():
    """Test the approach used by EngineeringPipeline (Direct Enhanced Generation)"""
    test_data = create_test_data()
    return test_training_pipeline_approach_with_data(test_data)

def test_training_pipeline_approach_with_data(test_data):
    """Test the approach used by EngineeringPipeline with provided data"""
    print("\n" + "=" * 60)
    print("TESTING TRAINING PIPELINE APPROACH (Direct Enhanced Generation)")
    print("=" * 60)
    
    print(f"Test data shape: {test_data.shape}")
    
    # This mirrors what EngineeringPipeline now does directly
    try:
        # Test with old dict format (what EngineeringPipeline expects from hypothesis configs)
        selected_features_dict = {
            'moving_averages': [],
            'time_of_day': [],
            'average_true_range': [],
            'relative_strength_index': [],
            'fx_trading_sessions': [],
            'volatility_regimes': [],
            'moving_average_convergence_divergence': []
        }
        
        print(f"Selected generators: {list(selected_features_dict.keys())}")
        
        # Direct enhanced generation (like updated EngineeringPipeline)
        feature_generator = ModelAwareFeatureGenerator(test_data.copy())
        
        # Handle dict format directly (like the updated pipeline does)
        if isinstance(selected_features_dict, dict):
            generator_names = list(selected_features_dict.keys())
            registry = feature_generator._registry
            execution_order = registry.resolve_dependencies(generator_names)
            
            print(f"Execution order: {execution_order}")
            
            for generator_name in execution_order:
                if hasattr(feature_generator, generator_name):
                    method = getattr(feature_generator, generator_name)
                    method()
            
            df_with_features = feature_generator.df
        
        print(f"Result shape: {df_with_features.shape}")
        
        # Check for expected features that should be generated
        expected_from_generators = [
            'sma_20', 'ema_20',       # from moving_averages
            'hour_sin', 'hour_cos',   # from time_of_day  
            'atr',                    # from average_true_range
            'rsi_14',                 # from relative_strength_index
            'asian_session',          # from fx_trading_sessions
            'returns',                # from volatility_regimes
            'macd_line'               # from moving_average_convergence_divergence
        ]
        
        found = [f for f in expected_from_generators if f in df_with_features.columns]
        missing = [f for f in expected_from_generators if f not in df_with_features.columns]
        
        print(f"✅ Found expected features: {found}")
        if missing:
            print(f"❌ Missing expected features: {missing}")
        
        return df_with_features, len(missing) == 0
        
    except Exception as e:
        print(f"❌ Error in training pipeline approach: {e}")
        import traceback
        traceback.print_exc()
        return None, False


def compare_results(strategy_df, training_df):
    """Compare results from both approaches"""
    print("\n" + "=" * 60)
    print("COMPARING RESULTS")
    print("=" * 60)
    
    if strategy_df is None or training_df is None:
        print("❌ Cannot compare - one or both approaches failed")
        return False
    
    # Find common features
    strategy_features = set(strategy_df.columns)
    training_features = set(training_df.columns)
    
    common_features = strategy_features.intersection(training_features)
    strategy_only = strategy_features - training_features
    training_only = training_features - strategy_features
    
    print(f"Strategy approach generated: {len(strategy_features)} features")
    print(f"Training approach generated: {len(training_features)} features")
    print(f"Common features: {len(common_features)}")
    
    if strategy_only:
        print(f"Strategy-only features: {sorted(list(strategy_only))}")
    if training_only:
        print(f"Training-only features: {sorted(list(training_only))}")
    
    # Compare values for common features (for basic OHLCV columns)
    basic_columns = ['open', 'high', 'low', 'close', 'volume']
    
    print(f"\n🔍 Comparing basic column values...")
    all_match = True
    
    for col in basic_columns:
        if col in strategy_df.columns and col in training_df.columns:
            # Compare values (should be identical since using same input data)
            values_match = strategy_df[col].equals(training_df[col])
            print(f"  {col}: {'✅ Match' if values_match else '❌ Differ'}")
            if not values_match:
                all_match = False
    
    # Test a few generated features if they exist in both
    test_features = ['atr', 'sma_20', 'hour_sin']
    print(f"\n🔍 Comparing generated feature values...")
    
    for feature in test_features:
        if feature in common_features:
            # Compare first few non-NaN values
            s_vals = strategy_df[feature].dropna().head(10)
            t_vals = training_df[feature].dropna().head(10)
            
            if len(s_vals) > 0 and len(t_vals) > 0:
                # Check if values are close (allowing for floating point differences)
                close_match = np.allclose(s_vals.values, t_vals.values, rtol=1e-10, atol=1e-10)
                print(f"  {feature}: {'✅ Values match' if close_match else '❌ Values differ'}")
                
                if not close_match:
                    print(f"    Strategy: {s_vals.iloc[0]}")
                    print(f"    Training: {t_vals.iloc[0]}")
                    all_match = False
    
    return all_match


def main():
    """Run comprehensive pipeline comparison test"""
    print("UNIFIED FEATURE GENERATION - PIPELINE CONSISTENCY TEST")
    print("=" * 60)
    print("Testing that both manual training pipeline and live strategy pipeline")
    print("produce consistent results using the unified feature system.")
    
    # Use same test data for both approaches to ensure fair comparison
    shared_test_data = create_test_data(300)
    
    # Run both approaches with same data
    strategy_df, strategy_success = test_strategy_approach_with_data(shared_test_data)
    training_df, training_success = test_training_pipeline_approach_with_data(shared_test_data)
    
    # Compare results
    comparison_success = compare_results(strategy_df, training_df)
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    results = [
        ("Strategy Approach (Model-Aware)", strategy_success),
        ("Training Approach (Direct Enhanced Generation)", training_success),
        ("Results Consistency", comparison_success)
    ]
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 SUCCESS! Both pipeline approaches work correctly and consistently!")
        print("\n✅ The unified feature generation system provides:")
        print("  • Consistent results between training and live trading")
        print("  • Backward compatibility for existing configs")
        print("  • Model-aware capabilities for strategy execution")
        print("  • Robust feature generation for both use cases")
        
        print("\n🚀 Your system is ready for:")
        print("  • Manual historical data processing for model training")
        print("  • Live feature generation during strategy execution")
        print("  • Seamless transition between training and trading workflows")
    else:
        print("\n⚠️  Some tests failed. The system may need additional adjustments.")
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)