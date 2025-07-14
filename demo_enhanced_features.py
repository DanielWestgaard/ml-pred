#!/usr/bin/env python3
"""
Demonstration of Enhanced Feature Generation System
==================================================

This script demonstrates how the new feature generation system solves
the problems with the current approach:

1. Automatic feature discovery
2. Model-aware generation  
3. Dependency resolution
4. Clear validation and error handling
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.features.enhanced_generation import ModelAwareFeatureGenerator


def create_test_data(n_rows=200):
    """Create test OHLCV data"""
    dates = pd.date_range(start='2024-01-01', periods=n_rows, freq='1H')
    
    # Generate realistic price data
    base_price = 1.2500
    price_changes = np.cumsum(np.random.normal(0, 0.0001, n_rows))
    close_prices = base_price + price_changes
    
    # Generate OHLC with proper relationships
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = base_price
    
    high_prices = np.maximum(open_prices, close_prices) + np.abs(np.random.normal(0, 0.00005, n_rows))
    low_prices = np.minimum(open_prices, close_prices) - np.abs(np.random.normal(0, 0.00005, n_rows))
    volumes = np.random.randint(100, 1000, n_rows)
    
    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }, index=dates)
    
    return df


def demo_automatic_discovery():
    """Demonstrate automatic feature discovery"""
    print("=" * 60)
    print("1. AUTOMATIC FEATURE DISCOVERY")
    print("=" * 60)
    
    # Show available features without manual mapping
    available_features = ModelAwareFeatureGenerator.get_available_features()
    available_generators = ModelAwareFeatureGenerator.get_available_generators()
    
    print(f"Available features ({len(available_features)}):")
    for i, feature in enumerate(available_features, 1):
        print(f"  {i:2d}. {feature}")
    
    print(f"\nAvailable generators ({len(available_generators)}):")
    for i, generator in enumerate(available_generators, 1):
        print(f"  {i:2d}. {generator}")


def demo_model_aware_generation():
    """Demonstrate model-aware generation"""
    print("\n" + "=" * 60)
    print("2. MODEL-AWARE GENERATION")
    print("=" * 60)
    
    # Create test data
    test_data = create_test_data()
    print(f"Created test data: {test_data.shape}")
    
    # Generate features for our model
    generator = ModelAwareFeatureGenerator(test_data)
    
    try:
        # This would automatically generate exactly what the model needs
        result_df = generator.generate_for_model(
            "storage/models/2025-07-03_12:45:35/model_metadata.yml"
        )
        
        print(f"Generated features for model: {result_df.shape}")
        print("Features in result:")
        for i, col in enumerate(result_df.columns, 1):
            print(f"  {i:2d}. {col}")
    
    except Exception as e:
        print(f"Model-aware generation example: {e}")
        print("(This would work with a valid model metadata file)")


def demo_dependency_resolution():
    """Demonstrate automatic dependency resolution"""
    print("\n" + "=" * 60)
    print("3. DEPENDENCY RESOLUTION")
    print("=" * 60)
    
    test_data = create_test_data()
    generator = ModelAwareFeatureGenerator(test_data)
    
    # Show how dependencies are automatically resolved
    registry = ModelAwareFeatureGenerator._registry
    
    # Example: session_timing_features depends on fx_trading_sessions
    print("Dependencies for session_timing_features:")
    timing_def = registry.get_generator_info('session_timing_features')
    if timing_def:
        print(f"  Dependencies: {timing_def.dependencies}")
    
    # Show execution order resolution
    generators = ['session_timing_features', 'session_transition_features', 'fx_trading_sessions']
    try:
        execution_order = registry.resolve_dependencies(generators)
        print(f"\nOriginal order: {generators}")
        print(f"Resolved order: {execution_order}")
    except Exception as e:
        print(f"Dependency resolution error: {e}")


def demo_selected_features():
    """Demonstrate generating specific features"""
    print("\n" + "=" * 60)
    print("4. SELECTIVE FEATURE GENERATION")
    print("=" * 60)
    
    test_data = create_test_data()
    generator = ModelAwareFeatureGenerator(test_data)
    
    # Generate only specific features
    selected_features = [
        'atr',                    # From average_true_range
        'hour_sin', 'hour_cos',   # From time_of_day
        'asian_session',          # From fx_trading_sessions
        'volatility_short'        # From volatility_regimes
    ]
    
    print(f"Requesting features: {selected_features}")
    
    result_df = generator.generate_selected(selected_features)
    
    print(f"Generated dataset shape: {result_df.shape}")
    
    # Check which features were actually generated
    generated = [f for f in selected_features if f in result_df.columns]
    missing = [f for f in selected_features if f not in result_df.columns]
    
    print(f"Successfully generated ({len(generated)}): {generated}")
    if missing:
        print(f"Missing features ({len(missing)}): {missing}")


def demo_validation_and_errors():
    """Demonstrate validation and error handling"""
    print("\n" + "=" * 60)
    print("5. VALIDATION AND ERROR HANDLING")
    print("=" * 60)
    
    test_data = create_test_data()
    generator = ModelAwareFeatureGenerator(test_data)
    registry = ModelAwareFeatureGenerator._registry
    
    # Test feature validation
    test_features = ['atr', 'nonexistent_feature', 'hour_sin', 'another_fake_feature']
    available, unavailable = registry.validate_feature_availability(test_features)
    
    print(f"Requested features: {test_features}")
    print(f"Available: {available}")
    print(f"Unavailable: {unavailable}")
    
    # Show what happens when we try to generate unavailable features
    try:
        result_df = generator.generate_selected(test_features)
        print(f"Result shape: {result_df.shape}")
        print("System gracefully handled unavailable features")
    except Exception as e:
        print(f"Error: {e}")


def compare_with_old_approach():
    """Compare new vs old approach"""
    print("\n" + "=" * 60)
    print("6. COMPARISON: NEW VS OLD APPROACH")
    print("=" * 60)
    
    print("OLD APPROACH PROBLEMS:")
    print("❌ Manual method name mapping required")
    print("❌ No automatic dependency resolution")
    print("❌ Order-dependent execution")
    print("❌ Hard to discover available features")
    print("❌ Error-prone configuration")
    print("❌ No model-aware generation")
    
    print("\nNEW APPROACH BENEFITS:")
    print("✅ Automatic feature discovery")
    print("✅ Automatic dependency resolution")
    print("✅ Model-aware generation")
    print("✅ Clear validation and error messages")
    print("✅ Easy to add new features")
    print("✅ Robust error handling")
    
    print("\nHOW TO ADD A NEW FEATURE (old vs new):")
    print("\nOLD WAY:")
    print("1. Add method to FeatureGenerator")
    print("2. Manually update config with method name")
    print("3. Handle dependencies manually")
    print("4. Update documentation")
    print("5. Hope the order is correct")
    
    print("\nNEW WAY:")
    print("@feature_generator(output_features=['new_feature'], dependencies=['existing_feature'])")
    print("def my_new_feature(self):")
    print("    # Implementation")
    print("    self.df['new_feature'] = calculation")
    print("\n✅ That's it! Automatic registration and dependency handling")


def main():
    """Run all demonstrations"""
    print("ENHANCED FEATURE GENERATION SYSTEM DEMO")
    print("=" * 60)
    
    demo_automatic_discovery()
    demo_model_aware_generation()
    demo_dependency_resolution()
    demo_selected_features()
    demo_validation_and_errors()
    compare_with_old_approach()
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("The new enhanced feature generation system:")
    print("• Eliminates manual configuration errors")
    print("• Automatically handles dependencies")
    print("• Can generate exactly what models need")
    print("• Provides clear feedback and validation")
    print("• Is easily extensible and maintainable")
    print("• Solves all the problems with the current approach")


if __name__ == "__main__":
    main()