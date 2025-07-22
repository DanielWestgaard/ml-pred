#!/usr/bin/env python3
"""
Feature Mapper Script - Maps model metadata features to FeatureGenerator methods
"""

import sys
import yaml
import inspect
from data.features.generation import FeatureGenerator
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_model_features():
    """Load the expected features from model metadata"""
    metadata_path = "storage/models/2025-07-03_12:45:35/model_metadata.yml"
    
    with open(metadata_path, 'r') as file:
        metadata = yaml.safe_load(file)
    
    return metadata['dataset']['features']

def get_feature_generator_methods():
    """Get all public methods from FeatureGenerator class"""
    methods = []
    for name, method in inspect.getmembers(FeatureGenerator, predicate=inspect.isfunction):
        if not name.startswith('_') and name not in ['run', 'safely_execute']:
            methods.append(name)
    return methods

def test_feature_generation(method_name):
    """Test what features a method generates"""
    try:
        # Create simple test data
        dates = pd.date_range(start='2024-01-01', periods=200, freq='1H')
        test_data = pd.DataFrame({
            'open': np.random.uniform(1.25, 1.26, 200),
            'high': np.random.uniform(1.25, 1.26, 200),
            'low': np.random.uniform(1.24, 1.25, 200),
            'close': np.random.uniform(1.25, 1.26, 200),
            'volume': np.random.randint(100, 1000, 200)
        }, index=dates)
        
        # Fix OHLC relationships
        test_data['high'] = test_data[['open', 'high', 'low', 'close']].max(axis=1)
        test_data['low'] = test_data[['open', 'high', 'low', 'close']].min(axis=1)
        
        # Get initial columns
        initial_cols = set(test_data.columns)
        
        # Generate features using specific method
        generator = FeatureGenerator(test_data)
        
        # Call the specific method
        method = getattr(generator, method_name)
        method()
        
        # Get new columns
        new_cols = set(generator.df.columns) - initial_cols
        
        return list(new_cols)
        
    except Exception as e:
        print(f"Error testing {method_name}: {e}")
        return []

def main():
    """Main function to map features"""
    print("Loading model features...")
    model_features = load_model_features()
    print(f"Model expects {len(model_features)} features:")
    for i, feature in enumerate(model_features, 1):
        print(f"  {i:2d}. {feature}")
    
    print("\nTesting FeatureGenerator methods...")
    methods = get_feature_generator_methods()
    
    # Test each method to see what features it generates
    method_feature_map = {}
    for method in methods:
        print(f"Testing {method}...")
        features = test_feature_generation(method)
        if features:
            method_feature_map[method] = features
            print(f"  -> Generates: {features}")
        else:
            print(f"  -> No features generated")
    
    # Find which methods generate the model features
    print("\nMapping model features to methods:")
    required_methods = set()
    feature_to_method = {}
    
    for model_feature in model_features:
        found = False
        for method, method_features in method_feature_map.items():
            if model_feature in method_features:
                feature_to_method[model_feature] = method
                required_methods.add(method)
                found = True
                break
        
        if not found:
            print(f"WARNING: Feature '{model_feature}' not found in any method!")
    
    print(f"\nRequired methods for config ({len(required_methods)}):")
    for method in sorted(required_methods):
        print(f"  - {method}")
    
    # Generate config format
    print("\nConfig format:")
    print("selected_features: [")
    for method in sorted(required_methods):
        print(f'    "{method}",')
    print("]")
    
    # Show detailed mapping
    print("\nDetailed feature mapping:")
    for method in sorted(required_methods):
        method_features = method_feature_map[method]
        model_features_in_method = [f for f in model_features if f in method_features]
        print(f"\n{method}:")
        for feature in model_features_in_method:
            print(f"  - {feature}")

if __name__ == "__main__":
    main()