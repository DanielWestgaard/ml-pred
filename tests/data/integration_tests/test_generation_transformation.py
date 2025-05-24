import unittest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import patch, MagicMock

# Import test utilities
from tests.utils.test_data_utils import create_sample_ohlcv_data
# Import the modules to test
from data.features.generation import FeatureGenerator
from data.features.transformation import FeatureTransformer


class TestGenerationTransformationIntegration(unittest.TestCase):
    
    def setUp(self):
        """Set up test data."""
        # Create sample data
        self.df = create_sample_ohlcv_data(rows=250, with_issues=False)
        self.df.set_index('date', inplace=True)
    
    def test_feature_generation_and_transformation(self):
        """Test integration between FeatureGenerator and FeatureTransformer."""
        # Step 1: Generate features
        generator = FeatureGenerator(self.df)
        df_with_features = generator.run()
        
        # Verify feature generation
        self.assertGreater(len(df_with_features.columns), len(self.df.columns),
                          "New features should be added during generation")
        
        # Check for NaN values that need to be handled in transformation
        has_nans = df_with_features.isna().any().any()
        
        # Step 2: Transform features
        transformer = FeatureTransformer(df_with_features)
        transformed_df = transformer.run()
        
        # Verify transformation
        self.assertEqual(transformed_df.isna().sum().sum(), 0,
                       "Transformed data should not have NaN values")
        
        # Check that close_og exists (original close preserved)
        self.assertIn('close_og', transformed_df.columns,
                     "Original close price should be preserved as close_og")
        
        # If there were NaN values, they should be handled now
        if has_nans:
            self.assertLess(transformed_df.isna().sum().sum(), df_with_features.isna().sum().sum(),
                          "Transformation should reduce NaN values")
    
    def test_normalization_after_generation(self):
        """Test that features are properly normalized after generation."""
        # Step 1: Generate specific features
        generator = FeatureGenerator(self.df)
        
        # Generate just a few key features for testing
        generator.moving_averages()
        generator.bollinger_bands()
        generator.relative_strength_index()
        
        df_with_features = generator.df
        
        # Step 2: Transform and normalize
        transformer = FeatureTransformer(df_with_features)
        transformed_df = transformer.run(preserve_original=True)
        
        # Verify that original and normalized versions exist
        self.assertIn('close', transformed_df.columns)
        self.assertIn('close_norm', transformed_df.columns)
        
        # Check that RSI (bounded indicator) is normalized correctly
        if 'rsi_14' in transformed_df.columns and 'rsi_14_norm' in transformed_df.columns:
            # Original should be in range [0, 100]
            original_rsi = transformed_df['rsi_14'].dropna()
            self.assertTrue(original_rsi.min() >= 0 and original_rsi.max() <= 100)
            
            # Normalized should be in range [0, 1]
            normalized_rsi = transformed_df['rsi_14_norm'].dropna()
            self.assertTrue(normalized_rsi.min() >= 0 and normalized_rsi.max() <= 1)
    
    def test_feature_dependencies(self):
        """Test that features with dependencies are properly generated and transformed."""
        # Some features depend on other features being generated first
        # For example, Bollinger Bands depend on SMA
        
        # Use FeatureGenerator to create features with dependencies
        generator = FeatureGenerator(self.df)
        
        # Only generate Bollinger Bands (depends on SMA)
        # This should implicitly calculate SMA if not already present
        generator.bollinger_bands()
        
        # Verify that both SMA and Bollinger Bands are present
        self.assertIn('sma_20', generator.df.columns)
        self.assertIn('upper_band', generator.df.columns)
        self.assertIn('lower_band', generator.df.columns)
        
        # Now transform these features
        transformer = FeatureTransformer(generator.df)
        transformed_df = transformer.run()
        
        # Verify transformation of dependent features
        self.assertEqual(transformed_df.isna().sum().sum(), 0,
                        "Transformed data should not have NaN values, even for dependent features")
    
    def test_edge_case_short_history(self):
        """Test generation and transformation with short price history."""
        # Create a very short price history
        short_df = create_sample_ohlcv_data(rows=30, with_issues=False)
        short_df.set_index('date', inplace=True)
        
        # Generate features (some features require longer history)
        generator = FeatureGenerator(short_df)
        
        # This should not crash but might have limited feature generation
        features_df = generator.run()
        
        # Transform the features
        transformer = FeatureTransformer(features_df)
        transformed_df = transformer.run()
        
        # Verify basics still work
        self.assertEqual(transformed_df.isna().sum().sum(), 0,
                        "Transformed data should not have NaN values, even for short history")
        self.assertIn('close_og', transformed_df.columns,
                     "Original close price should be preserved as close_og")
