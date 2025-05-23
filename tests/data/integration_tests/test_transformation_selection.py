import unittest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import patch, MagicMock

# Import test utilities
from tests.utils.test_data_utils import create_sample_ohlcv_data, mock_model_utils
# Import the modules to test
from data.features.transformation import FeatureTransformer
from data.features.selection import FeatureSelector


class TestTransformationSelectionIntegration(unittest.TestCase):
    
    def setUp(self):
        """Set up test data."""
        # Create mock for model_utils that might be imported
        self.mock_module = mock_model_utils()
        
        # Create sample data with features
        self.df = create_sample_ohlcv_data(rows=200, with_issues=False)
        self.df.set_index('date', inplace=True)
        
        # Add synthetic features
        self.df['sma_5'] = self.df['close'].rolling(window=5).mean()
        self.df['sma_20'] = self.df['close'].rolling(window=20).mean()
        self.df['rsi_14'] = 50 + np.random.normal(0, 15, len(self.df)).clip(-50, 50)
        self.df['macd_line'] = self.df['close'] - self.df['sma_20']
        self.df['signal_line'] = self.df['macd_line'].ewm(span=9).mean()
        self.df['histogram'] = self.df['macd_line'] - self.df['signal_line']
        
        # Add highly correlated features
        self.df['close_clone'] = self.df['close'] * 1.01
        self.df['volume_scaled'] = self.df['volume'] * 0.5
        
        # Create NaN values
        self.df['sma_5'].iloc[:5] = np.nan
        self.df['rsi_14'].iloc[:15] = np.nan
        
        # Create temporary directory for saving plots
        self.temp_dir = os.path.join(os.getcwd(), 'temp_test_transform_select')
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
    
    @patch('data.features.selection.config', MagicMock())
    @patch('data.features.selection.os.mkdir')
    def test_transformation_selection_integration(self, mock_mkdir):
        """Test integration between FeatureTransformer and FeatureSelector."""
        # Mock config for storage path
        with patch('data.features.selection.config.FE_SEL_BASE_DIR', self.temp_dir):
            # Step 1: Transform features
            transformer = FeatureTransformer(self.df.copy())
            transformed_df = transformer.run()
            
            # Verify transformation
            self.assertEqual(transformed_df.isna().sum().sum(), 0,
                           "Transformed data should not have NaN values")
            
            # Step 2: Select features
            selector = FeatureSelector(transformed_df, target_col='close')
            
            # Mock the feature selection methods to avoid actual plotting and model training
            with patch.object(selector, 'cfs', return_value=['close', 'sma_5', 'rsi_14']), \
                 patch.object(selector, 'xgb_regressor', return_value=['close', 'sma_20', 'macd_line']), \
                 patch.object(selector, 'rfe', return_value=['close', 'rsi_14', 'signal_line']):
                
                selected_df = selector.run(methods=['cfs', 'xgb', 'rfe'])
                
                # Verify feature selection
                self.assertLess(len(selected_df.columns), len(transformed_df.columns),
                              "Selected features should be fewer than transformed features")
                self.assertIn('close', selected_df.columns,
                             "Target column should be included in selected features")
    
    @patch('data.features.selection.config', MagicMock())
    @patch('data.features.selection.os.mkdir')
    def test_correlated_feature_handling(self, mock_mkdir):
        """Test handling of correlated features in transformation and selection."""
        # Mock config for storage path
        with patch('data.features.selection.config.FE_SEL_BASE_DIR', self.temp_dir):
            # Step 1: Transform features with preserve_original=True
            transformer = FeatureTransformer(self.df.copy())
            transformed_df = transformer.run(preserve_original=True)
            
            # This should create normalized versions of features
            self.assertIn('close_norm', transformed_df.columns)
            
            # Step 2: Select features using actual correlation method
            # (this will actually test correlation-based selection)
            selector = FeatureSelector(transformed_df, target_col='close')
            
            # Test with just CFS to check correlation handling
            selected_features = selector.cfs(k=5, corr_threshold=0.7)
            
            # Verify that not both highly correlated features are selected
            correlated_pair = ['close', 'close_clone']
            self.assertLessEqual(sum(1 for f in correlated_pair if f in selected_features), 1,
                               "Highly correlated features should not both be selected")
    
    @patch('data.features.selection.config', MagicMock())
    @patch('data.features.selection.os.mkdir')
    def test_normalization_impact_on_selection(self, mock_mkdir):
        """Test how normalization affects feature selection."""
        # Mock config for storage path
        with patch('data.features.selection.config.FE_SEL_BASE_DIR', self.temp_dir):
            # Transform features in two ways: with and without preserve_original
            transformer1 = FeatureTransformer(self.df.copy())
            df_with_original = transformer1.run(preserve_original=True)
            
            transformer2 = FeatureTransformer(self.df.copy())
            df_normalized = transformer2.run(preserve_original=False)
            
            # Select features from both dataframes
            selector1 = FeatureSelector(df_with_original, target_col='close')
            
            # Use a different target for the normalized data (since 'close' is normalized)
            selector2 = FeatureSelector(df_normalized, target_col='close')
            
            # Test with correlation method to analyze real differences
            # No mocking here to see the actual effect of normalization
            features_with_original = selector1.cfs(k=5)
            features_normalized = selector2.cfs(k=5)
            
            # Both sets should include the target
            self.assertIn('close', features_with_original)
            self.assertIn('close', features_normalized)
            
            # The feature sets might differ somewhat due to normalization
            # But we still check they have reasonable size
            self.assertGreater(len(features_with_original), 1)
            self.assertGreater(len(features_normalized), 1)
