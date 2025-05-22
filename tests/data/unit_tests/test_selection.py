import unittest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import patch, MagicMock, ANY

# Import test utilities
from tests.utils.test_data_utils import create_sample_ohlcv_data, mock_model_utils
# Import the module to test
from data.features.selection import FeatureSelector


class TestFeatureSelector(unittest.TestCase):
    
    def setUp(self):
        """Set up test data."""
        # Create mock for model_utils that might be imported
        self.mock_module = mock_model_utils()
        
        # Create sample data
        self.df = create_sample_ohlcv_data(rows=200, with_issues=False)
        self.df.set_index('date', inplace=True)
        
        # Add some synthetic features for testing selection
        self.df['sma_5'] = self.df['close'].rolling(window=5).mean()
        self.df['sma_20'] = self.df['close'].rolling(window=20).mean()
        self.df['rsi_14'] = 50 + np.random.normal(0, 15, len(self.df))  # Simulate RSI values
        self.df['rsi_14'] = self.df['rsi_14'].clip(0, 100)  # Ensure RSI is in [0, 100]
        self.df['macd_line'] = self.df['close'] - self.df['sma_20']
        self.df['atr'] = np.random.random(len(self.df)) * 2
        
        # Create highly correlated features for testing
        self.df['correlated_with_close'] = self.df['close'] * 1.05 + np.random.normal(0, 0.1, len(self.df))
        self.df['correlated_with_sma5'] = self.df['sma_5'] * 0.95 + np.random.normal(0, 0.1, len(self.df))
        
        # Create temporary directory for saving plots
        self.temp_dir = os.path.join(os.getcwd(), 'temp_test_selection')
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
    def test_initialization(self, mock_mkdir):
        """Test initialization of FeatureSelector."""
        # Mock config for storage path
        with patch('data.features.selection.config.FE_SEL_BASE_DIR', self.temp_dir):
            selector = FeatureSelector(self.df, target_col='close')
            
            # Check DataFrame is stored correctly
            self.assertEqual(len(selector.df), len(self.df))
            self.assertEqual(selector.target_col, 'close')
            
            # Check X and y are created for model training
            self.assertEqual(len(selector.X), len(self.df))
            self.assertEqual(len(selector.y), len(self.df))
            
            # Check directory is created for saving plots
            mock_mkdir.assert_called_once()
    
    @patch('data.features.selection.config', MagicMock())
    @patch('data.features.selection.os.mkdir')
    @patch('matplotlib.pyplot.savefig')
    def test_cfs_method(self, mock_savefig, mock_mkdir):
        """Test Correlation-based Feature Selection (CFS) method."""
        # Mock config for storage path
        with patch('data.features.selection.config.FE_SEL_BASE_DIR', self.temp_dir):
            selector = FeatureSelector(self.df, target_col='close')
            
            # Run CFS
            selected_features = selector.cfs(k=5)
            
            # Check that some features are selected
            self.assertGreater(len(selected_features), 0)
            
            # Check that target column is included
            self.assertIn('close', selected_features)
            
            # Check that highly correlated features aren't both selected
            if 'correlated_with_close' in selected_features:
                # Should only select one of a highly correlated pair
                correlated_pair = ['correlated_with_close', 'correlated_with_sma5']
                self.assertLessEqual(sum(1 for f in correlated_pair if f in selected_features), 1)
            
            # Check that plot was saved
            mock_savefig.assert_called_once()
    
    @patch('data.features.selection.config', MagicMock())
    @patch('data.features.selection.os.mkdir')
    @patch('matplotlib.pyplot.savefig')
    def test_xgb_regressor_method(self, mock_savefig, mock_mkdir):
        """Test XGBoost feature importance selection method."""
        # Mock config for storage path
        with patch('data.features.selection.config.FE_SEL_BASE_DIR', self.temp_dir):
            selector = FeatureSelector(self.df, target_col='close')
            
            # Run XGBoost feature selection
            selected_features = selector.xgb_regressor(threshold=0.01)
            
            # Check that some features are selected
            self.assertGreater(len(selected_features), 0)
            
            # Check that plot was saved
            mock_savefig.assert_called_once()
    
    @patch('data.features.selection.config', MagicMock())
    @patch('data.features.selection.os.mkdir')
    @patch('matplotlib.pyplot.savefig')
    def test_rfe_method(self, mock_savefig, mock_mkdir):
        """Test Recursive Feature Elimination (RFE) method."""
        # Mock config for storage path
        with patch('data.features.selection.config.FE_SEL_BASE_DIR', self.temp_dir):
            selector = FeatureSelector(self.df, target_col='close')
            
            # Run RFE with specified number of features
            n_features = 5
            selected_features = selector.rfe(n_features_to_select=n_features)
            
            # Check that the correct number of features is selected
            self.assertLessEqual(len(selected_features), n_features + 1)  # +1 for target if included
            
            # Check that plot was saved
            mock_savefig.assert_called_once()
    
    @patch('data.features.selection.config', MagicMock())
    @patch('data.features.selection.os.mkdir')
    def test_run_method_single_method(self, mock_mkdir):
        """Test run method with a single selection method."""
        # Mock config for storage path
        with patch('data.features.selection.config.FE_SEL_BASE_DIR', self.temp_dir):
            selector = FeatureSelector(self.df, target_col='close')
            
            # Patch the selection methods to return known feature sets
            with patch.object(selector, 'cfs', return_value=['close', 'sma_5', 'rsi_14']):
                # Run with only CFS method
                selected_df = selector.run(methods=['cfs'], k_features=3)
                
                # Check that the result is a DataFrame
                self.assertIsInstance(selected_df, pd.DataFrame)
                
                # Check that it has the expected columns
                expected_columns = ['close', 'sma_5', 'rsi_14']
                self.assertEqual(set(selected_df.columns), set(expected_columns))
    
    @patch('data.features.selection.config', MagicMock())
    @patch('data.features.selection.os.mkdir')
    def test_run_method_multiple_methods(self, mock_mkdir):
        """Test run method with multiple selection methods."""
        # Mock config for storage path
        with patch('data.features.selection.config.FE_SEL_BASE_DIR', self.temp_dir):
            selector = FeatureSelector(self.df, target_col='close')
            
            # Patch the selection methods to return known feature sets
            with patch.object(selector, 'cfs', return_value=['close', 'sma_5', 'rsi_14']), \
                 patch.object(selector, 'xgb_regressor', return_value=['close', 'sma_5', 'macd_line']), \
                 patch.object(selector, 'rfe', return_value=['close', 'rsi_14', 'macd_line']):
                
                # Run with all three methods
                selected_df = selector.run(methods=['cfs', 'xgb', 'rfe'], k_features=3)
                
                # Check that the result is a DataFrame
                self.assertIsInstance(selected_df, pd.DataFrame)
                
                # Features selected by at least 2 methods should be:
                # 'close' (all 3), 'rsi_14' (cfs, rfe), 'sma_5' (cfs, xgb), 'macd_line' (xgb, rfe)
                expected_columns = ['close', 'rsi_14', 'sma_5', 'macd_line']
                self.assertEqual(set(selected_df.columns), set(expected_columns))
    
    @patch('data.features.selection.config', MagicMock())
    @patch('data.features.selection.os.mkdir')
    def test_error_handling(self, mock_mkdir):
        """Test error handling in selection methods."""
        # Mock config for storage path
        with patch('data.features.selection.config.FE_SEL_BASE_DIR', self.temp_dir):
            selector = FeatureSelector(self.df, target_col='close')
            
            # Patch the selection methods to raise exceptions
            with patch.object(selector, 'cfs', side_effect=Exception("Test error")):
                # Should return None but not crash
                result = selector.cfs()
                self.assertIsNone(result)
            
            with patch.object(selector, 'xgb_regressor', side_effect=Exception("Test error")):
                # Should return None but not crash
                result = selector.xgb_regressor()
                self.assertIsNone(result)
            
            with patch.object(selector, 'rfe', side_effect=Exception("Test error")):
                # Should return None but not crash
                result = selector.rfe()
                self.assertIsNone(result)
