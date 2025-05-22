import unittest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import patch, MagicMock

# Import test utilities
from tests.utils.test_data_utils import create_sample_ohlcv_data
# Import the module to test
from data.features.transformation import FeatureTransformer


class TestFeatureTransformer(unittest.TestCase):
    
    def setUp(self):
        """Set up test data."""
        # Create sample data
        self.df = create_sample_ohlcv_data(rows=100, with_issues=False)
        self.df.set_index('date', inplace=True)
        
        # Add some synthetic features for testing transformation
        self.df['sma_5'] = self.df['close'].rolling(window=5).mean()
        self.df['sma_20'] = self.df['close'].rolling(window=20).mean()
        self.df['rsi_14'] = 50 + np.random.normal(0, 15, len(self.df))  # Simulate RSI values
        self.df['rsi_14'] = self.df['rsi_14'].clip(0, 100)  # Ensure RSI is in [0, 100]
        
        # Create some NaN values
        self.df.loc[self.df.index[10:15], 'sma_5'] = np.nan
        
        # Create some duplicated columns
        dup_col = self.df['close'].copy()
        self.df_with_dups = self.df.copy()
        self.df_with_dups['close'] = dup_col  # Create a duplicate 'close' column
    
    def test_initialization(self):
        """Test initialization of FeatureTransformer."""
        transformer = FeatureTransformer(self.df)
        self.assertEqual(len(transformer.df), len(self.df))
        self.assertEqual(len(transformer.df.columns), len(self.df.columns))
    
    def test_handle_missing_values(self):
        """Test handling of missing values."""
        # Count NaN values
        transformer = FeatureTransformer(self.df.copy())
        nan_count_before = transformer.df.isna().sum().sum()
        
        # Ensure we have NaNs to test with
        self.assertGreater(nan_count_before, 0)
        
        # Handle missing values
        transformer.handle_missing_values()
        
        # Check that NaNs are handled
        nan_count_after = transformer.df.isna().sum().sum()
        self.assertEqual(nan_count_after, 0, "All NaN values should be handled")
    
    def test_handle_duplicates(self):
        """Test handling of duplicate columns."""
        # Create transformer with duplicate columns
        transformer = FeatureTransformer(self.df_with_dups)
        
        # Find duplicates
        duplicate_cols = transformer.df.columns[transformer.df.columns.duplicated()].tolist()
        self.assertGreater(len(duplicate_cols), 0, "Test data should have duplicate columns")
        
        # Handle duplicates
        transformer.handle_duplicates()
        
        # Check that duplicates are renamed
        duplicate_cols_after = transformer.df.columns[transformer.df.columns.duplicated()].tolist()
        self.assertEqual(len(duplicate_cols_after), 0, "No duplicate columns should remain")
        
        # Check that a renamed column exists (e.g., 'close_1')
        self.assertIn('close_1', transformer.df.columns)
    
    def test_z_score_normalization(self):
        """Test z-score normalization."""
        transformer = FeatureTransformer(self.df.copy())
        
        # Apply z-score normalization to 'close' and 'volume'
        transformer.z_score(columns=['close', 'volume'], window=20)
        
        # Check that the normalized values have mean close to 0 and std close to 1
        # (after warmup period)
        for col in ['close', 'volume']:
            normalized_values = transformer.df[col].iloc[30:]  # Skip warmup period
            self.assertAlmostEqual(normalized_values.mean(), 0, delta=0.5)
            self.assertAlmostEqual(normalized_values.std(), 1, delta=0.5)
    
    def test_rolling_minmax(self):
        """Test min-max scaling using rolling window."""
        transformer = FeatureTransformer(self.df.copy())
        
        # Apply min-max scaling to RSI
        transformer.rolling_minmax(columns=['rsi_14'], window=20)
        
        # Check that the scaled values are in [0, 1]
        scaled_values = transformer.df['rsi_14'].dropna()
        self.assertTrue((scaled_values >= 0).all() and (scaled_values <= 1).all())
    
    def test_normalize_method(self):
        """Test the normalize method which applies different normalization based on column type."""
        transformer = FeatureTransformer(self.df.copy())
        
        # Apply normalization
        transformer.normalize(preserve_original=True, window=20)
        
        # Check that normalized columns are created when preserve_original=True
        self.assertIn('close_norm', transformer.df.columns)
        
        # Check that volume is log-transformed
        if 'volume_log' in transformer.df.columns:
            # Check that log transformation makes volume less skewed
            original_skew = np.abs(transformer.df['volume'].skew())
            log_skew = np.abs(transformer.df['volume_log'].skew())
            self.assertLess(log_skew, original_skew)
    
    def test_run_method_preserve_original(self):
        """Test the run method with preserve_original=True."""
        transformer = FeatureTransformer(self.df.copy())
        result_df = transformer.run(preserve_original=True, window=20)
        
        # Check that both original and normalized columns exist
        self.assertIn('close', result_df.columns)
        self.assertIn('close_norm', result_df.columns)
        
        # Check that NaNs are handled
        self.assertEqual(result_df.isna().sum().sum(), 0)
    
    def test_run_method_replace_original(self):
        """Test the run method with preserve_original=False."""
        original_cols = self.df.columns.tolist()
        transformer = FeatureTransformer(self.df.copy())
        result_df = transformer.run(preserve_original=False, window=20)
        
        # Check that columns are preserved but values are replaced
        self.assertEqual(set(result_df.columns.tolist()), set(original_cols))
        
        # Check that close_og is preserved (special case mentioned in code)
        self.assertIn('close_og', result_df.columns)
        
        # Check that NaNs are handled
        self.assertEqual(result_df.isna().sum().sum(), 0)
    
    def test_filter_features(self):
        """Test the _filter_features method."""
        # Create data with different window sizes in feature names
        df = self.df.copy()
        df['sma_200'] = df['close'].rolling(window=200).mean()  # Large window
        df['sma_5'] = df['close'].rolling(window=5).mean()  # Small window
        
        # Add more NaNs than the threshold for one feature
        df['test_feature'] = np.nan  # 100% NaNs
        
        transformer = FeatureTransformer(df)
        
        # Run filter features
        transformer._filter_features(threshold=0.5)
        
        # Check that features with too many NaNs are dropped
        self.assertNotIn('test_feature', transformer.df.columns)
        
        # Check that the dataframe is truncated based on the largest window
        self.assertLess(len(transformer.df), len(df))
    
    def test_handle_scattered_nans(self):
        """Test the _handle_scattered_nans method."""
        df = self.df.copy()
        
        # Create scattered NaNs
        scattered_indices = [30, 40, 50, 60]
        for idx in scattered_indices:
            df.loc[df.index[idx], 'sma_20'] = np.nan
        
        transformer = FeatureTransformer(df)
        
        # Handle scattered NaNs
        transformer._handle_scattered_nans()
        
        # Check that NaNs are filled
        self.assertEqual(transformer.df['sma_20'].isna().sum(), 0)
    
    def test_normalize_with_special_cases(self):
        """Test normalize with different feature groups."""
        df = self.df.copy()
        
        # Add features for different normalization groups
        df['rsi_14'] = 50 + np.random.normal(0, 15, len(df)).clip(-50, 50)  # For min-max
        df['macd'] = np.random.normal(0, 1, len(df))  # For z-score
        df['adx'] = 25 + np.random.normal(0, 10, len(df)).clip(-25, 75)  # For z-score
        df['stoch'] = np.random.random(len(df)) * 100  # For min-max
        
        transformer = FeatureTransformer(df)
        transformer.normalize(preserve_original=True, window=20)
        
        # Check that the appropriate normalization is applied
        # RSI and stochastic should be min-max scaled
        for col in ['rsi_14_norm', 'stoch_norm']:
            if col in transformer.df.columns:
                values = transformer.df[col].dropna()
                self.assertTrue((values >= 0).all() and (values <= 1).all())
        
        # MACD and ADX should be z-score normalized
        for col in ['macd_norm', 'adx_norm']:
            if col in transformer.df.columns:
                values = transformer.df[col].iloc[30:]  # Skip warmup
                self.assertAlmostEqual(values.mean(), 0, delta=0.5)
                self.assertAlmostEqual(values.std(), 1, delta=0.5)
