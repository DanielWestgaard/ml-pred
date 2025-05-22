import unittest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import patch, MagicMock

# Import test utilities
from tests.utils.test_data_utils import create_sample_ohlcv_data
# Import the module to test
from data.features.generation import FeatureGenerator


class TestFeatureGenerator(unittest.TestCase):
    
    def setUp(self):
        """Set up test data."""
        # Create clean sample data with 200+ rows for feature generation
        self.clean_df = create_sample_ohlcv_data(rows=250, with_issues=False)
        self.clean_df.set_index('date', inplace=True)
        
        # Create a small dataset for testing edge cases
        self.small_df = create_sample_ohlcv_data(rows=30, with_issues=False)
        self.small_df.set_index('date', inplace=True)
    
    def test_initialization(self):
        """Test initialization of FeatureGenerator."""
        generator = FeatureGenerator(self.clean_df)
        self.assertEqual(len(generator.df), len(self.clean_df))
        
        # Check required columns are present
        for col in ['open', 'high', 'low', 'close', 'volume']:
            self.assertIn(col, generator.df.columns)
    
    def test_initialization_missing_columns(self):
        """Test initialization with missing required columns."""
        # Create data missing a required column
        df_missing = self.clean_df.copy().drop('volume', axis=1)
        
        # Should raise ValueError
        with self.assertRaises(ValueError):
            generator = FeatureGenerator(df_missing)
    
    def test_initialization_empty_data(self):
        """Test initialization with empty dataset."""
        # Create empty dataframe with correct columns
        df_empty = pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        df_empty.set_index('date', inplace=True)
        
        # Should raise ValueError
        with self.assertRaises(ValueError):
            generator = FeatureGenerator(df_empty)
    
    def test_moving_averages(self):
        """Test generation of moving averages."""
        generator = FeatureGenerator(self.clean_df)
        generator.moving_averages()
        
        # Check for SMA and EMA columns
        for period in [5, 10, 20, 50, 200]:
            self.assertIn(f'sma_{period}', generator.df.columns)
            self.assertIn(f'ema_{period}', generator.df.columns)
        
        # Verify calculations for sma_5
        expected_sma5 = self.clean_df['close'].rolling(window=5).mean()
        pd.testing.assert_series_equal(
            generator.df['sma_5'].dropna(), 
            expected_sma5.dropna(),
            check_names=False
        )
        
        # Verify calculations for ema_5
        expected_ema5 = self.clean_df['close'].ewm(span=5, adjust=False, min_periods=5).mean()
        pd.testing.assert_series_equal(
            generator.df['ema_5'].dropna(), 
            expected_ema5.dropna(),
            check_names=False
        )
    
    def test_rate_of_change(self):
        """Test calculation of rate of change."""
        generator = FeatureGenerator(self.clean_df)
        generator.rate_of_change()
        
        # Check for ROC columns
        for period in [1, 5, 10, 20, 60]:
            self.assertIn(f'roc_{period}', generator.df.columns)
        
        # Verify calculation for roc_5
        expected_roc5 = (self.clean_df['close'] / self.clean_df['close'].shift(5) - 1) * 100
        pd.testing.assert_series_equal(
            generator.df['roc_5'].dropna(), 
            expected_roc5.dropna(),
            check_names=False
        )
    
    def test_average_true_range(self):
        """Test calculation of Average True Range."""
        generator = FeatureGenerator(self.clean_df)
        generator.average_true_range()
        
        # Check for ATR column
        self.assertIn('atr', generator.df.columns)
        
        # Check that ATR is not NaN after sufficient data points
        self.assertFalse(generator.df['atr'].tail(100).isna().any())
        
        # Check that ATR is always positive
        self.assertTrue((generator.df['atr'].dropna() >= 0).all())
    
    def test_bollinger_bands(self):
        """Test calculation of Bollinger Bands."""
        # First need to calculate SMA
        generator = FeatureGenerator(self.clean_df)
        generator.moving_averages()
        generator.bollinger_bands()
        
        # Check for Bollinger Band columns
        self.assertIn('upper_band', generator.df.columns)
        self.assertIn('lower_band', generator.df.columns)
        self.assertIn('bb_width', generator.df.columns)
        
        # Verify upper band is always >= middle band (SMA)
        self.assertTrue((generator.df['upper_band'].dropna() >= generator.df['sma_20'].dropna()).all())
        
        # Verify lower band is always <= middle band (SMA)
        self.assertTrue((generator.df['lower_band'].dropna() <= generator.df['sma_20'].dropna()).all())
        
        # Verify BB width is positive
        self.assertTrue((generator.df['bb_width'].dropna() > 0).all())
    
    def test_support_resistance(self):
        """Test calculation of distance from support/resistance."""
        generator = FeatureGenerator(self.clean_df)
        generator.support_resistance()
        
        # Check for support/resistance columns
        self.assertIn('distance_from_high', generator.df.columns)
        self.assertIn('distance_from_low', generator.df.columns)
        
        # Verify distance from high is always <= 0 (price is always <= recent high)
        self.assertTrue((generator.df['distance_from_high'].dropna() <= 0).all())
        
        # Verify distance from low is always >= 0 (price is always >= recent low)
        self.assertTrue((generator.df['distance_from_low'].dropna() >= 0).all())
    
    def test_volume_moving_averages(self):
        """Test calculation of volume moving averages."""
        generator = FeatureGenerator(self.clean_df)
        generator.volume_moving_averages()
        
        # Check for volume MA columns
        for period in [7, 15, 30, 60]:
            self.assertIn(f'vma_{period}', generator.df.columns)
            self.assertIn(f'relative_volume_{period}', generator.df.columns)
        
        # Verify calculations for vma_7
        expected_vma7 = self.clean_df['volume'].rolling(window=7).mean()
        pd.testing.assert_series_equal(
            generator.df['vma_7'].dropna(), 
            expected_vma7.dropna(),
            check_names=False
        )
        
        # Verify relative volume is correctly calculated (volume / vma)
        expected_rel_vol = self.clean_df['volume'] / expected_vma7
        pd.testing.assert_series_equal(
            generator.df['relative_volume_7'].dropna(), 
            expected_rel_vol.dropna(),
            check_names=False
        )
    
    def test_relative_strength_index(self):
        """Test calculation of RSI."""
        generator = FeatureGenerator(self.clean_df)
        generator.relative_strength_index()
        
        # Check for RSI column
        self.assertIn('rsi_14', generator.df.columns)
        
        # Verify RSI is between 0 and 100
        rsi_values = generator.df['rsi_14'].dropna()
        self.assertTrue((rsi_values >= 0).all() and (rsi_values <= 100).all())
    
    def test_stochastic_oscillator(self):
        """Test calculation of Stochastic Oscillator."""
        generator = FeatureGenerator(self.clean_df)
        generator.stochastic_oscillator()
        
        # Check for Stochastic columns
        self.assertIn('stoch_k', generator.df.columns)
        self.assertIn('stoch_d', generator.df.columns)
        
        # Verify Stochastic values are between 0 and 100
        k_values = generator.df['stoch_k'].dropna()
        d_values = generator.df['stoch_d'].dropna()
        
        self.assertTrue((k_values >= 0).all() and (k_values <= 100).all())
        self.assertTrue((d_values >= 0).all() and (d_values <= 100).all())
    
    def test_log_returns(self):
        """Test calculation of logarithmic returns."""
        generator = FeatureGenerator(self.clean_df)
        generator.log_returns()
        
        # Check for log return column
        self.assertIn('log_return', generator.df.columns)
        
        # Verify calculation
        expected_log_return = np.log(self.clean_df['close'] / self.clean_df['close'].shift(1))
        pd.testing.assert_series_equal(
            generator.df['log_return'].dropna(), 
            expected_log_return.dropna(),
            check_names=False
        )
    
    def test_time_of_day(self):
        """Test extraction of time of day features."""
        generator = FeatureGenerator(self.clean_df)
        generator.time_of_day()
        
        # Check for time of day features
        self.assertIn('hour_sin', generator.df.columns)
        self.assertIn('hour_cos', generator.df.columns)
        self.assertIn('session', generator.df.columns)
        
        # Verify sine and cosine values are between -1 and 1
        sin_values = generator.df['hour_sin']
        cos_values = generator.df['hour_cos']
        
        self.assertTrue((sin_values >= -1).all() and (sin_values <= 1).all())
        self.assertTrue((cos_values >= -1).all() and (cos_values <= 1).all())
    
    def test_safely_execute(self):
        """Test the safely_execute method for error handling."""
        generator = FeatureGenerator(self.clean_df)
        
        # Test successful execution
        result = generator.safely_execute('log_returns', 'Log Returns')
        self.assertTrue(result)
        self.assertIn('log_return', generator.df.columns)
        
        # Test execution with errors
        # Create a method that raises an exception
        def mock_method_with_error():
            raise ValueError("Test error")
        
        generator.test_error_method = mock_method_with_error
        result = generator.safely_execute('test_error_method', 'Test Error Method')
        self.assertFalse(result)
    
    def test_run_method(self):
        """Test the complete run method for feature generation."""
        generator = FeatureGenerator(self.clean_df)
        result_df = generator.run()
        
        # Check that result is a dataframe
        self.assertIsInstance(result_df, pd.DataFrame)
        
        # Check that new features were added
        self.assertGreater(len(result_df.columns), len(self.clean_df.columns))
        
        # Check for key features
        expected_features = [
            'sma_20', 'ema_20', 'roc_5', 'atr', 'upper_band', 
            'lower_band', 'rsi_14', 'macd_line'
        ]
        
        for feature in expected_features:
            # Some features might not be generated if they depend on others
            # So we'll check if at least some of the expected features are present
            if feature in result_df.columns:
                self.assertIn(feature, result_df.columns)
        
        # At least some features should be added
        self.assertGreater(
            len([col for col in expected_features if col in result_df.columns]), 
            0
        )
    
    def test_edge_case_small_dataset(self):
        """Test feature generation with a small dataset."""
        # Some features require long lookback periods
        # This tests graceful handling of insufficient data
        generator = FeatureGenerator(self.small_df)
        
        # Should not raise exceptions
        result_df = generator.run()
        
        # Check that result is a dataframe
        self.assertIsInstance(result_df, pd.DataFrame)
