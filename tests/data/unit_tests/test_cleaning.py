import unittest
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

# Import test utilities
from tests.utils.test_data_utils import create_sample_ohlcv_data, save_temp_csv, clean_temp_file

# Import the module to test
from data.processing.cleaning import DataCleaner


class TestDataCleaner(unittest.TestCase):
    
    def setUp(self):
        """Set up test data."""
        # Create sample data with issues
        self.df_with_issues = create_sample_ohlcv_data(rows=50, with_issues=True)
        self.temp_file_path = save_temp_csv(self.df_with_issues)
        
        # Create clean sample data
        self.clean_df = create_sample_ohlcv_data(rows=50, with_issues=False)
    
    def tearDown(self):
        """Clean up temporary files."""
        clean_temp_file(self.temp_file_path)
    
    def test_init_with_dataframe(self):
        """Test initializing with a DataFrame."""
        cleaner = DataCleaner(self.df_with_issues)
        self.assertEqual(len(cleaner.df), len(self.df_with_issues))
        self.assertEqual(len(cleaner.df.columns), len(self.df_with_issues.columns))
    
    def test_init_with_filepath(self):
        """Test initializing with a file path."""
        cleaner = DataCleaner(self.temp_file_path)
        self.assertEqual(len(cleaner.df), len(self.df_with_issues))
        self.assertEqual(len(cleaner.df.columns), len(self.df_with_issues.columns))
    
    def test_handle_missing_values(self):
        """Test handling of missing values."""
        # Count NaN values before cleaning
        cleaner = DataCleaner(self.df_with_issues)
        nan_count_before = cleaner.df.isna().sum().sum()

        # Ensure we have NaNs to test with
        self.assertGreater(nan_count_before, 0, "Test data should have NaN values for this test")
        
        # Run the missing values handler
        cleaner._handle_missing_values()
        
        # Check that NaNs are gone
        nan_count_after = cleaner.df.isna().sum().sum()
        self.assertEqual(nan_count_after, 0, "All NaN values should be handled")
        
        # Check that OHLC values are reasonable (not all identical)
        close_values = cleaner.df['close'].unique()
        self.assertGreater(len(close_values), 10, "Close values should vary after missing value handling")
    
    def test_remove_duplicates(self):
        """Test removal of duplicate dates."""
        # Count duplicate dates before cleaning
        cleaner = DataCleaner(self.df_with_issues.copy())
        duplicate_count_before = cleaner.df['date'].duplicated().sum()
        
        # Ensure we have duplicates to test with
        self.assertGreater(duplicate_count_before, 0, "Test data should have duplicate dates for this test")
        
        # Run the duplicates handler
        cleaner._remove_duplicates()
        
        # Check that duplicates are gone
        duplicate_count_after = cleaner.df['date'].duplicated().sum()
        self.assertEqual(duplicate_count_after, 0, "All duplicate dates should be removed")
    
    def test_timestamp_alignment(self):
        """Test timestamp alignment and timezone handling."""
        # Create data with mixed timezones
        df = self.df_with_issues.copy()
        
        # Convert date column to datetime without timezone
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        
        cleaner = DataCleaner(df)
        
        # FIXED: Remove duplicates first, as the cleaning pipeline does
        cleaner._remove_duplicates()
        
        # Now test timestamp alignment
        cleaner._timestamp_alignment()
        
        # Check that we now have a datetime index
        self.assertTrue(isinstance(cleaner.df.index, pd.DatetimeIndex), "Index should be DatetimeIndex")
        
        # Check that the index is sorted
        self.assertTrue(cleaner.df.index.is_monotonic_increasing, "Index should be monotonically increasing")
        
        # Check that there are no duplicate index values
        self.assertEqual(cleaner.df.index.duplicated().sum(), 0, "No duplicate index values should remain")
    
    def test_handle_outliers(self):
        """Test outlier handling in OHLCV data."""
        # Create data with price outliers
        df = self.clean_df.copy()
        
        # Add extreme outliers that should definitely be caught
        extreme_indices = [10, 20, 30]
        original_close_values = []
        original_volume_values = []
        
        for idx in extreme_indices:
            original_close_values.append(df.loc[idx, 'close'])
            original_volume_values.append(df.loc[idx, 'volume'])
            
            df.loc[idx, 'close'] = df.loc[idx, 'close'] * 10  # 10x higher
            df.loc[idx, 'volume'] = df.loc[idx, 'volume'] * 100  # 100x higher
        
        cleaner = DataCleaner(df)
        
        # Get extreme values before outlier handling
        extreme_close = df.loc[extreme_indices, 'close'].values
        extreme_volume = df.loc[extreme_indices, 'volume'].values
        
        # Handle outliers
        cleaner._handle_outliers()
        
        # Check that extreme values are capped
        for i, idx in enumerate(extreme_indices):
            # Close price should be capped (should be less than the extreme value)
            self.assertLess(cleaner.df.loc[idx, 'close'], extreme_close[i], 
                           f"Close price at index {idx} should be capped from {extreme_close[i]} to {cleaner.df.loc[idx, 'close']}")
            
            # Volume should be capped (should be less than the extreme value)
            self.assertLess(cleaner.df.loc[idx, 'volume'], extreme_volume[i],
                           f"Volume at index {idx} should be capped from {extreme_volume[i]} to {cleaner.df.loc[idx, 'volume']}")
            
            # But values should still be reasonable (not too close to original)
            # Close should be somewhere between original and extreme
            self.assertGreater(cleaner.df.loc[idx, 'close'], original_close_values[i] * 1.1,
                              "Capped close should be higher than original value")
            
            # Volume should be somewhere between original and extreme  
            self.assertGreater(cleaner.df.loc[idx, 'volume'], original_volume_values[i] * 1.1,
                              "Capped volume should be higher than original value")
    
    def test_datatype_consistency(self):
        """Test datatype consistency checks."""
        # Create data with mixed types
        df = self.clean_df.copy()
        
        # Mix data types - use pandas' object type to avoid warnings
        df['close'] = df['close'].astype('object')  # Convert column to object type first
        df.loc[10, 'close'] = str(df.loc[10, 'close'])  # Then set string value
        
        df['volume'] = df['volume'].astype('object')  # Same for volume
        df.loc[15, 'volume'] = float(df.loc[15, 'volume'])  # Convert to float
        
        cleaner = DataCleaner(df)
        cleaner._datatype_consistency()
        
        # Check that types are corrected
        self.assertTrue(np.issubdtype(cleaner.df['close'].dtype, np.number),
                    "Close column should be numeric")
        self.assertTrue(np.issubdtype(cleaner.df['volume'].dtype, np.integer),
                    "Volume column should be integer")
    
    def test_ohlc_validity(self):
        """Test validation of OHLC relationships."""
        # Create data with invalid OHLC relationships
        df = self.clean_df.copy()
        
        # Create invalid OHLC relationships
        invalid_indices = [5, 15, 25]
        for idx in invalid_indices:
            df.loc[idx, 'high'] = df.loc[idx, 'close'] * 0.9  # High below close
            df.loc[idx, 'low'] = df.loc[idx, 'open'] * 1.1  # Low above open
        
        cleaner = DataCleaner(df)
        cleaner._ohlc_validity()
        
        # Check that OHLC relationships are fixed
        for idx in invalid_indices:
            self.assertGreaterEqual(cleaner.df.loc[idx, 'high'], cleaner.df.loc[idx, 'close'])
            self.assertLessEqual(cleaner.df.loc[idx, 'low'], cleaner.df.loc[idx, 'open'])
    
    def test_run_method(self):
        """Test the complete run method with all cleaning steps."""
        # Create a cleaner with issues
        cleaner = DataCleaner(self.df_with_issues.copy())
        
        # Run the full cleaning process
        cleaned_df = cleaner.run()
        
        # Verify the cleaned dataframe
        self.assertEqual(cleaned_df.isna().sum().sum(), 0, "No NaN values should remain")
        self.assertEqual(cleaned_df.index.duplicated().sum(), 0, "No duplicate indices should remain")
        
        # Check OHLC relationships
        self.assertTrue(all(cleaned_df['high'] >= cleaned_df['open']), "High should be >= Open")
        self.assertTrue(all(cleaned_df['high'] >= cleaned_df['close']), "High should be >= Close")
        self.assertTrue(all(cleaned_df['low'] <= cleaned_df['open']), "Low should be <= Open")
        self.assertTrue(all(cleaned_df['low'] <= cleaned_df['close']), "Low should be <= Close")
        
        # Volume should be non-negative
        self.assertTrue(all(cleaned_df['volume'] >= 0), "Volume should be non-negative")
        
        # Data should have datetime index
        self.assertTrue(isinstance(cleaned_df.index, pd.DatetimeIndex), "Index should be DatetimeIndex")
        
        # Index should be sorted
        self.assertTrue(cleaned_df.index.is_monotonic_increasing, "Index should be monotonically increasing")

if __name__ == '__main__':
    unittest.main()