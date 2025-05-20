
import unittest
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

# Import test utilities
from tests.utils.test_data_utils import create_sample_ohlcv_data

# Import the module to test
from data.processing.validation import DataValidator


class TestDataValidator(unittest.TestCase):
    
    def setUp(self):
        """Set up test data."""
        # Create clean sample data
        self.clean_df = create_sample_ohlcv_data(rows=50, with_issues=False)
        
        # Create data with issues for testing validation failures
        self.df_with_issues = create_sample_ohlcv_data(rows=50, with_issues=True)
        
        # Set date as index for DataValidator
        self.clean_df_indexed = self.clean_df.copy()
        self.clean_df_indexed.set_index('date', inplace=True)
        
        self.issues_df_indexed = self.df_with_issues.copy()
        self.issues_df_indexed.set_index('date', inplace=True)
    
    def test_initialization(self):
        """Test the initialization of DataValidator."""
        validator = DataValidator(self.clean_df)
        self.assertEqual(len(validator.df), len(self.clean_df))
        self.assertEqual(len(validator.validation_issues), 0)
    
    def test_validate_ohlc_relationships_clean(self):
        """Test OHLC validation with clean data."""
        validator = DataValidator(self.clean_df_indexed)
        validator.validate_ohlc_relationships()
        
        # Should not have any issues
        self.assertEqual(len(validator.validation_issues), 0)
    
    def test_validate_ohlc_relationships_issues(self):
        """Test OHLC validation with problematic data."""
        # Create specific issues in OHLC relationships
        df = self.clean_df_indexed.copy()
        
        # Make high < open for a few rows
        issue_indices = [5, 15, 25]
        for idx in issue_indices:
            df.loc[df.index[idx], 'high'] = df.loc[df.index[idx], 'open'] * 0.9
        
        validator = DataValidator(df)
        validator.validate_ohlc_relationships()
        
        # Should have identified the issues
        self.assertGreater(len(validator.validation_issues), 0)
        
        # Check if first issue matches expected type
        self.assertEqual(validator.validation_issues[0]['type'], 'OHLC Relationship')
    
    def test_validate_timestamps_clean(self):
        """Test timestamp validation with clean data."""
        validator = DataValidator(self.clean_df_indexed)
        validator.validate_timestamps()
        
        # Should not have any issues
        self.assertEqual(len(validator.validation_issues), 0)
    
    def test_validate_timestamps_duplicates(self):
        """Test timestamp validation with duplicate timestamps."""
        # Create duplicate timestamps
        df = self.clean_df_indexed.copy()
        
        # Convert index to a modifiable list
        index_list = df.index.tolist()
        
        # Create duplicate timestamp
        duplicate_idx = 10
        index_list[duplicate_idx] = index_list[duplicate_idx-1]
        # Assign the modified list back as the new index
        df.index = index_list
        df.index = pd.DatetimeIndex(df.index)  # Convert back to DatetimeIndex
        
        validator = DataValidator(df)
        validator.validate_timestamps()
        
        # Should have identified the duplicate timestamps
        self.assertGreater(len(validator.validation_issues), 0)
        self.assertEqual(validator.validation_issues[0]['type'], 'Duplicate Timestamps')
    
    def test_validate_timestamps_future(self):
        """Test timestamp validation with future timestamps."""
        # Create future timestamps
        df = self.clean_df_indexed.copy()

        # Convert index to a modifiable list
        index_list = df.index.tolist()

        # Make future timestamps
        future_idx = 20
        future_date = datetime.now(timezone.utc) + timedelta(days=10)
        index_list[future_idx] = future_date  # Modify the list, not the index directly

        # Assign the modified list back as the new index
        df.index = index_list
        df.index = pd.DatetimeIndex(df.index)  # Convert back to DatetimeIndex
        
        validator = DataValidator(df)
        validator.validate_timestamps()
        
        # Should have identified future timestamps
        timestamp_issues = [issue for issue in validator.validation_issues 
                            if issue['type'] == 'Future Timestamps']
        self.assertGreater(len(timestamp_issues), 0)
    
    def test_validate_volume_negative(self):
        """Test volume validation with negative volume values."""
        # Create negative volume
        df = self.clean_df_indexed.copy()
        
        # Make negative volume
        negative_indices = [7, 17, 27]
        for idx in negative_indices:
            df.loc[df.index[idx], 'volume'] = -100
        
        validator = DataValidator(df)
        validator.validate_volume()
        
        # Should have identified negative volume
        volume_issues = [issue for issue in validator.validation_issues 
                         if issue['type'] == 'Invalid Volume']
        self.assertGreater(len(volume_issues), 0)
    
    def test_validate_price_movement_extreme(self):
        """Test price movement validation with extreme price changes."""
        # Create extreme price changes
        df = self.clean_df_indexed.copy()
        
        # Make extreme price changes (>20%)
        extreme_idx = 30
        df.loc[df.index[extreme_idx], 'close'] = df.loc[df.index[extreme_idx-1], 'close'] * 1.3  # 30% increase
        
        validator = DataValidator(df)
        validator.validate_price_movement()
        
        # Should have identified extreme price movements
        price_issues = [issue for issue in validator.validation_issues 
                        if issue['type'] == 'Extreme Price Movement']
        self.assertGreater(len(price_issues), 0)
    
    def test_validate_data_completeness_missing_columns(self):
        """Test data completeness validation with missing required columns."""
        # Create data missing a required column
        df = self.clean_df_indexed.copy()
        df = df.drop('open', axis=1)  # Remove 'open' column
        
        validator = DataValidator(df)
        validator.validate_data_completeness()
        
        # Should have identified missing required column
        completeness_issues = [issue for issue in validator.validation_issues 
                              if issue['type'] == 'Missing Required Columns']
        self.assertGreater(len(completeness_issues), 0)
        
        # Check that 'open' is mentioned in the description
        self.assertIn('open', completeness_issues[0]['description'])
    
    def test_validate_data_completeness_insufficient_data(self):
        """Test data completeness validation with insufficient data points."""
        # Create data with few rows
        df = create_sample_ohlcv_data(rows=10, with_issues=False)
        df.set_index('date', inplace=True)
        
        # Artificially reduce it further to trigger the validation
        small_df = df.iloc[:20]
        
        validator = DataValidator(small_df)
        validator.validate_data_completeness()
        
        # If there's a minimum threshold check less than 30 rows
        # should identify insufficient data
        if len(small_df) < 30:
            completeness_issues = [issue for issue in validator.validation_issues 
                                if issue['type'] == 'Insufficient Data']
            self.assertGreater(len(completeness_issues), 0)
    
    def test_run_method_clean_data(self):
        """Test the run method with clean data."""
        validator = DataValidator(self.clean_df_indexed)
        result = validator.run()
        
        # Should indicate valid data
        self.assertTrue(result['is_valid'])
        self.assertEqual(len(result['issues']), 0)
    
    def test_run_method_issues_data(self):
        """Test the run method with data containing issues."""
        # Create data with various issues
        df = self.issues_df_indexed.copy()
        
        validator = DataValidator(df)
        result = validator.run()
        
        # Should indicate invalid data with issues
        self.assertFalse(result['is_valid'])
        self.assertGreater(len(result['issues']), 0)
