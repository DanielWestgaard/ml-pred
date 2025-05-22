import unittest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import patch, MagicMock

# Import test utilities
from tests.utils.test_data_utils import create_sample_ohlcv_data, save_temp_csv, clean_temp_file
# Import the modules to test
from data.processing.cleaning import DataCleaner
from data.processing.validation import DataValidator


class TestCleaningValidationIntegration(unittest.TestCase):
    
    def setUp(self):
        """Set up test data."""
        # Create sample data with issues
        self.df_with_issues = create_sample_ohlcv_data(rows=100, with_issues=True)
        self.temp_file_path = save_temp_csv(self.df_with_issues)
        
        # Create clean sample data for comparison
        self.clean_df = create_sample_ohlcv_data(rows=100, with_issues=False)
    
    def tearDown(self):
        """Clean up temporary files."""
        clean_temp_file(self.temp_file_path)
    
    def test_cleaning_validation_integration(self):
        """Test integration between DataCleaner and DataValidator."""
        # Step 1: Clean the data
        cleaner = DataCleaner(self.temp_file_path)
        cleaned_df = cleaner.run()
        
        # Step 2: Validate the cleaned data
        validator = DataValidator(cleaned_df)
        validation_result = validator.run()
        
        # Verify that cleaning resolves validation issues
        self.assertTrue(validation_result['is_valid'], 
                       f"Cleaned data should be valid but found issues: {validation_result['issues']}")
        self.assertEqual(len(validation_result['issues']), 0, 
                        "There should be no validation issues after cleaning")
    
    def test_validation_before_cleaning(self):
        """Test validation on uncleaned data."""
        # First validate without cleaning
        df_with_issues = pd.read_csv(self.temp_file_path)
        df_with_issues.set_index('date', inplace=True)
        
        validator = DataValidator(df_with_issues)
        validation_result = validator.run()
        
        # Verify that uncleaned data has validation issues
        self.assertFalse(validation_result['is_valid'], 
                        "Uncleaned data should have validation issues")
        self.assertGreater(len(validation_result['issues']), 0, 
                          "There should be validation issues before cleaning")
        
        # Then clean and validate again
        cleaner = DataCleaner(df_with_issues)
        cleaned_df = cleaner.run()
        
        validator = DataValidator(cleaned_df)
        validation_result = validator.run()
        
        # Verify that cleaning resolves validation issues
        self.assertTrue(validation_result['is_valid'], 
                       "Cleaned data should be valid")
    
    def test_cleaning_edge_cases(self):
        """Test cleaning and validation with edge cases."""
        # Create extreme data quality issues
        df = self.df_with_issues.copy()
        
        # Add extreme values
        df.loc[10, 'high'] = df.loc[10, 'close'] * 100  # Very high spike
        df.loc[20, 'low'] = df.loc[20, 'close'] * 0.01  # Very low spike
        df.loc[30, 'volume'] = -1000  # Negative volume
        
        # Create invalid OHLC relationships
        df.loc[40, 'high'] = df.loc[40, 'low'] * 0.5  # High below low
        
        # Save to temporary file
        temp_extreme_path = save_temp_csv(df)
        
        try:
            # Clean and validate
            cleaner = DataCleaner(temp_extreme_path)
            cleaned_df = cleaner.run()
            
            validator = DataValidator(cleaned_df)
            validation_result = validator.run()
            
            # Even with extreme issues, cleaning should make data valid
            self.assertTrue(validation_result['is_valid'], 
                           "Cleaned data should be valid even with extreme initial issues")
            
        finally:
            # Clean up
            clean_temp_file(temp_extreme_path)
    
    def test_missing_column_handling(self):
        """Test how cleaning and validation handle missing columns."""
        # Create data missing a required column
        df = self.df_with_issues.copy()
        df = df.drop('low', axis=1)  # Remove 'low' column
        
        # Save to temporary file
        temp_missing_col_path = save_temp_csv(df)
        
        try:
            # Clean and validate
            cleaner = DataCleaner(temp_missing_col_path)
            cleaned_df = cleaner.run()
            
            validator = DataValidator(cleaned_df)
            validation_result = validator.run()
            
            # Should identify missing required column
            self.assertFalse(validation_result['is_valid'], 
                            "Data missing required column should not be valid")
            
            missing_col_issues = [issue for issue in validation_result['issues'] 
                                if issue['type'] == 'Missing Required Columns']
            self.assertGreater(len(missing_col_issues), 0, 
                              "Validation should identify missing required column")
            
        finally:
            # Clean up
            clean_temp_file(temp_missing_col_path)
