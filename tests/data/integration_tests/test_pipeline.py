import unittest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import patch, MagicMock

# Import test utilities
from tests.utils.test_data_utils import create_sample_ohlcv_data, save_temp_csv, clean_temp_file, mock_model_utils
# Import the modules to test
import config.config as config
from data.pipelines.engineer_pipeline import EngineeringPipeline
from data.processing.cleaning import DataCleaner
from data.processing.validation import DataValidator
from data.features.generation import FeatureGenerator
from data.features.transformation import FeatureTransformer
from data.features.selection import FeatureSelector


class TestEngineeringPipeline(unittest.TestCase):
    
    def setUp(self):
        """Set up test data."""
        # Create mock for model_utils that might be imported
        self.mock_module = mock_model_utils()
        
        # Create sample data with issues
        self.df_with_issues = create_sample_ohlcv_data(rows=250, with_issues=True)
        self.temp_file_path = save_temp_csv(self.df_with_issues)
        
        # Create temporary directory for output
        self.temp_output_dir = os.path.join(os.getcwd(), 'temp_pipeline_output')
        if not os.path.exists(self.temp_output_dir):
            os.makedirs(self.temp_output_dir)
        
        # Create temporary directory for feature selection plots
        self.fe_sel_dir = os.path.join(os.getcwd(), 'temp_fe_sel')
        if not os.path.exists(self.fe_sel_dir):
            os.makedirs(self.fe_sel_dir)
    
    def tearDown(self):
        """Clean up temporary files and directories."""
        clean_temp_file(self.temp_file_path)
        
        # Remove temporary directories
        import shutil
        if os.path.exists(self.temp_output_dir):
            shutil.rmtree(self.temp_output_dir)
        if os.path.exists(self.fe_sel_dir):
            shutil.rmtree(self.fe_sel_dir)
    
    def _create_mock_ohlcv_df(self, additional_data=None):
        """Helper method to create mock DataFrame with required OHLCV columns."""
        base_data = {
            'open': [100.0],
            'high': [105.0], 
            'low': [98.0],
            'close': [102.0],
            'volume': [1000]
        }
        if additional_data:
            base_data.update(additional_data)
        return pd.DataFrame(base_data)
    
    @patch('config.config.FE_SEL_BASE_DIR', MagicMock(return_value='temp_fe_sel'))
    @patch('os.makedirs')  # Mock directory creation
    @patch('utils.data_utils.save_processed_file')
    def test_full_pipeline(self, mock_save, mock_makedirs):
        """Test the full engineering pipeline from raw data to selected features."""
        # Patch config.FE_SEL_BASE_DIR to use our temporary directory
        config.FE_SEL_BASE_DIR = self.fe_sel_dir
        
        # Initialize pipeline with raw data
        pipeline = EngineeringPipeline(raw_dataset=self.temp_file_path, 
                                     output_path=self.temp_output_dir)
        
        # Verify initialization
        self.assertEqual(pipeline.raw_data_path, self.temp_file_path)
        self.assertEqual(pipeline.output_path, self.temp_output_dir)
        
        # Run the pipeline
        result_df = pipeline.run()
        
        # Verify the result
        self.assertIsInstance(result_df, pd.DataFrame)
        
        # Check that data was processed (should be fewer rows due to NaN handling)
        self.assertLessEqual(len(result_df), len(self.df_with_issues))
        
        # Verify that save_processed_file was called
        mock_save.assert_called_once()
    
    @patch('config.config.FE_SEL_BASE_DIR', MagicMock(return_value='temp_fe_sel'))
    @patch('os.makedirs')  # Mock directory creation
    def test_pipeline_component_integration(self, mock_makedirs):
        """Test integration of all pipeline components with mocked methods."""
        # Patch config.FE_SEL_BASE_DIR to use our temporary directory
        config.FE_SEL_BASE_DIR = self.fe_sel_dir
        
        # Create pipeline
        pipeline = EngineeringPipeline(raw_dataset=self.temp_file_path)
        
        # Create mock DataFrames with required OHLCV columns
        cleaned_df = self._create_mock_ohlcv_df({'cleaned': ['data']})
        validated_df = self._create_mock_ohlcv_df({'validated': ['data']})
        features_df = self._create_mock_ohlcv_df({'features': ['data']})
        transformed_df = self._create_mock_ohlcv_df({'transformed': ['data']})
        selected_df = self._create_mock_ohlcv_df({'selected': ['data']})
        
        # Mock component return values
        with patch.object(DataCleaner, 'run', return_value=cleaned_df), \
             patch.object(DataValidator, 'run', return_value={
                 'is_valid': True, 
                 'issues': [],
                 'validated_data': validated_df
             }), \
             patch.object(FeatureGenerator, 'run', return_value=features_df), \
             patch.object(FeatureTransformer, 'run', return_value=transformed_df), \
             patch.object(FeatureSelector, 'run', return_value=selected_df), \
             patch('utils.data_utils.save_processed_file') as mock_save, \
             patch('utils.data_utils.check_validation', return_value=None) as mock_check:
            
            # Run the pipeline
            result = pipeline.run()
            
            # Verify the result
            self.assertTrue(result.equals(selected_df))
            
            # Verify that all components were called
            mock_save.assert_called_once()
            mock_check.assert_called_once()
    
    @patch('config.config.FE_SEL_BASE_DIR', MagicMock(return_value='temp_fe_sel'))
    @patch('os.makedirs')  # Mock directory creation
    def test_pipeline_with_validation_failure(self, mock_makedirs):
        """Test pipeline behavior when validation fails."""
        # Patch config.FE_SEL_BASE_DIR to use our temporary directory
        config.FE_SEL_BASE_DIR = self.fe_sel_dir
        
        # Create pipeline
        pipeline = EngineeringPipeline(raw_dataset=self.temp_file_path)
        
        # Create mock DataFrame with required OHLCV columns
        cleaned_df = self._create_mock_ohlcv_df({'cleaned': ['data']})
        
        # Validation fails
        validation_result = {
            'is_valid': False,
            'issues': [{'type': 'Mock Issue', 'description': 'Test issue'}],
            'validated_data': cleaned_df  # Return the cleaned data anyway
        }
        
        # Mock component return values
        with patch.object(DataCleaner, 'run', return_value=cleaned_df), \
             patch.object(DataValidator, 'run', return_value=validation_result), \
             patch.object(FeatureGenerator, 'run', return_value=cleaned_df), \
             patch.object(FeatureTransformer, 'run', return_value=cleaned_df), \
             patch.object(FeatureSelector, 'run', return_value=cleaned_df), \
             patch('utils.data_utils.save_processed_file') as mock_save, \
             patch('utils.data_utils.check_validation', return_value=None) as mock_check:
            
            # Run the pipeline
            result = pipeline.run()
            
            # Pipeline should continue despite validation issues
            self.assertTrue(result.equals(cleaned_df))
            
            # Verify that validation was checked
            mock_check.assert_called_once_with(False, [{'type': 'Mock Issue', 'description': 'Test issue'}])
            
            # Verify that data is still saved
            mock_save.assert_called_once()
    
    @patch('config.config.FE_SEL_BASE_DIR', MagicMock(return_value='temp_fe_sel'))
    @patch('os.makedirs')  # Mock directory creation
    def test_pipeline_error_handling(self, mock_makedirs):
        """Test pipeline error handling."""
        # Patch config.FE_SEL_BASE_DIR to use our temporary directory
        config.FE_SEL_BASE_DIR = self.fe_sel_dir
        
        # Create pipeline
        pipeline = EngineeringPipeline(raw_dataset=self.temp_file_path)
        
        # Make DataCleaner.run raise an exception
        with patch.object(DataCleaner, 'run', side_effect=Exception('Test error')):
            # Pipeline should raise the exception
            with self.assertRaises(Exception):
                pipeline.run()
        
        # Test error handling in saving
        with patch.object(DataCleaner, 'run', return_value=pipeline.df), \
             patch.object(DataValidator, 'run', return_value={
                 'is_valid': True, 
                 'issues': [],
                 'validated_data': pipeline.df
             }), \
             patch.object(FeatureGenerator, 'run', return_value=pipeline.df), \
             patch.object(FeatureTransformer, 'run', return_value=pipeline.df), \
             patch.object(FeatureSelector, 'run', return_value=pipeline.df), \
             patch('utils.data_utils.save_processed_file', side_effect=Exception('Save error')), \
             patch('utils.data_utils.check_validation', return_value=None), \
             patch('logging.error') as mock_error:
            
            # Run the pipeline - this should handle the save error
            result = pipeline.run()
            
            # Verify error was logged
            mock_error.assert_called()
