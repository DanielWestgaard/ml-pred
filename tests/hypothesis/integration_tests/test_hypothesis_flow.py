import unittest
import tempfile
import os
import sys
import yaml
import pandas as pd
from unittest.mock import patch, MagicMock

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from data.pipelines.engineer_pipeline import EngineeringPipeline
from model.model_training import ModelTraining
from evaluation.evaluation import ModelEvaluation
from utils.shared_utils import handle_hypothesis_source
import tests.utils.test_hypothesis_utils as test_hypothesis_utils


class TestHypothesisIntegrationFlow(unittest.TestCase):
    """Integration tests for complete hypothesis flows"""
    
    def setUp(self):
        """Set up test environment with sample data and configs"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, 'data')
        self.model_dir = os.path.join(self.temp_dir, 'models')
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Create sample raw data
        self.sample_data = test_hypothesis_utils.create_sample_raw_data()
        
        self.raw_data_path = os.path.join(self.data_dir, 'raw_test_data.csv')
        self.sample_data.to_csv(self.raw_data_path, index=False)
        
        # Complete hypothesis config
        self.complete_config = test_hypothesis_utils.complete_config(raw_data_path = self.raw_data_path)
        
        # Data pipeline only config
        self.data_only_config = test_hypothesis_utils.data_pipeline_only_config(raw_data_path = self.raw_data_path)
        
    def tearDown(self):
        """Clean up temp files"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_configuration_parsing_integration(self):
        """Test that configuration is properly parsed and contains expected keys"""
        config_path = test_hypothesis_utils.create_config_file(config_dict = self.complete_config, temp_dir = self.temp_dir)
        
        result = handle_hypothesis_source(config_path)
        
        # Verify all expected configurations are parsed
        expected_keys = [
            'hypothesis_name', 'raw_data_source', 'features', 
            'model_type', 'parameters', 'evaluate_model', 'evaluation_type'
        ]
        
        for key in expected_keys:
            self.assertIn(key, result, f"Missing key: {key}")
        
        # Verify specific values
        self.assertEqual(result['hypothesis_name'], 'session_based_tod_anomalies')
        self.assertEqual(result['model_type'], 'xgboost')
        self.assertEqual(result['parameters']['n_estimators'], 10)
    
    @patch('data.pipelines.engineer_pipeline.FeatureGenerator')
    @patch('data.pipelines.engineer_pipeline.DataCleaner')
    @patch('data.pipelines.engineer_pipeline.DataValidator')
    def test_data_pipeline_hypothesis_integration(self, mock_validator, mock_cleaner, mock_generator):
        """Test data pipeline integration with hypothesis configuration"""
        config_path = test_hypothesis_utils.create_config_file(config_dict = self.data_only_config, temp_dir = self.temp_dir)
        
        # Mock the pipeline components
        mock_cleaner_instance = MagicMock()
        mock_cleaner_instance.run.return_value = self.sample_data
        mock_cleaner.return_value = mock_cleaner_instance
        
        mock_validator_instance = MagicMock()
        mock_validator_instance.run.return_value = {
            'validated_data': self.sample_data,
            'is_valid': True,
            'issues': []
        }
        mock_validator.return_value = mock_validator_instance
        
        mock_generator_instance = MagicMock()
        mock_generator_instance.run.return_value = self.sample_data
        mock_generator.return_value = mock_generator_instance
        
        # Test pipeline initialization with hypothesis
        pipeline = EngineeringPipeline(hypothesis_source=config_path)
        
        # Verify hypothesis configuration was loaded
        self.assertEqual(pipeline.hypothesis_name, 'test_hypothesis')
        self.assertEqual(pipeline.raw_data_path, self.raw_data_path)
        self.assertIsNotNone(pipeline.selected_features)
        self.assertIn('time_of_day', pipeline.selected_features)
    
    @patch('model.model_training.importlib.import_module')
    def test_model_training_hypothesis_integration(self, mock_import):
        """Test model training integration with hypothesis configuration"""
        config_path = test_hypothesis_utils.create_config_file(config_dict = self.complete_config, temp_dir = self.temp_dir)
        
        # Create processed data for model training
        processed_data = self.sample_data.copy()
        processed_data['target_session_based_tod_anomalies'] = [0.1 * i for i in range(1000)]
        processed_path = os.path.join(self.data_dir, 'processed_test_data.csv')
        processed_data.to_csv(processed_path, index=False)
        
        # Update config to point to processed data
        config_with_processed = self.complete_config.copy()
        config_with_processed['model_training']['processed_data'] = processed_path
        config_path = test_hypothesis_utils.create_config_file(config_dict = config_with_processed, temp_dir = self.temp_dir)
        
        # Mock the model class
        mock_model_class = MagicMock()
        mock_model_instance = MagicMock()
        mock_trained_model = MagicMock()
        mock_trained_model.type = 'xgboost'
        
        mock_model_instance.train.return_value = mock_trained_model
        mock_model_class.return_value = mock_model_instance
        
        mock_module = MagicMock()
        mock_module.XgboostModel = mock_model_class
        mock_import.return_value = mock_module
        
        # Test model training initialization with hypothesis
        with patch('model.model_training.ModelTraining.save_model'):
            trainer = ModelTraining(hypothesis_source=config_path)
            
            # Verify hypothesis configuration was loaded
            self.assertEqual(trainer.model_type, 'xgboost')
            self.assertEqual(trainer.dataset_path, processed_path)
    
    def test_missing_hypothesis_name_integration(self):
        """Test integration behavior when hypothesis name is missing"""
        invalid_config = {
            'data_pipeline': {
                'raw_source': self.raw_data_path
            }
        }
        config_path = test_hypothesis_utils.create_config_file(config_dict = invalid_config, temp_dir = self.temp_dir)
        
        # Should exit when trying to initialize components
        with self.assertRaises(SystemExit):
            EngineeringPipeline(hypothesis_source=config_path)
    
    def test_configuration_key_precedence(self):
        """Test that hypothesis configuration takes precedence over direct parameters"""
        config_path = test_hypothesis_utils.create_config_file(config_dict = self.complete_config, temp_dir = self.temp_dir)
        
        # Create pipeline with both direct dataset and hypothesis source
        pipeline = EngineeringPipeline(
            raw_dataset='should_be_ignored.csv',
            hypothesis_source=config_path
        )
        
        # Hypothesis source should take precedence
        self.assertEqual(pipeline.raw_data_path, self.raw_data_path)
        self.assertEqual(pipeline.hypothesis_name, 'session_based_tod_anomalies')
    
    def test_data_path_resolution_integration(self):
        """Test that data paths are correctly resolved between raw and processed"""
        # Test with only raw data source
        config = {
            'hypothesis': {'name': 'test'},
            'data_pipeline': {'raw_source': 'storage/data/raw/test.csv'},
            'model_training': {'type': 'xgboost'}
        }
        config_path = test_hypothesis_utils.create_config_file(config_dict = config, temp_dir = self.temp_dir)
        
        result = handle_hypothesis_source(config_path)
        
        self.assertEqual(result['raw_data_source'], 'storage/data/raw/test.csv')
        self.assertNotIn('processed_data', result)
    
    def test_evaluation_configuration_integration(self):
        """Test evaluation configuration parsing and integration"""
        # Test 'new' evaluation type
        new_eval_config = {
            'hypothesis': {'name': 'test'},
            'evaluate_model': {'type': 'new'}
        }
        config_path = test_hypothesis_utils.create_config_file(config_dict = new_eval_config, temp_dir = self.temp_dir)
        result = handle_hypothesis_source(config_path)
        
        self.assertEqual(result['evaluation_type'], 'new')
        self.assertNotIn('model_folder', result)
        
        # Test 'stored' evaluation type
        stored_eval_config = {
            'hypothesis': {'name': 'test'},
            'evaluate_model': {
                'type': 'stored',
                'model_folder': 'storage/models/test_model'
            }
        }
        config_path = test_hypothesis_utils.create_config_file(config_dict = stored_eval_config, temp_dir = self.temp_dir, filename = 'stored_config.yml')
        result = handle_hypothesis_source(config_path)
        
        self.assertEqual(result['evaluation_type'], 'stored')
        self.assertEqual(result['model_folder'], 'storage/models/test_model')
    
    def test_feature_configuration_integration(self):
        """Test that feature configurations are properly handled"""
        feature_config = {
            'hypothesis': {'name': 'feature_test'},
            'data_pipeline': {
                'raw_source': self.raw_data_path,
                'selected_features': ['hour', 'minute', 'london_session', 'ny_session']
            }
        }
        config_path = test_hypothesis_utils.create_config_file(config_dict = feature_config, temp_dir = self.temp_dir)
        
        result = handle_hypothesis_source(config_path)
        
        self.assertIn('features', result)
        features = result['features']
        self.assertIn('time_of_day', features)
        self.assertIn('volatility_regimes', features)
        self.assertIn('session_timing_features', features)
        
        # Test that specific feature parameters are preserved
        self.assertEqual(features['time_of_day'], ['hour', 'minute'])
        self.assertEqual(features['volatility_regimes'], [])
        self.assertEqual(features['session_timing_features'], ['london_session', 'ny_session'])


class TestHypothesisFlowErrorHandling(unittest.TestCase):
    """Test error handling in hypothesis flows"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_corrupted_config_handling(self):
        """Test handling of corrupted configuration files"""
        corrupted_config_path = os.path.join(self.temp_dir, 'corrupted.yml')
        with open(corrupted_config_path, 'w') as f:
            f.write('hypothesis:\n  name: test\n[invalid yaml structure')
        
        with self.assertRaises(SystemExit):
            handle_hypothesis_source(corrupted_config_path)
    
    def test_empty_config_handling(self):
        """Test handling of empty configuration files"""
        empty_config_path = os.path.join(self.temp_dir, 'empty.yml')
        with open(empty_config_path, 'w') as f:
            f.write('')
        
        with self.assertRaises(SystemExit):
            handle_hypothesis_source(empty_config_path)
    
    def test_partial_config_robustness(self):
        """Test that partial configurations don't break the system"""
        partial_configs = [
            {'hypothesis': {'name': 'test1'}},  # Only hypothesis
            {'hypothesis': {'name': 'test2'}, 'data_pipeline': {}},  # Empty data pipeline
            {'hypothesis': {'name': 'test3'}, 'model_training': {}},  # Empty model training
        ]
        
        for i, config in enumerate(partial_configs):
            config_path = os.path.join(self.temp_dir, f'partial_{i}.yml')
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            # Should not raise exceptions for partial configs
            result = handle_hypothesis_source(config_path)
            self.assertIn('hypothesis_name', result)


if __name__ == '__main__':
    unittest.main()