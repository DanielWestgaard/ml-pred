import unittest
import tempfile
import os
import yaml
from unittest.mock import patch
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from utils.shared_utils import handle_hypothesis_source
import tests.utils.test_hypothesis_utils as test_hypothesis_utils


class TestHypothesisConfiguration(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures with different config scenarios"""
        self.temp_dir = tempfile.mkdtemp()
        
        self.valid_config = test_hypothesis_utils.valid_config()
        
        self.minimal_config = test_hypothesis_utils.minimal_config()
        
        self.invalid_config = test_hypothesis_utils.invalid_config()
        
    def tearDown(self):
        """Clean up temp files"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_temp_config(self, config_dict):
        """Helper to create temporary config file"""
        config_path = os.path.join(self.temp_dir, 'config.yml')
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)
        return config_path
    
    def test_valid_complete_config(self):
        """Test loading a complete valid configuration"""
        config_path = self._create_temp_config(self.valid_config)
        
        result = handle_hypothesis_source(config_path)
        
        self.assertEqual(result['hypothesis_name'], 'test_hypothesis')
        self.assertEqual(result['raw_data_source'], 'storage/data/test/raw_data.csv')
        self.assertEqual(result['model_type'], 'xgboost')
        self.assertEqual(result['processed_data'], 'storage/data/test/processed_data.csv')
        self.assertIn('time_of_day', result['features'])
        self.assertIn('volatility_regimes', result['features'])
        self.assertEqual(result['parameters']['n_estimators'], 100)
        self.assertEqual(result['evaluation_type'], 'new')
        self.assertEqual(result['model_folder'], 'storage/models/test_model')
    
    def test_minimal_config(self):
        """Test loading minimal configuration with only hypothesis name"""
        config_path = self._create_temp_config(self.minimal_config)
        
        result = handle_hypothesis_source(config_path)
        
        self.assertEqual(result['hypothesis_name'], 'minimal_test')
        self.assertNotIn('raw_data_source', result)
        self.assertNotIn('model_type', result)
        self.assertNotIn('features', result)
    
    def test_missing_hypothesis_name(self):
        """Test that missing hypothesis name causes system exit"""
        config_path = self._create_temp_config(self.invalid_config)
        
        with self.assertRaises(SystemExit):
            handle_hypothesis_source(config_path)
    
    def test_file_not_found(self):
        """Test handling of non-existent config file"""
        with self.assertRaises(SystemExit):
            handle_hypothesis_source('/non/existent/path.yml')
    
    def test_invalid_yaml(self):
        """Test handling of malformed YAML"""
        invalid_yaml_path = os.path.join(self.temp_dir, 'invalid.yml')
        with open(invalid_yaml_path, 'w') as f:
            f.write('invalid: yaml: content: [unclosed')
        
        with self.assertRaises(SystemExit):
            handle_hypothesis_source(invalid_yaml_path)
    
    def test_partial_data_pipeline_config(self):
        """Test config with only raw_source but no features"""
        partial_config = {
            'hypothesis': {'name': 'partial_test'},
            'data_pipeline': {'raw_source': 'test_data.csv'}
        }
        config_path = self._create_temp_config(partial_config)
        
        result = handle_hypothesis_source(config_path)
        
        self.assertEqual(result['hypothesis_name'], 'partial_test')
        self.assertEqual(result['raw_data_source'], 'test_data.csv')
        self.assertNotIn('features', result)
    
    def test_model_training_without_processed_data(self):
        """Test model training config without processed_data path"""
        config = {
            'hypothesis': {'name': 'test'},
            'model_training': {
                'type': 'random_forest',
                'parameters': {'n_estimators': 50}
            }
        }
        config_path = self._create_temp_config(config)
        
        result = handle_hypothesis_source(config_path)
        
        self.assertEqual(result['model_type'], 'random_forest')
        self.assertEqual(result['parameters']['n_estimators'], 50)
        self.assertNotIn('processed_data', result)
    
    def test_evaluation_type_stored(self):
        """Test evaluation config with type 'stored'"""
        config = {
            'hypothesis': {'name': 'test'},
            'evaluate_model': {
                'type': 'stored',
                'model_folder': 'storage/models/existing_model'
            }
        }
        config_path = self._create_temp_config(config)
        
        result = handle_hypothesis_source(config_path)
        
        self.assertEqual(result['evaluation_type'], 'stored')
        self.assertEqual(result['model_folder'], 'storage/models/existing_model')
    
    @patch('utils.shared_utils.logging')
    def test_logging_calls(self, mock_logging):
        """Test that appropriate logging calls are made"""
        config_path = self._create_temp_config(self.valid_config)
        
        handle_hypothesis_source(config_path)
        
        mock_logging.info.assert_called()
        mock_logging.debug.assert_called()


if __name__ == '__main__':
    unittest.main()