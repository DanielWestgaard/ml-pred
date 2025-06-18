import os
import pandas as pd
import yaml


# Integration test of Hypothesis Flow
def create_sample_raw_data():
    """Create sample raw data (simple)."""
    # Create sample raw data
    sample_data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=1000, freq='h'),
        'open': [1.2 + 0.001 * i for i in range(1000)],
        'high': [1.21 + 0.001 * i for i in range(1000)],
        'low': [1.19 + 0.001 * i for i in range(1000)],
        'close': [1.205 + 0.001 * i for i in range(1000)],
        'volume': [100 + i for i in range(1000)]
    })
    
    return sample_data

def complete_config(raw_data_path):
    "Create complete hypothesis data"
    # Complete hypothesis config
    complete_config = {
        'hypothesis': {
            'name': 'session_based_tod_anomalies'
        },
        'data_pipeline': {
            'raw_source': raw_data_path,
            'required_features': {
                'time_of_day': [],
                'volatility_regimes': []
            }
        },
        'model_training': {
            'type': 'xgboost',
            'parameters': {
                'n_estimators': 10,  # Small for testing
                'max_depth': 3
            }
        },
        'evaluate_model': {
            'type': 'new'
        }
    }
    
    return complete_config

def data_pipeline_only_config(raw_data_path):
    """Data pipeline only config."""
    data_only_config = {
        'hypothesis': {
            'name': 'test_hypothesis'
        },
        'data_pipeline': {
            'raw_source': raw_data_path,
            'required_features': {
                'time_of_day': []
            }
        }
    }
    
    return data_only_config

def create_config_file(config_dict, temp_dir, filename='config.yml'):
    """Helper to create config file"""
    config_path = os.path.join(temp_dir, filename)
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f)
    return config_path


# Unit tests of Hypothesis config
def valid_config():
    valid_config = {
        'hypothesis': {
            'name': 'test_hypothesis'
        },
        'data_pipeline': {
            'raw_source': 'storage/data/test/raw_data.csv',
            'required_features': {
                'time_of_day': [],
                'volatility_regimes': []
            }
        },
        'model_training': {
            'type': 'xgboost',
            'processed_data': 'storage/data/test/processed_data.csv',
            'parameters': {
                'n_estimators': 100,
                'max_depth': 6
            }
        },
        'evaluate_model': {
            'type': 'new',
            'model_folder': 'storage/models/test_model'
        }
    }
    
    return valid_config

def minimal_config():
    minimal_config = {
        'hypothesis': {
            'name': 'minimal_test'
        }
    }
    
    return minimal_config

def invalid_config():
    invalid_config = {
        'data_pipeline': {
            'raw_source': 'test.csv'
        }
    }
    
    return invalid_config