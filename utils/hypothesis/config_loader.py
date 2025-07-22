import logging
import os
from typing import Any, Dict, Optional, List, Union
import yaml


class HypothesisConfigError(Exception):
    """Custom exception for hypothesis configuration errors."""
    pass


class LoadHypothesisConfigMetadata:
    """Utility class for loading and accessing hypothesis configuration from YAML files."""
    
    def __init__(self, config_file: str):
        """
        Initialize with path to hypothesis config file or directory.
        
        Args:
            config_file: Path to config.yml file or directory containing it
        """
        if os.path.isdir(config_file):
            # If directory provided, look for config file
            config_path = os.path.join(config_file, 'config.yml')
        else:
            config_path = config_file
            
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        self.config_file = config_path
        self.logger = logging.getLogger(__name__)
        self.config_data = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            
            if not isinstance(config, dict):
                raise HypothesisConfigError(f"Config file must contain a dictionary, got {type(config)}")
                
            self.logger.debug(f"Successfully loaded config from {self.config_file}")
            return config
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {self.config_file}")
        except yaml.YAMLError as e:
            raise HypothesisConfigError(f"Error parsing YAML file {self.config_file}: {e}")
        except Exception as e:
            raise HypothesisConfigError(f"Unexpected error loading config: {e}")
    
    def validate_config(self) -> List[str]:
        """
        Validate the configuration and return list of validation errors.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check hypothesis section
        if not self._safe_get(['hypothesis']):
            errors.append("Missing required section: 'hypothesis'")
        elif not self._safe_get(['hypothesis', 'name']):
            errors.append("Missing required field: 'hypothesis.name'")
        
        # Validate data_pipeline if present
        data_pipeline = self._safe_get(['data_pipeline'])
        if data_pipeline:
            if not isinstance(data_pipeline, dict):
                errors.append("Section 'data_pipeline' must be a dictionary")
            elif not self._safe_get(['data_pipeline', 'raw_source']):
                errors.append("Missing required field: 'data_pipeline.raw_source'")
        
        # Validate model_training if present
        model_training = self._safe_get(['model_training'])
        if model_training:
            if not isinstance(model_training, dict):
                errors.append("Section 'model_training' must be a dictionary")
            elif not self._safe_get(['model_training', 'type']):
                errors.append("Missing required field: 'model_training.type'")
        
        # Validate strategy if present
        strategy = self._safe_get(['strategy'])
        if strategy and not isinstance(strategy, dict):
            errors.append("Section 'strategy' must be a dictionary")
        
        # Validate evaluate_model if present
        evaluate_model = self._safe_get(['evaluate_model'])
        if evaluate_model and not isinstance(evaluate_model, dict):
            errors.append("Section 'evaluate_model' must be a dictionary")
        
        return errors
    
    def _safe_get(self, path: List[str], default: Any = None) -> Any:
        """Safely get nested dictionary value using path."""
        current = self.config_data
        try:
            for key in path:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    def _extract_section(self, section_name: str, required_fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """Extract a configuration section with optional required field validation."""
        section_data = {}
        section_config = self._safe_get([section_name])
        
        if section_config is None:
            self.logger.debug(f"Section '{section_name}' not found in configuration")
            return section_data
        
        if not isinstance(section_config, dict):
            self.logger.warning(f"Section '{section_name}' is not a dictionary, skipping")
            return section_data
        
        # Check required fields
        if required_fields:
            missing_fields = [field for field in required_fields if field not in section_config]
            if missing_fields:
                self.logger.warning(f"Missing required fields in '{section_name}': {missing_fields}")
        
        # Extract all fields from section
        for key, value in section_config.items():
            section_data[f"{section_name}_{key}" if section_name != 'hypothesis' else key] = value
            self.logger.debug(f"Extracted {section_name}.{key}: {value}")
        
        # Also include the full section
        section_data[section_name] = section_config
        
        return section_data
    
    def handle_hypothesis_source(self) -> Dict[str, Any]:
        """
        Extract configuration from hypothesis config file.
        
        Returns:
            Dict containing successfully extracted configuration components.
            Example: {'hypothesis_name': 'session_based_tod_anomalies', 'raw_data_source': 'path/to/data.csv'}
        """
        config = {}
        
        # Extract hypothesis name (required)
        hypothesis_name = self._safe_get(['hypothesis', 'name'])
        if not hypothesis_name:
            raise HypothesisConfigError("Missing required field: hypothesis.name")
        
        config['hypothesis_name'] = hypothesis_name
        self.logger.info(f"Loaded hypothesis: {hypothesis_name}")
        
        # Extract all sections using helper method
        sections_to_extract = [
            ('data_pipeline', ['raw_source']),  # raw_source is required if section exists
            ('model_training', ['type']),       # type is required if section exists  
            ('evaluate_model', None),           # No required fields
            ('strategy', None)                  # No required fields
        ]
        
        for section_name, required_fields in sections_to_extract:
            section_data = self._extract_section(section_name, required_fields)
            config.update(section_data)
        
        # Special handling for required_features from data_pipeline
        if 'data_pipeline' in config and 'required_features' in config['data_pipeline']:
            config['features'] = config['data_pipeline']['required_features']
            self.logger.debug(f"Extracted required_features as 'features': {config['features']}")
        
        # Log summary
        sections_found = [section for section in ['data_pipeline', 'model_training', 'evaluate_model', 'strategy'] 
                         if section in config]
        self.logger.info(f"Successfully loaded config sections: {sections_found}")
        
        return config


# Standalone utility functions
def load_hypothesis_config(config_path: str) -> LoadHypothesisConfigMetadata:
    """
    Convenience function to load hypothesis configuration.
    
    Args:
        config_path: Path to config.yml file or directory containing it
        
    Returns:
        LoadHypothesisConfigMetadata instance
    """
    return LoadHypothesisConfigMetadata(config_path)