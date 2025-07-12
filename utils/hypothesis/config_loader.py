import inspect
import logging
import os
import sys
from typing import Any, Dict
import yaml


class LoadHypothesisConfigMetadata:
    """Utility class for loading and accessing hypothesis metadata from YAML files."""
    
    def __init__(self, config_file: str):
        """
        Initialize with path to metadata file or model directory.
        
        Args:
            metadata_path: Path to model_metadata.yml file or directory containing it
        """
        if os.path.isdir(config_file):
            # If directory provided, look for metadata file
            metadata_file = os.path.join(config_file, 'model_metadata.yml')
        else:
            metadata_file = config_file
            
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
            
        self.config_file = metadata_file
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from YAML file."""
        try:
            with open(self.config_file, 'r') as file:
                metadata = yaml.safe_load(file)
            logging.debug(f"Successfully loaded metadata from {self.config_file}")
            return metadata
        except Exception as e:
            logging.error(f"Error loading metadata: {e}")
            raise
    
    def handle_hypothesis_source(config_file: str) -> dict:
        """
        Function for checking provided config.yml source for hypothesis, and returning the configurations it finds.
        
        Args:
            - config_file: Path to config.yml, specifying hypothesis (-name), selected features, etc.
        
        returns:
            - list of successfully ectracted 'components'/configurations in Dict format.
                - Eg: '{'hypothesis_name': 'session_based_tod_anomalies', 'raw_data_source': 'bla/raw_data.csv', 'features': {'time_of_day': [], 'time_of_week': []}'
                - Can be accessed like this: features = shared_utils.handle_hypothesis_source.get('features') 
        """
        # Load YAML file
        try:
            with open(config_file, 'r') as file:
                config_h = yaml.safe_load(file)
        except FileNotFoundError:
            logging.error(f"Error: Config file '{config_file}' not found!")
            sys.exit(1)
        except yaml.YAMLError as e:
            logging.error(f"Error parsing YAML file: {e}")
            sys.exit(1)
        
        # Dict to populate with successfully extracted 'values'
        dict = {}
        
        # Hypothesis name - must match with the name of its target method in target_hypothesis.py
        try:
            hypothesis_name = config_h['hypothesis']['name']
            dict.update({'hypothesis_name': hypothesis_name})
            logging.info(f"hypothesis_name: {hypothesis_name}")
        except Exception as e:
            logging.error(f"Unable to get hypothesis name : {e}")
            sys.exit()
        
        # Data Pipeline
        try:
            # Raw data
            raw_data_source = config_h['data_pipeline']['raw_source']
            dict.update({'raw_data_source': raw_data_source})
            try:
                # Provided features
                features = config_h['data_pipeline']['required_features']
                dict.update({"features": features})
                logging.debug(f"Data source: '{raw_data_source}' and required Features: {features}")
            except Exception as e:
                logging.error(f"Unable to extract Required/Selected Features for generation : {e}")
        except Exception as e:
            logging.warning(f"Unable to extract Raw Dataset, for Data Pipeline, from configurations (might not exist?) : {e}")
        
        # Model Training
        try:
            # Model Type
            model_type = config_h['model_training']['type']
            dict.update({'model_type': model_type})
            logging.info(f"model_type: {model_type}")
            
            try:
                # Potential Processed Data
                processed_data = config_h['model_training']['processed_data']
                dict.update({"processed_data": processed_data})
                logging.debug(f"processed_data: {processed_data}")
            except Exception as e:
                logging.warning(f"Unable to extract processed_data for model training, but they might also not be there : {e}")
            
            try:
                # Potential Parameters
                parameters = config_h['model_training']['parameters']
                dict.update({"parameters": parameters})
                logging.debug(f"parameters: {parameters}")
            except Exception as e:
                logging.warning(f"Unable to extract Parameters for model training, but they might also not be there : {e}")
            
        except Exception as e:
            logging.warning(f"Unable to get model_type from configuration, might not be there? : {e}")

        # Evaluation
        try:
            # Evaluate model
            evaluate_model = config_h['evaluate_model']
            dict.update({"evaluate_model": evaluate_model})
            logging.debug(f"evaluate_model: {evaluate_model}")
            try:
                evaluation_type = config_h['evaluate_model']['type']
                dict.update({"evaluation_type": evaluation_type})
                logging.debug(f"evaluation_type: {evaluation_type}")
                try:
                    evaluation_model_path = config_h['evaluate_model']['model_folder']
                    dict.update({"model_folder": evaluation_model_path})
                    logging.debug(f"model_folder: {evaluation_model_path}")
                except:
                    logging.warning(f"Unable to get evaluation model folder. Might having type 'new'?")
            except:
                logging.warning(f"Unable to get evaluation type.")
        except Exception as e:
            logging.warning(f"Unable to extract evaluation from model training, but might also not be there : {e}")
        
        # Strategy
        # ...
        
        return dict


# Standalone utility functions
def load_model_metadata(metadata_path: str) -> LoadHypothesisConfigMetadata:
    """
    Convenience function to load hypothesis config.
    
    Args:
        metadata_path: Path to metadata file or model directory
        
    Returns:
        ModelMetadataLoader instance
    """
    return LoadHypothesisConfigMetadata(metadata_path)