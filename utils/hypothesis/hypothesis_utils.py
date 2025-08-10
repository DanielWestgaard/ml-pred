import os
import yaml
from typing import Optional

from utils.model.model_metadata_loader import load_model_metadata
import utils.hypothesis.config_loader as config_loader
from utils.logging_utils import get_logger

logger = get_logger()


def determine_hypothesis_type(hypothesis_config: Optional[str] = None, 
                            model_directory: Optional[str] = None) -> str:
    """
    Utility function to determine hypothesis type from config or model metadata.
    Avoids duplication between evaluation/evaluation.py, strategy/strategy.py, 
    and multi_data_sources/ds_orchestrator.py
    
    Args:
        hypothesis_config: Path to hypothesis config file (optional)
        model_directory: Path to model directory (optional)
        
    Returns:
        Hypothesis type string
        
    Raises:
        ValueError: If hypothesis type cannot be determined
    """
    # Option 1: From hypothesis config file
    if hypothesis_config and os.path.exists(hypothesis_config):
        try:
            config_data = config_loader.load_hypothesis_config(hypothesis_config).handle_hypothesis_source()
            hypothesis_type = config_data.get('hypothesis_name')
            if hypothesis_type:
                logger.info(f"Determined hypothesis type from config: {hypothesis_type}")
                return hypothesis_type
        except Exception as e:
            logger.warning(f"Failed to load hypothesis type from config {hypothesis_config}: {e}")
    
    # Option 2: From model metadata
    if model_directory and os.path.exists(model_directory):
        try:
            model_metadata = load_model_metadata(model_directory)
            hypothesis_type = model_metadata.get('hypothesis_name')
            if hypothesis_type:
                logger.info(f"Determined hypothesis type from model metadata: {hypothesis_type}")
                return hypothesis_type
        except Exception as e:
            logger.warning(f"Failed to load hypothesis type from model metadata {model_directory}: {e}")
    
    # Option 3: From config file in model directory
    if model_directory:
        config_path = os.path.join(model_directory, 'config.yml')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    hypothesis_type = config.get('hypothesis', {}).get('name')
                    if hypothesis_type:
                        logger.info(f"Determined hypothesis type from model config: {hypothesis_type}")
                        return hypothesis_type
            except Exception as e:
                logger.warning(f"Failed to load hypothesis type from model config {config_path}: {e}")
    
    # If all options fail
    error_msg = "Could not determine hypothesis type. Please provide either:\n"
    error_msg += "1. A valid hypothesis_config path, or\n"
    error_msg += "2. A valid model_directory with metadata"
    
    logger.error(error_msg)
    raise ValueError(error_msg)