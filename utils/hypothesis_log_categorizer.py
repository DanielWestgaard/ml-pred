# utils/hypothesis_log_categorizer.py
"""
Utility to categorize log files based on hypothesis configuration content.
This helps organize logs into appropriate directories based on what components
are being used in the hypothesis workflow.
"""

from typing import Dict, Any
import yaml

def categorize_hypothesis_config(config_path: str) -> str:
    """
    Analyze a hypothesis configuration file and determine the appropriate log category.
    
    Args:
        config_path: Path to the hypothesis configuration YAML file
        
    Returns:
        Log category string: 'mixed', 'strategy', 'evaluation', 'training', or 'data'
        
    Examples:
        - config with only 'strategy' section -> 'strategy'
        - config with 'data_pipeline' + 'model_training' -> 'mixed'
        - config with only 'evaluate_model' -> 'evaluation'
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        return _determine_log_category(config)
        
    except Exception:
        # Fallback to 'mixed' if we can't parse the config
        return 'mixed'

def _determine_log_category(config: Dict[str, Any]) -> str:
    """
    Determine log category based on configuration sections present.
    
    Args:
        config: Parsed configuration dictionary
        
    Returns:
        Log category string
    """
    # Define component categories
    component_sections = {
        'data': ['data_pipeline'],
        'training': ['model_training'],
        'evaluation': ['evaluate_model'],
        'strategy': ['strategy'],
        'backtesting': ['backtesting']
    }
    
    # Count which categories are present
    present_categories = []
    
    for category, sections in component_sections.items():
        for section in sections:
            if section in config and config[section] is not None:
                # Check if section is not commented out (has actual content)
                if _is_section_active(config[section]):
                    present_categories.append(category)
                    break
    
    # Determine final category
    if len(present_categories) == 0:
        return 'mixed'  # No clear components, default to mixed
    elif len(present_categories) == 1:
        return present_categories[0]  # Single component
    else:
        return 'mixed'  # Multiple components

def _is_section_active(section_config: Any) -> bool:
    """
    Check if a configuration section is active (not just commented structure).
    
    Args:
        section_config: Configuration section content
        
    Returns:
        True if section is active, False if it's just commented structure
    """
    if section_config is None:
        return False
        
    if isinstance(section_config, dict):
        # Check if it has meaningful content (not just empty dict)
        return len(section_config) > 0
        
    # For other types, consider them active
    return True

def get_dynamic_log_path(config_path: str) -> str:
    """
    Generate a dynamic log path based on hypothesis configuration content.
    
    Args:
        config_path: Path to the hypothesis configuration file
        
    Returns:
        Log directory path relative to logs/ folder
        
    Examples:
        - 'strategy' for strategy-only configs
        - 'mixed' for multi-component configs
        - 'evaluation' for evaluation-only configs
    """
    category = categorize_hypothesis_config(config_path)
    return category

# Mapping for user-friendly category descriptions
CATEGORY_DESCRIPTIONS = {
    'strategy': 'Strategy execution and live trading',
    'evaluation': 'Model evaluation and performance analysis', 
    'training': 'Model training and hyperparameter tuning',
    'data': 'Data pipeline and feature engineering',
    'mixed': 'Mixed workflow with multiple components'
}

def get_category_description(category: str) -> str:
    """Get a user-friendly description of the log category."""
    return CATEGORY_DESCRIPTIONS.get(category, 'Unknown workflow type')