from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from utils.logging_utils import get_logger

logger = get_logger()

class HypothesisPlugin(ABC):
    """
    Base class for hypothesis-specific plugins in the backtesting system.
    
    This abstract class defines the interface for hypothesis customizations,
    allowing new trading strategies to be added through configuration rather
    than code changes.
    """
    
    def __init__(self, hypothesis_name: str, config: Dict[str, Any]):
        """
        Initialize the hypothesis plugin
        
        Args:
            hypothesis_name: Name of the hypothesis (e.g., 'spread_compression_patterns')
            config: Hypothesis-specific configuration from YAML
        """
        self.hypothesis_name = hypothesis_name
        self.config = config
        self.exit_config = config.get('backtesting', {}).get('exit_strategy', {})
        self.signal_tracking_config = config.get('backtesting', {}).get('signal_tracking', {})
        
        logger.info(f"Initialized {self.__class__.__name__} for hypothesis: {hypothesis_name}")
    
    @abstractmethod
    def get_exit_strategy(self):
        """
        Get the exit strategy instance for this hypothesis
        
        Returns:
            ExitStrategy instance configured for this hypothesis
        """
        pass
    
    @abstractmethod
    def get_signal_tracker(self):
        """
        Get the signal tracker instance for this hypothesis
        
        Returns:
            SignalTracker instance configured for this hypothesis
        """
        pass
    
    def customize_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Customize feature values for hypothesis-specific requirements
        
        Args:
            features: Raw feature dictionary
            
        Returns:
            Modified feature dictionary
        """
        # Default implementation - no customization
        return features
    
    def validate_configuration(self) -> Tuple[bool, list]:
        """
        Validate hypothesis-specific configuration
        
        Returns:
            Tuple of (is_valid: bool, error_messages: list)
        """
        # Default validation - can be overridden
        errors = []
        
        # Check for required config sections
        if 'backtesting' not in self.config:
            errors.append("Missing 'backtesting' configuration section")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def get_hypothesis_info(self) -> Dict[str, Any]:
        """
        Get information about this hypothesis plugin
        
        Returns:
            Dictionary with plugin metadata
        """
        return {
            'plugin_class': self.__class__.__name__,
            'hypothesis_name': self.hypothesis_name,
            'supports_exit_strategy': hasattr(self, 'get_exit_strategy'),
            'supports_signal_tracking': hasattr(self, 'get_signal_tracker'),
            'supports_feature_customization': hasattr(self, 'customize_features')
        }