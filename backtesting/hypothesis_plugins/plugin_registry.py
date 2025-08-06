from typing import Dict, Any, Optional, Type
from .base_plugin import HypothesisPlugin
from .exit_strategies import ExitStrategyFactory
from .signal_trackers import SignalTrackerFactory
from utils.logging_utils import get_logger

logger = get_logger()

class HypothesisPluginRegistry:
    """
    Registry for hypothesis plugins that manages plugin discovery,
    instantiation, and configuration validation.
    """
    
    _plugins: Dict[str, Type[HypothesisPlugin]] = {}
    
    @classmethod
    def register_plugin(cls, hypothesis_name: str, plugin_class: Type[HypothesisPlugin]):
        """
        Register a hypothesis plugin
        
        Args:
            hypothesis_name: Name/pattern for the hypothesis
            plugin_class: Plugin class to register
        """
        cls._plugins[hypothesis_name] = plugin_class
        logger.info(f"Registered hypothesis plugin: {hypothesis_name} -> {plugin_class.__name__}")
    
    @classmethod
    def get_plugin(cls, hypothesis_name: str, config: Dict[str, Any]) -> HypothesisPlugin:
        """
        Get plugin instance for hypothesis
        
        Args:
            hypothesis_name: Name of the hypothesis
            config: Full hypothesis configuration
            
        Returns:
            HypothesisPlugin instance or DefaultHypothesisPlugin
        """
        # Try exact match first
        if hypothesis_name in cls._plugins:
            plugin_class = cls._plugins[hypothesis_name]
            return plugin_class(hypothesis_name, config)
        
        # Try pattern matching
        for pattern, plugin_class in cls._plugins.items():
            if pattern in hypothesis_name.lower():
                return plugin_class(hypothesis_name, config)
        
        # Fall back to default plugin
        logger.info(f"No specific plugin found for '{hypothesis_name}', using DefaultHypothesisPlugin")
        return DefaultHypothesisPlugin(hypothesis_name, config)
    
    @classmethod
    def get_available_plugins(cls) -> Dict[str, str]:
        """Get available plugins mapping"""
        return {name: plugin_class.__name__ for name, plugin_class in cls._plugins.items()}
    
    @classmethod
    def validate_plugin_config(cls, hypothesis_name: str, config: Dict[str, Any]) -> tuple:
        """
        Validate configuration for a hypothesis plugin
        
        Args:
            hypothesis_name: Name of the hypothesis
            config: Configuration to validate
            
        Returns:
            Tuple of (is_valid: bool, errors: list)
        """
        try:
            plugin = cls.get_plugin(hypothesis_name, config)
            return plugin.validate_configuration()
        except Exception as e:
            return False, [f"Plugin validation failed: {str(e)}"]


class DefaultHypothesisPlugin(HypothesisPlugin):
    """
    Default plugin for hypotheses without specific customizations.
    
    Provides basic functionality using the factory patterns for
    exit strategies and signal trackers.
    """
    
    def get_exit_strategy(self):
        """Get default exit strategy"""
        return ExitStrategyFactory.create(self.hypothesis_name, self.exit_config)
    
    def get_signal_tracker(self):
        """Get default signal tracker"""
        return SignalTrackerFactory.create(self.hypothesis_name, self.signal_tracking_config)
    
    def validate_configuration(self) -> tuple:
        """Validate default configuration requirements"""
        errors = []
        
        # Check basic backtesting config
        if 'backtesting' not in self.config:
            errors.append("Missing 'backtesting' configuration section")
            return False, errors
        
        backtest_config = self.config['backtesting']
        
        # Check required fields
        required_fields = ['enabled', 'initial_capital']
        for field in required_fields:
            if field not in backtest_config:
                errors.append(f"Missing required field: backtesting.{field}")
        
        # Validate numeric fields
        numeric_fields = {
            'initial_capital': (100, 1000000),  # $100 to $1M
            'confidence_threshold': (0.0, 1.0),
            'transaction_costs': (0.0, 0.01),  # 0% to 1%
            'slippage': (0.0, 0.01)
        }
        
        for field, (min_val, max_val) in numeric_fields.items():
            value = backtest_config.get(field)
            if value is not None:
                try:
                    value = float(value)
                    if not (min_val <= value <= max_val):
                        errors.append(f"Field {field}={value} must be between {min_val} and {max_val}")
                except (ValueError, TypeError):
                    errors.append(f"Field {field} must be numeric")
        
        is_valid = len(errors) == 0
        return is_valid, errors


# Auto-register the default plugin
HypothesisPluginRegistry.register_plugin('default', DefaultHypothesisPlugin)

# Auto-register hypothesis-specific plugins
def _auto_register_plugins():
    """Auto-register plugins from the types directory"""
    try:
        from .types import SpreadCompressionPlugin
        HypothesisPluginRegistry.register_plugin('spread_compression', SpreadCompressionPlugin)
        logger.info("Auto-registered hypothesis-specific plugins")
    except ImportError as e:
        logger.warning(f"Could not auto-register some hypothesis plugins: {e}")

# Perform auto-registration
_auto_register_plugins()