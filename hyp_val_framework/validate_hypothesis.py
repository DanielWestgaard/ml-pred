"""
Updated HypothesisValidator that integrates with the new framework.
Maintains backward compatibility while providing enhanced validation capabilities.
"""

from typing import Dict, Union
import pandas as pd

from data.processing.cleaning import DataCleaner
from data.processing.validation import DataValidator
from utils import data_utils
from utils.data_utils import check_and_return_df
from utils.performance_utils import memory_monitor, PerformanceContext
from utils.logging_utils import get_logger

from hyp_val_framework.base_hypothesis_validator import BaseHypothesisValidator, SimpleHypothesisValidator
from hyp_val_framework.quick_validator import quick_validate_hypothesis, HypothesisTemplates

logger = get_logger()


class HypothesisValidator:
    """
    Enhanced hypothesis validator that provides both simple and advanced validation.
    
    This class serves as a facade for the validation framework, providing:
    1. Backward compatibility with existing code
    2. Easy access to quick validation methods
    3. Integration with the full validation framework
    """
    
    def __init__(self, dataset: Union[str, pd.DataFrame], hypothesis_config: Dict = None):
        """
        Initialize the hypothesis validator.
        
        Args:
            dataset: Dataset path or DataFrame to test hypothesis on
            hypothesis_config: Optional configuration for specific hypothesis
        """
        # Get the data (maintain compatibility)
        self.df, self.og_df = check_and_return_df(dataset)
        self.cleaner = DataCleaner(raw_dataset=self.df, save_cleaned_data=False)
        
        # Store for advanced validation
        self.dataset = dataset
        self.hypothesis_config = hypothesis_config or {}
        
        # Initialize with basic processing
        self.estimation_df = None
        self.validation_df = None
        self.validator_results = None
    
    # Quick validation methods
    def quick_test_volume_hypothesis(self, volume_threshold_pct: int = 95):
        """Quick test of volume spike hypothesis."""
        return quick_validate_hypothesis(
            self.dataset, 
            **HypothesisTemplates.volume_spike_reversion(volume_threshold_pct)
        )
    
    def quick_test_momentum_hypothesis(self, lookback_periods: int = 20, momentum_threshold_pct: int = 80):
        """Quick test of momentum continuation hypothesis."""
        return quick_validate_hypothesis(
            self.dataset,
            **HypothesisTemplates.price_momentum_continuation(lookback_periods, momentum_threshold_pct)
        )
    
    def quick_test_volatility_hypothesis(self, volatility_window: int = 20, regime_threshold_pct: int = 90):
        """Quick test of volatility regime change hypothesis."""
        return quick_validate_hypothesis(
            self.dataset,
            **HypothesisTemplates.volatility_regime_change(volatility_window, regime_threshold_pct)
        )
    
    def quick_test_custom_hypothesis(self, 
                                   hypothesis_name: str,
                                   hypothesis_description: str,
                                   signal_logic,
                                   parameter_logic,
                                   data_preparation=None,
                                   config=None):
        """Quick test of a custom hypothesis."""
        return quick_validate_hypothesis(
            self.dataset,
            hypothesis_name,
            hypothesis_description,
            signal_logic,
            parameter_logic,
            data_preparation,
            config
        )
    
    def create_advanced_validator(self, validator_class: type) -> BaseHypothesisValidator:
        """
        Create an advanced validator for complex hypothesis testing.
        
        Args:
            validator_class: Class that inherits from BaseHypothesisValidator
            
        Returns:
            Instance of the advanced validator
        """
        if not issubclass(validator_class, BaseHypothesisValidator):
            raise ValueError("validator_class must inherit from BaseHypothesisValidator")
        
        return validator_class(self.dataset, self.hypothesis_config)
    
    def run_comprehensive_validation(self, validator_class: type) -> Dict:
        """
        Run comprehensive validation using an advanced validator.
        
        Args:
            validator_class: Advanced validator class
            
        Returns:
            Complete validation results
        """
        advanced_validator = self.create_advanced_validator(validator_class)
        results = advanced_validator.run()
        advanced_validator.print_validation_summary()
        return results


# Convenience functions for backward compatibility
def create_hypothesis_validator(dataset, config=None):
    """Factory function to create a hypothesis validator."""
    return HypothesisValidator(dataset, config)


def validate_spread_expansion_hypothesis(dataset, config=None):
    """Validate the spread expansion hypothesis (example)."""
    from examples.spread_expansion_validator import create_spread_expansion_validator
    
    validator = create_spread_expansion_validator(dataset, config)
    results = validator.run()
    validator.print_validation_summary()
    return results
