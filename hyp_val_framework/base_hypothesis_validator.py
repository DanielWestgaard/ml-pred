"""
Base Hypothesis Validator with abstract interface for extensible hypothesis testing.

This provides the core validation framework while allowing specific hypotheses
to implement their own signal generation and testing logic.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data.processing.cleaning import DataCleaner
from data.processing.validation import DataValidator
from utils.data_utils import check_and_return_df
from utils.performance_utils import PerformanceContext
from utils.logging_utils import get_logger

logger = get_logger()

class BaseHypothesisValidator(ABC):
    """
    Abstract base class for hypothesis validation.
    
    All specific hypothesis validators should inherit from this class
    and implement the required abstract methods.
    """
    
    def __init__(self, dataset: Union[str, pd.DataFrame], hypothesis_config: Optional[Dict] = None):
        """
        Initialize the hypothesis validator.
        
        Args:
            dataset: Dataset path or DataFrame
            hypothesis_config: Configuration dictionary for the specific hypothesis
        """
        self.hypothesis_config = hypothesis_config or {}
        
        # Get and clean data
        self.df, self.og_df = check_and_return_df(dataset)
        self.cleaner = DataCleaner(raw_dataset=self.df, save_cleaned_data=False)
        
        # Validation results storage
        self.validation_results = {}
        self.estimation_df = None
        self.validation_df = None
        
    def run(self, validation_size: float = 0.3) -> Dict:
        """
        Run the complete hypothesis validation pipeline.
        
        Args:
            validation_size: Fraction of data to use for out-of-sample testing
            
        Returns:
            Dictionary containing all validation results
        """
        logger.info(f"Starting hypothesis validation: {self.get_hypothesis_name()}")
        
        # 1. Data preparation
        self._prepare_data()
        
        # 2. Chronological splits
        self._create_estimation_validation_periods(validation_size)
        
        # 3. Parameter estimation (on estimation data only)
        parameters = self.estimate_parameters(self.estimation_df)
        logger.info(f"Estimated parameters: {parameters}")
        
        # 4. Signal generation (on validation data)
        validation_signals = self.generate_signals(self.validation_df, parameters)
        logger.info(f"Generated {validation_signals.sum()} signals out of {len(validation_signals)} periods")
        
        # 5. Hypothesis testing
        hypothesis_results = self.test_hypothesis(self.validation_df, validation_signals, parameters)
        
        # 6. Additional validation methods
        additional_results = self.run_additional_validation(parameters)
        
        # 7. Compile results
        self.validation_results = {
            'hypothesis_name': self.get_hypothesis_name(),
            'hypothesis_description': self.get_hypothesis_description(),
            'parameters': parameters,
            'signal_count': validation_signals.sum(),
            'signal_frequency': validation_signals.mean(),
            'hypothesis_test_results': hypothesis_results,
            'additional_validation': additional_results,
            'validation_score': self._calculate_validation_score(hypothesis_results, additional_results),
            'recommendation': self._generate_recommendation()
        }
        
        logger.info(f"Validation complete. Score: {self.validation_results['validation_score']:.2f}")
        return self.validation_results
    
    def _prepare_data(self):
        """Prepare data for hypothesis testing."""
        with PerformanceContext("Data Cleaning"):
            self.df = self.cleaner.run()
        
        with PerformanceContext("Data Validation"):
            validator = DataValidator(self.df)
            validator_results = validator.run()
            self.df = validator_results["validated_data"]
        
        # Calculate forward returns (common to most hypotheses)
        self._calculate_forward_returns()
        
        # Let specific hypothesis do additional preparation
        self.prepare_hypothesis_data()
    
    def _calculate_forward_returns(self):
        """Calculate forward returns for common timeframes."""
        # Get price column (flexible naming)
        price_col = self._get_price_column()
        
        if price_col:
            timeframes = self.hypothesis_config.get('forward_timeframes', [15, 30, 45, 60])
            for minutes in timeframes:
                self.df[f'return_{minutes}m'] = (
                    self.df[price_col].shift(-minutes) - self.df[price_col]
                ) / self.df[price_col]
    
    def _get_price_column(self) -> Optional[str]:
        """Identify the main price column in the dataset."""
        price_candidates = ['mid_price', 'Close', 'close', 'price']
        for col in price_candidates:
            if col in self.df.columns:
                return col
        
        # If no standard price column, check for bid/ask
        if 'bid_close' in self.df.columns and 'ask_close' in self.df.columns:
            self.df['mid_price'] = (self.df['bid_close'] + self.df['ask_close']) / 2
            return 'mid_price'
        
        return None
    
    def _create_estimation_validation_periods(self, validation_size: float):
        """Create chronological data splits."""
        split_idx = int(len(self.df) * (1 - validation_size))
        
        self.estimation_df = self.df.iloc[:split_idx].copy()
        self.validation_df = self.df.iloc[split_idx:].copy()
        
        logger.info(f"Estimation period: {self.estimation_df.index.min()} to {self.estimation_df.index.max()}")
        logger.info(f"Validation period: {self.validation_df.index.min()} to {self.validation_df.index.max()}")
        logger.info(f"Estimation samples: {len(self.estimation_df):,}")
        logger.info(f"Validation samples: {len(self.validation_df):,}")
    
    def test_hypothesis(self, validation_data: pd.DataFrame, signals: pd.Series, parameters: Dict) -> Dict:
        """
        Standard hypothesis testing framework.
        
        Tests if the generated signals predict future price movements.
        """
        results = {}
        
        # Get signal events
        signal_events = validation_data[signals]
        
        if len(signal_events) < 10:
            return {'error': 'Insufficient signal events for testing', 'min_required': 10, 'actual': len(signal_events)}
        
        # Test each forward timeframe
        timeframes = self.hypothesis_config.get('test_timeframes', [30, 45, 60])
        
        for minutes in timeframes:
            return_col = f'return_{minutes}m'
            
            if return_col not in validation_data.columns:
                continue
            
            signal_returns = signal_events[return_col].dropna()
            all_returns = validation_data[return_col].dropna()
            
            if len(signal_returns) < 5:
                continue
            
            # Statistical tests
            test_result = self._perform_statistical_tests(signal_returns, all_returns)
            test_result['timeframe'] = minutes
            test_result['signal_count'] = len(signal_returns)
            
            results[f'{minutes}min'] = test_result
        
        return results
    
    def _perform_statistical_tests(self, signal_returns: pd.Series, all_returns: pd.Series) -> Dict:
        """Perform standard statistical tests."""
        # Basic statistics
        mean_signal = signal_returns.mean()
        mean_all = all_returns.mean()
        std_signal = signal_returns.std()
        
        # T-test
        try:
            t_stat, p_value = stats.ttest_ind(signal_returns, all_returns)
        except:
            t_stat, p_value = np.nan, np.nan
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(signal_returns) - 1) * std_signal**2 +
             (len(all_returns) - 1) * all_returns.std()**2) /
            (len(signal_returns) + len(all_returns) - 2)
        )
        cohens_d = (mean_signal - mean_all) / pooled_std if pooled_std > 0 else 0
        
        # Direction analysis
        positive_pct = (signal_returns > 0).mean() * 100
        negative_pct = (signal_returns < 0).mean() * 100
        
        return {
            'mean_signal': mean_signal,
            'mean_all': mean_all,
            'std_signal': std_signal,
            't_statistic': t_stat,
            'p_value': p_value,
            'effect_size': abs(cohens_d),
            'positive_direction_pct': positive_pct,
            'negative_direction_pct': negative_pct,
            'statistically_significant': p_value < 0.05 if not np.isnan(p_value) else False,
            'practically_significant': abs(cohens_d) > 0.2,  # Small effect size threshold
        }
    
    def run_additional_validation(self, parameters: Dict) -> Dict:
        """Run additional validation methods (can be overridden by subclasses)."""
        additional_results = {}
        
        # Walk-forward validation (if implemented)
        if hasattr(self, 'walk_forward_validation'):
            additional_results['walk_forward'] = self.walk_forward_validation(parameters)
        
        # Bootstrap confidence intervals (if implemented)
        if hasattr(self, 'bootstrap_validation'):
            additional_results['bootstrap'] = self.bootstrap_validation(parameters)
        
        return additional_results
    
    def _calculate_validation_score(self, hypothesis_results: Dict, additional_results: Dict) -> float:
        """Calculate overall validation score (0-1)."""
        score = 0.0
        weight_sum = 0.0
        
        # Score from hypothesis tests
        if hypothesis_results and 'error' not in hypothesis_results:
            significant_tests = sum(1 for result in hypothesis_results.values() 
                                  if isinstance(result, dict) and result.get('statistically_significant', False))
            total_tests = len(hypothesis_results)
            
            if total_tests > 0:
                hypothesis_score = significant_tests / total_tests
                score += hypothesis_score * 0.6  # 60% weight
                weight_sum += 0.6
        
        # Score from additional validation
        if additional_results:
            additional_score = 0.0
            additional_weight = 0.0
            
            # Walk-forward validation
            if 'walk_forward' in additional_results:
                wf_result = additional_results['walk_forward']
                if isinstance(wf_result, dict) and 'success_rate' in wf_result:
                    additional_score += wf_result['success_rate'] * 0.5
                    additional_weight += 0.5
            
            # Bootstrap validation
            if 'bootstrap' in additional_results:
                bs_result = additional_results['bootstrap']
                if isinstance(bs_result, dict) and 'ci_excludes_zero' in bs_result:
                    additional_score += (1.0 if bs_result['ci_excludes_zero'] else 0.0) * 0.5
                    additional_weight += 0.5
            
            if additional_weight > 0:
                score += (additional_score / additional_weight) * 0.4  # 40% weight
                weight_sum += 0.4
        
        return score / weight_sum if weight_sum > 0 else 0.0
    
    def _generate_recommendation(self) -> str:
        """Generate recommendation based on validation results."""
        score = self.validation_results.get('validation_score', 0)
        
        if score >= 0.7:
            return "✅ PROCEED - Strong evidence supports the hypothesis"
        elif score >= 0.5:
            return "⚠️ CAUTION - Mixed evidence, proceed with careful risk management"
        else:
            return "❌ ABANDON - Insufficient evidence to support the hypothesis"
    
    def print_validation_summary(self):
        """Print a summary of validation results."""
        if not self.validation_results:
            print("❌ No validation results available. Run validation first.")
            return
        
        results = self.validation_results
        
        print("=" * 80)
        print(f"HYPOTHESIS VALIDATION SUMMARY")
        print("=" * 80)
        print(f"Hypothesis: {results['hypothesis_name']}")
        print(f"Description: {results['hypothesis_description']}")
        print(f"Validation Score: {results['validation_score']:.2f} / 1.00")
        print(f"Recommendation: {results['recommendation']}")
        print()
        
        # Parameters
        print("Parameters Estimated:")
        for key, value in results['parameters'].items():
            print(f"  • {key}: {value}")
        print()
        
        # Signal statistics
        print(f"Signal Statistics:")
        print(f"  • Total signals: {results['signal_count']}")
        print(f"  • Signal frequency: {results['signal_frequency']:.1%}")
        print()
        
        # Hypothesis test results
        print("Statistical Test Results:")
        for timeframe, test_result in results['hypothesis_test_results'].items():
            if isinstance(test_result, dict) and 'error' not in test_result:
                sig_mark = "✅" if test_result['statistically_significant'] else "❌"
                effect_mark = "📈" if test_result['practically_significant'] else "📉"
                
                print(f"  {timeframe}: {sig_mark} {effect_mark}")
                print(f"    P-value: {test_result['p_value']:.4f}")
                print(f"    Effect size: {test_result['effect_size']:.3f}")
                print(f"    Mean return (signal): {test_result['mean_signal']:.6f}")
                print(f"    Direction bias: {test_result['positive_direction_pct']:.1f}% positive")
                print()
    
    def save_results(self, filepath: Optional[str] = None):
        """Save validation results to file."""
        if not self.validation_results:
            logger.warning("No validation results to save")
            return
        
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            hypothesis_name = self.get_hypothesis_name().lower().replace(" ", "_")
            filepath = f"validation_results_{hypothesis_name}_{timestamp}.json"
        
        import json
        with open(filepath, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            serializable_results = self._make_json_serializable(self.validation_results)
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Validation results saved to: {filepath}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    # Abstract methods that must be implemented by subclasses
    @abstractmethod
    def get_hypothesis_name(self) -> str:
        """Return a descriptive name for the hypothesis."""
        pass
    
    @abstractmethod
    def get_hypothesis_description(self) -> str:
        """Return a detailed description of the hypothesis."""
        pass
    
    @abstractmethod
    def prepare_hypothesis_data(self):
        """Prepare data specific to this hypothesis (e.g., calculate indicators)."""
        pass
    
    @abstractmethod
    def estimate_parameters(self, estimation_data: pd.DataFrame) -> Dict:
        """
        Estimate parameters from the estimation dataset.
        
        Args:
            estimation_data: Historical data for parameter estimation
            
        Returns:
            Dictionary of estimated parameters
        """
        pass
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, parameters: Dict) -> pd.Series:
        """
        Generate trading signals based on the hypothesis.
        
        Args:
            data: Dataset to generate signals for
            parameters: Parameters estimated from historical data
            
        Returns:
            Boolean Series indicating where signals occur
        """
        pass


class SimpleHypothesisValidator(BaseHypothesisValidator):
    """
    Simplified validator for quick hypothesis testing.
    Reduces boilerplate for simple hypotheses.
    """
    
    def __init__(self, dataset: Union[str, pd.DataFrame], 
                 hypothesis_config: Dict):
        """
        Args:
            hypothesis_config: Must contain:
                - name: Hypothesis name
                - description: Hypothesis description  
                - signal_logic: Function that takes (data, parameters) and returns signals
                - parameter_logic: Function that takes estimation_data and returns parameters
        """
        required_keys = ['name', 'description', 'signal_logic', 'parameter_logic']
        missing_keys = [key for key in required_keys if key not in hypothesis_config]
        
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")
        
        super().__init__(dataset, hypothesis_config)
    
    def get_hypothesis_name(self) -> str:
        return self.hypothesis_config['name']
    
    def get_hypothesis_description(self) -> str:
        return self.hypothesis_config['description']
    
    def prepare_hypothesis_data(self):
        # Default implementation - can be overridden in config
        if 'data_preparation' in self.hypothesis_config:
            self.hypothesis_config['data_preparation'](self.df)
    
    def estimate_parameters(self, estimation_data: pd.DataFrame) -> Dict:
        return self.hypothesis_config['parameter_logic'](estimation_data)
    
    def generate_signals(self, data: pd.DataFrame, parameters: Dict) -> pd.Series:
        return self.hypothesis_config['signal_logic'](data, parameters)