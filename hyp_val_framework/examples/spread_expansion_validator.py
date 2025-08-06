"""
Spread Expansion Hypothesis Validator

Example implementation of the BaseHypothesisValidator for testing spread expansion patterns.
Demonstrates how to create a specific hypothesis validator.
"""

import pandas as pd
import numpy as np
from typing import Dict
from scipy import stats

from ..base_hypothesis_validator import BaseHypothesisValidator


class SpreadExpansionValidator(BaseHypothesisValidator):
    """
    Validator for the spread expansion hypothesis:
    "When spread expands to high percentiles with above-average volume, 
    price will mean-revert (negative returns) over 45-60 minutes"
    """
    
    def get_hypothesis_name(self) -> str:
        return "Spread Expansion Mean Reversion"
    
    def get_hypothesis_description(self) -> str:
        return (
            "When spread expands to 85th-95th percentiles with above-average volume, "
            "price will experience mean reversion (negative returns) over 45-60 minute horizons. "
            "The theory is that extreme spread expansion indicates temporary market stress "
            "that subsequently reverts to normal levels."
        )
    
    def prepare_hypothesis_data(self):
        """Prepare spread expansion specific data."""
        # Ensure we have required columns
        required_cols = ['spread_close', 'volume']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        
        if missing_cols:
            # Try to calculate spread if we have bid/ask
            if 'bid_close' in self.df.columns and 'ask_close' in self.df.columns:
                self.df['spread_close'] = self.df['ask_close'] - self.df['bid_close']
            else:
                raise ValueError(f"Missing required columns for spread expansion: {missing_cols}")
        
        # Calculate mid price if not present
        if 'mid_price' not in self.df.columns:
            if 'bid_close' in self.df.columns and 'ask_close' in self.df.columns:
                self.df['mid_price'] = (self.df['bid_close'] + self.df['ask_close']) / 2
    
    def estimate_parameters(self, estimation_data: pd.DataFrame) -> Dict:
        """Estimate spread expansion thresholds and volume averages."""
        # Get configuration or use defaults
        config = self.hypothesis_config
        
        percentiles = config.get('spread_percentiles', [80, 85, 90, 95])
        volume_window = config.get('volume_window_hours', 4) * 60  # Convert to minutes
        
        parameters = {
            'volume_avg': estimation_data['volume'].mean(),
            'min_signal_events': config.get('min_signal_events', 20)
        }
        
        # Calculate spread percentiles
        for pct in percentiles:
            threshold_name = f'spread_{pct}pct'
            parameters[threshold_name] = estimation_data['spread_close'].quantile(pct / 100)
        
        return parameters
    
    def generate_signals(self, data: pd.DataFrame, parameters: Dict) -> pd.Series:
        """Generate spread expansion signals."""
        # Get configuration
        config = self.hypothesis_config
        target_percentile = config.get('target_percentile', 85)
        
        threshold_key = f'spread_{target_percentile}pct'
        
        if threshold_key not in parameters:
            raise ValueError(f"Missing threshold parameter: {threshold_key}")
        
        # Generate expansion signals
        expansion_condition = (
            (data['spread_close'] >= parameters[threshold_key]) &
            (data['volume'] > parameters['volume_avg'])
        )
        
        return expansion_condition
    
    def walk_forward_validation(self, parameters: Dict) -> Dict:
        """
        Perform walk-forward validation for spread expansion.
        Tests hypothesis stability over time.
        """
        # Configuration
        config = self.hypothesis_config
        walk_window_months = config.get('walk_window_months', 6)
        walk_step_months = config.get('walk_step_months', 1)
        
        # Convert to approximate rows (assuming 1-minute data)
        walk_window_rows = walk_window_months * 30 * 24 * 60
        walk_step_rows = walk_step_months * 30 * 24 * 60
        
        results = []
        
        # Start from enough data for meaningful estimation
        start_idx = walk_window_rows
        
        while start_idx + walk_step_rows < len(self.df):
            # Define periods
            train_start = start_idx - walk_window_rows
            train_end = start_idx
            test_start = start_idx
            test_end = start_idx + walk_step_rows
            
            # Get data
            train_data = self.df.iloc[train_start:train_end]
            test_data = self.df.iloc[test_start:test_end]
            
            if len(train_data) < 1000 or len(test_data) < 100:
                start_idx += walk_step_rows
                continue
            
            try:
                # Estimate parameters on training window
                walk_params = self.estimate_parameters(train_data)
                
                # Generate signals on test window
                walk_signals = self.generate_signals(test_data, walk_params)
                
                if walk_signals.sum() < 5:  # Need minimum signals
                    start_idx += walk_step_rows
                    continue
                
                # Test hypothesis on this window
                signal_events = test_data[walk_signals]
                signal_returns = signal_events['return_60m'].dropna()
                all_returns = test_data['return_60m'].dropna()
                
                if len(signal_returns) >= 3 and len(all_returns) >= 10:
                    # Statistical test
                    t_stat, p_value = stats.ttest_ind(signal_returns, all_returns)
                    
                    results.append({
                        'period_start': test_data.index.min(),
                        'period_end': test_data.index.max(),
                        'signal_count': len(signal_returns),
                        'mean_signal_return': signal_returns.mean(),
                        'mean_all_return': all_returns.mean(),
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    })
                    
            except Exception as e:
                # Log error but continue
                print(f"Walk-forward step failed: {e}")
            
            start_idx += walk_step_rows
        
        if not results:
            return {'error': 'No successful walk-forward steps completed'}
        
        # Analyze results
        walk_df = pd.DataFrame(results)
        success_rate = walk_df['significant'].mean()
        
        return {
            'total_steps': len(results),
            'significant_steps': walk_df['significant'].sum(),
            'success_rate': success_rate,
            'mean_signal_return': walk_df['mean_signal_return'].mean(),
            'mean_all_return': walk_df['mean_all_return'].mean(),
            'results_detail': results
        }
    
    def bootstrap_validation(self, parameters: Dict) -> Dict:
        """
        Bootstrap confidence intervals for effect size.
        """
        # Generate signals on validation data
        validation_signals = self.generate_signals(self.validation_df, parameters)
        signal_events = self.validation_df[validation_signals]
        
        # Get returns
        signal_returns = signal_events['return_60m'].dropna()
        all_returns = self.validation_df['return_60m'].dropna()
        
        if len(signal_returns) < 10:
            return {'error': 'Insufficient signal events for bootstrap'}
        
        # Bootstrap configuration
        n_bootstrap = self.hypothesis_config.get('bootstrap_samples', 1000)
        
        bootstrap_effects = []
        
        for i in range(n_bootstrap):
            # Resample with replacement
            boot_signal = np.random.choice(signal_returns, size=len(signal_returns), replace=True)
            boot_all = np.random.choice(all_returns, size=len(all_returns), replace=True)
            
            effect = boot_signal.mean() - boot_all.mean()
            bootstrap_effects.append(effect)
        
        bootstrap_effects = np.array(bootstrap_effects)
        
        # Calculate confidence intervals
        ci_lower = np.percentile(bootstrap_effects, 2.5)
        ci_upper = np.percentile(bootstrap_effects, 97.5)
        mean_effect = bootstrap_effects.mean()
        
        return {
            'mean_effect': mean_effect,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_excludes_zero': (ci_upper < 0) or (ci_lower > 0),
            'bootstrap_effects': bootstrap_effects.tolist()
        }


# Example usage and factory function
def create_spread_expansion_validator(dataset, config=None):
    """
    Factory function to create a configured spread expansion validator.
    
    Args:
        dataset: Data path or DataFrame
        config: Optional configuration overrides
        
    Returns:
        Configured SpreadExpansionValidator instance
    """
    default_config = {
        'spread_percentiles': [80, 85, 90, 95],
        'target_percentile': 85,
        'volume_window_hours': 4,
        'min_signal_events': 20,
        'forward_timeframes': [15, 30, 45, 60],
        'test_timeframes': [45, 60],
        'walk_window_months': 6,
        'walk_step_months': 1,
        'bootstrap_samples': 1000
    }
    
    if config:
        default_config.update(config)
    
    return SpreadExpansionValidator(dataset, default_config)


# Example usage
if __name__ == "__main__":
    # Example of how to use the spread expansion validator
    
    data_path = "../storage/data/capital_com/GBPUSD/raw/1m/2024-06-01_2025-05-23.csv"
    
    # Create validator with custom configuration
    config = {
        'target_percentile': 90,  # Test 90th percentile instead of 85th
        'test_timeframes': [60],  # Only test 60-minute returns
        'bootstrap_samples': 500  # Fewer bootstrap samples for speed
    }
    
    validator = create_spread_expansion_validator(data_path, config)
    
    # Run validation
    results = validator.run(validation_size=0.3)
    
    # Print results
    validator.print_validation_summary()
    
    # Save results
    validator.save_results()