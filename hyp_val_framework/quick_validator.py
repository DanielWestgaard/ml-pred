"""
Quick Hypothesis Validator

For rapid testing of simple hypotheses without creating full validator classes.
Uses the SimpleHypothesisValidator with configuration-driven approach.
"""

import pandas as pd
import numpy as np
from typing import Dict, Callable, Union
from hyp_val_framework.base_hypothesis_validator import SimpleHypothesisValidator


def quick_validate_hypothesis(
    dataset: Union[str, pd.DataFrame],
    name: str,
    description: str,
    signal_logic: Callable[[pd.DataFrame, Dict], pd.Series],
    parameter_logic: Callable[[pd.DataFrame], Dict],
    data_preparation: Callable[[pd.DataFrame], None] = None,
    config: Dict = None
) -> Dict:
    """
    Quick validation of a hypothesis without creating a full validator class.
    
    Args:
        dataset: Data path or DataFrame
        hypothesis_name: Name of the hypothesis
        hypothesis_description: Description of what the hypothesis tests
        signal_logic: Function that takes (data, parameters) and returns boolean Series
        parameter_logic: Function that takes estimation_data and returns parameter dict
        data_preparation: Optional function to prepare data
        config: Additional configuration
        
    Returns:
        Dictionary of validation results
    """
    
    hypothesis_config = {
        'name': name,
        'description': description,
        'signal_logic': signal_logic,
        'parameter_logic': parameter_logic,
        'forward_timeframes': [15, 30, 45, 60],
        'test_timeframes': [30, 45, 60],
    }
    
    if data_preparation:
        hypothesis_config['data_preparation'] = data_preparation
    
    if config:
        hypothesis_config.update(config)
    
    # Create and run validator
    validator = SimpleHypothesisValidator(dataset, hypothesis_config)
    results = validator.run()
    validator.print_validation_summary()
    
    return results


# Pre-built hypothesis templates
class HypothesisTemplates:
    """Collection of common hypothesis patterns for quick testing."""
    
    @staticmethod
    def volume_spike_reversion(volume_threshold_pct=95):
        """
        Template: Volume spike mean reversion
        "High volume spikes predict mean reversion"
        """
        def prepare_data(df):
            if 'volume' not in df.columns:
                raise ValueError("Volume column required")
        
        def estimate_params(estimation_data):
            return {
                'volume_threshold': estimation_data['volume'].quantile(volume_threshold_pct/100),
                'mean_return': estimation_data[[col for col in estimation_data.columns 
                                             if col.startswith('return_')]].mean().mean()
            }
        
        def generate_signals(data, params):
            return data['volume'] >= params['volume_threshold']
        
        return {
            'name': f'Volume Spike Mean Reversion ({volume_threshold_pct}th percentile)',
            'description': f'High volume spikes (>{volume_threshold_pct}th percentile) predict mean reversion',
            'signal_logic': generate_signals,
            'parameter_logic': estimate_params,
            'data_preparation': prepare_data
        }
    
    @staticmethod
    def price_momentum_continuation(lookback_periods=20, momentum_threshold_pct=80):
        """
        Template: Price momentum continuation
        "Strong recent momentum continues in the near term"
        """
        def prepare_data(df):
            price_col = None
            for col in ['Close', 'close', 'mid_price', 'price']:
                if col in df.columns:
                    price_col = col
                    break
            
            if not price_col:
                if 'bid_close' in df.columns and 'ask_close' in df.columns:
                    df['mid_price'] = (df['bid_close'] + df['ask_close']) / 2
                    price_col = 'mid_price'
                else:
                    raise ValueError("No price column found")
            
            # Calculate momentum
            df[f'momentum_{lookback_periods}'] = (
                df[price_col] / df[price_col].shift(lookback_periods) - 1
            )
        
        def estimate_params(estimation_data):
            momentum_col = f'momentum_{lookback_periods}'
            return {
                'momentum_threshold': estimation_data[momentum_col].quantile(momentum_threshold_pct/100),
                'momentum_col': momentum_col
            }
        
        def generate_signals(data, params):
            return data[params['momentum_col']] >= params['momentum_threshold']
        
        return {
            'name': f'Price Momentum Continuation ({lookback_periods}p, {momentum_threshold_pct}th pct)',
            'description': f'Strong {lookback_periods}-period momentum (>{momentum_threshold_pct}th percentile) continues',
            'signal_logic': generate_signals,
            'parameter_logic': estimate_params,
            'data_preparation': prepare_data
        }
    
    @staticmethod
    def volatility_regime_change(volatility_window=20, regime_threshold_pct=90):
        """
        Template: Volatility regime change
        "Volatility spikes predict increased future volatility"
        """
        def prepare_data(df):
            price_col = None
            for col in ['Close', 'close', 'mid_price', 'price']:
                if col in df.columns:
                    price_col = col
                    break
            
            if not price_col:
                raise ValueError("No price column found")
            
            # Calculate returns and volatility
            df['returns'] = df[price_col].pct_change()
            df[f'volatility_{volatility_window}'] = (
                df['returns'].rolling(window=volatility_window).std()
            )
        
        def estimate_params(estimation_data):
            vol_col = f'volatility_{volatility_window}'
            return {
                'volatility_threshold': estimation_data[vol_col].quantile(regime_threshold_pct/100),
                'volatility_col': vol_col
            }
        
        def generate_signals(data, params):
            return data[params['volatility_col']] >= params['volatility_threshold']
        
        return {
            'name': f'Volatility Regime Change ({volatility_window}p, {regime_threshold_pct}th pct)',
            'description': f'Volatility spikes (>{regime_threshold_pct}th percentile) predict regime change',
            'signal_logic': generate_signals,
            'parameter_logic': estimate_params,
            'data_preparation': prepare_data,
            'test_timeframes': [60, 120, 240]  # Test longer horizons for volatility
        }


# Convenience functions for common use cases
def test_volume_hypothesis(dataset, volume_threshold_pct=95):
    """Quick test of volume spike hypothesis."""
    template = HypothesisTemplates.volume_spike_reversion(volume_threshold_pct)
    return quick_validate_hypothesis(dataset, **template)

def test_momentum_hypothesis(dataset, lookback_periods=20, momentum_threshold_pct=80):
    """Quick test of momentum continuation hypothesis."""
    template = HypothesisTemplates.price_momentum_continuation(lookback_periods, momentum_threshold_pct)
    return quick_validate_hypothesis(dataset, **template)

def test_volatility_hypothesis(dataset, volatility_window=20, regime_threshold_pct=90):
    """Quick test of volatility regime change hypothesis."""
    template = HypothesisTemplates.volatility_regime_change(volatility_window, regime_threshold_pct)
    return quick_validate_hypothesis(dataset, **template)


# Example: Custom hypothesis testing
def test_custom_hypothesis_example():
    """Example of how to test a completely custom hypothesis."""
    
    data_path = "storage/data/capital_com/GBPUSD/raw/1m/2024-06-01_2025-05-23.csv"
    
    # Define custom hypothesis: "Price gaps predict continuation"
    def prepare_gap_data(df):
        """Calculate price gaps."""
        if 'bid_close' in df.columns and 'ask_close' in df.columns:
            df['mid_price'] = (df['bid_close'] + df['ask_close']) / 2
        elif 'Close' in df.columns:
            df['mid_price'] = df['Close']
        else:
            raise ValueError("No price data found")
        
        # Calculate gaps (current open vs previous close)
        if 'bid_open' in df.columns:
            df['gap'] = df['bid_open'] - df['mid_price'].shift(1)
            df['gap_pct'] = df['gap'] / df['mid_price'].shift(1)
        else:
            # Approximate gap using price change
            df['gap_pct'] = df['mid_price'].pct_change()
    
    def estimate_gap_params(estimation_data):
        """Estimate gap thresholds."""
        return {
            'positive_gap_threshold': estimation_data['gap_pct'].quantile(0.95),
            'negative_gap_threshold': estimation_data['gap_pct'].quantile(0.05),
        }
    
    def generate_gap_signals(data, params):
        """Generate gap continuation signals."""
        # Signal when we have significant gaps (positive or negative)
        return (
            (data['gap_pct'] >= params['positive_gap_threshold']) |
            (data['gap_pct'] <= params['negative_gap_threshold'])
        )
    
    # Test the hypothesis
    results = quick_validate_hypothesis(
        dataset=data_path,
        hypothesis_name="Price Gap Continuation",
        hypothesis_description="Significant price gaps (>95th or <5th percentile) predict continuation in the same direction",
        signal_logic=generate_gap_signals,
        parameter_logic=estimate_gap_params,
        data_preparation=prepare_gap_data,
        config={'test_timeframes': [5, 15, 30]}  # Test shorter timeframes for gaps
    )
    
    return results


if __name__ == "__main__":
    # Example usage of different hypothesis types
    
    data_path = "storage/data/capital_com/GBPUSD/raw/1m/2024-06-01_2025-05-23.csv"
    
    print("Testing Volume Spike Hypothesis...")
    volume_results = test_volume_hypothesis(data_path, volume_threshold_pct=90)
    
    print("\n" + "="*80)
    print("Testing Momentum Continuation Hypothesis...")
    momentum_results = test_momentum_hypothesis(data_path, lookback_periods=30, momentum_threshold_pct=85)
    
    print("\n" + "="*80)
    print("Testing Custom Gap Hypothesis...")
    gap_results = test_custom_hypothesis_example()
    
    # Compare results
    hypotheses = [
        ("Volume Spike", volume_results),
        ("Momentum", momentum_results), 
        ("Price Gaps", gap_results)
    ]
    
    print("\n" + "="*80)
    print("HYPOTHESIS COMPARISON")
    print("="*80)
    
    for name, results in hypotheses:
        score = results.get('validation_score', 0)
        signal_count = results.get('signal_count', 0)
        print(f"{name:20} | Score: {score:.2f} | Signals: {signal_count:5d} | {results.get('recommendation', 'N/A')}")