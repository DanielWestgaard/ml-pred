import datetime
import os
import pandas as pd
import numpy as np
import json
import yaml
from typing import Dict, List, Any, Tuple
from datetime import datetime

from utils.logging_utils import get_logger
from utils.hypothesis.config_loader import load_hypothesis_config
from model.model_training import ModelTraining
from backtesting.backtester import BacktestOrchestrator
import config.config as config

logger = get_logger()

class WalkForwardValidator:
    """
    Walk-Forward Validation system that provides robust out-of-sample testing
    by training models on expanding windows and testing on future periods.
    
    This addresses temporal data leakage by ensuring strict chronological boundaries
    where models are never trained on future data relative to the test period.
    """
    
    def __init__(self, hypothesis_config_path: str, validation_config: Dict[str, Any] = None):
        """
        Initialize walk-forward validator.
        
        Args:
            hypothesis_config_path: Path to hypothesis configuration file
            validation_config: Dictionary with validation parameters
        """
        self.hypothesis_config_path = hypothesis_config_path
        self.validation_config = validation_config or {}
        
        # Load hypothesis configuration
        self.hypothesis_config = load_hypothesis_config(hypothesis_config_path).handle_hypothesis_source()
        
        # Validation parameters with defaults
        self.initial_training_months = self.validation_config.get('initial_training_months', 6)
        self.validation_period_months = self.validation_config.get('validation_period_months', 1)  
        self.step_forward_months = self.validation_config.get('step_forward_months', 1)
        self.min_training_samples = self.validation_config.get('min_training_samples', 50000)
        
        # Data and results storage
        self.data_path = None
        self.results = []
        self.validation_directory = None
        
        logger.info(f"Walk-Forward Validator initialized with {self.initial_training_months}m initial training, "
                   f"{self.validation_period_months}m validation periods, {self.step_forward_months}m steps")
    
    def load_data(self) -> pd.DataFrame:
        """Load and prepare data for walk-forward validation."""
        # Extract data path from hypothesis config
        self.data_path = self.hypothesis_config.get('processed_data') or self.hypothesis_config.get('model_training_processed_data')
        if not self.data_path:
            raise ValueError("No processed_data path found in hypothesis configuration")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        logger.info(f"Loading data from: {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # Ensure datetime index
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime')
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        elif df.index.name not in ['datetime', 'date']:
            df.index = pd.to_datetime(df.index)
        
        df = df.sort_index()
        logger.info(f"Loaded {len(df)} samples from {df.index.min()} to {df.index.max()}")
        
        return df
    
    def create_validation_periods(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Create validation periods with strict temporal boundaries.
        
        Args:
            data: Full dataset with datetime index
            
        Returns:
            List of validation period configurations
        """
        periods = []
        start_date = data.index.min()
        end_date = data.index.max()
        
        # Calculate initial training end
        initial_training_end = start_date + pd.DateOffset(months=self.initial_training_months)
        
        current_validation_start = initial_training_end + pd.DateOffset(days=1)
        
        period_id = 1
        while current_validation_start < end_date:
            # Validation period end
            validation_end = min(
                current_validation_start + pd.DateOffset(months=self.validation_period_months),
                end_date
            )
            
            # Training period (all data before validation start)
            training_end = current_validation_start - pd.DateOffset(days=1)
            
            # Check if we have sufficient data
            training_data = data.loc[start_date:training_end]
            validation_data = data.loc[current_validation_start:validation_end]
            
            if len(training_data) < self.min_training_samples:
                logger.warning(f"Insufficient training data ({len(training_data)} samples) for period {period_id}")
                break
                
            if len(validation_data) < 100:  # Minimum validation samples
                logger.warning(f"Insufficient validation data ({len(validation_data)} samples) for period {period_id}")
                break
            
            period_config = {
                'period_id': period_id,
                'training_start': start_date,
                'training_end': training_end, 
                'validation_start': current_validation_start,
                'validation_end': validation_end,
                'training_samples': len(training_data),
                'validation_samples': len(validation_data)
            }
            
            periods.append(period_config)
            logger.info(f"Period {period_id}: Train {start_date.strftime('%Y-%m-%d')} to "
                       f"{training_end.strftime('%Y-%m-%d')} ({len(training_data)} samples), "
                       f"Validate {current_validation_start.strftime('%Y-%m-%d')} to "
                       f"{validation_end.strftime('%Y-%m-%d')} ({len(validation_data)} samples)")
            
            # Move to next period
            current_validation_start = validation_end + pd.DateOffset(months=self.step_forward_months)
            period_id += 1
        
        logger.info(f"Created {len(periods)} validation periods")
        return periods
    
    def train_model_for_period(self, data: pd.DataFrame, period: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Train model for specific validation period.
        
        Args:
            data: Full dataset
            period: Period configuration
            
        Returns:
            Tuple of (model_directory, training_metadata)
        """
        period_id = period['period_id']
        logger.info(f"Training model for period {period_id}")
        
        # Extract training data with strict temporal boundaries
        training_data = data.loc[period['training_start']:period['training_end']].copy()
        
        # Create period-specific data file
        period_data_path = os.path.join(
            self.validation_directory,
            f"period_{period_id}",
            "training_data.csv"
        )
        os.makedirs(os.path.dirname(period_data_path), exist_ok=True)
        training_data.to_csv(period_data_path)
        
        # Create period-specific hypothesis config with proper structure
        period_hypothesis_config = {
            'hypothesis': {
                'name': f'walk_forward_validation_period_{period_id}'
            },
            'model_training': {
                'type': self.hypothesis_config.get('model_training_type', 'random_forest'),
                'processed_data': period_data_path,
                'target_column': self.hypothesis_config.get('model_training_target_column', 'target_spread_compression_v2_directional'),
                'parameters': self.hypothesis_config.get('model_training_parameters', {})
            }
        }
        
        period_config_path = os.path.join(
            self.validation_directory,
            f"period_{period_id}",
            "hypothesis_config.yml"
        )
        
        with open(period_config_path, 'w') as f:
            yaml.dump(period_hypothesis_config, f)
        
        # Train model using existing ModelTraining class
        model_trainer = ModelTraining(hypothesis_source=period_config_path)
        model_trainer.run()
        
        training_metadata = {
            'period_id': period_id,
            'training_start': period['training_start'].isoformat(),
            'training_end': period['training_end'].isoformat(),
            'training_samples': period['training_samples'],
            'model_directory': model_trainer.model_directory,
            'model_type': model_trainer.model_type,
            'target_column': model_trainer.target_col
        }
        
        logger.info(f"Model trained for period {period_id}: {model_trainer.model_directory}")
        return model_trainer.model_directory, training_metadata
    
    def validate_model_for_period(self, period: Dict[str, Any], 
                                model_directory: str) -> Dict[str, Any]:
        """
        Validate model on out-of-sample period using backtesting.
        
        Args:
            period: Period configuration
            model_directory: Path to trained model
            
        Returns:
            Validation results dictionary
        """
        period_id = period['period_id']
        logger.info(f"Validating model for period {period_id}")
        
        # Create validation-specific config for backtesting with proper structure
        validation_config = {
            'hypothesis': {
                'name': f'walk_forward_validation_backtest_period_{period_id}'
            },
            'backtesting': {
                'start_date': period['validation_start'].strftime('%Y-%m-%d'),
                'end_date': period['validation_end'].strftime('%Y-%m-%d'),
                'data_source': self.data_path,
                'initial_balance': 10000,
                'transaction_cost': 0.0001,
                'confidence_threshold': self.hypothesis_config.get('backtesting_confidence_threshold', 0.35),
                'max_position_size': 0.02,
                'max_daily_trades': 10
            }
        }
        
        # Save validation config
        validation_config_path = os.path.join(
            self.validation_directory,
            f"period_{period_id}",
            "validation_config.yml"
        )
        
        with open(validation_config_path, 'w') as f:
            yaml.dump(validation_config, f)
        
        # Run backtesting using existing BacktestOrchestrator
        backtester = BacktestOrchestrator(
            hypothesis_config_path=validation_config_path,
            model_folder=model_directory
        )
        
        validation_results = backtester.run_backtest()
        
        # Save validation results
        results_path = os.path.join(
            self.validation_directory,
            f"period_{period_id}",
            "validation_results.json"
        )
        
        with open(results_path, 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        
        logger.info(f"Validation completed for period {period_id}")
        return validation_results
    
    def run_validation(self) -> Dict[str, Any]:
        """
        Execute complete walk-forward validation process.
        
        Returns:
            Aggregate validation results
        """
        logger.info("Starting Walk-Forward Validation")
        
        # Create validation directory
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.validation_directory = os.path.join(
            config.BASE_STORAGE_DIR,
            'walk_forward_validation',
            f"{current_time}_validation"
        )
        os.makedirs(self.validation_directory, exist_ok=True)
        
        # Load data
        data = self.load_data()
        
        # Create validation periods
        periods = self.create_validation_periods(data)
        
        if not periods:
            raise ValueError("No valid validation periods could be created")
        
        # Execute validation for each period
        validation_results = []
        
        for period in periods:
            try:
                # Train model
                model_directory, training_metadata = self.train_model_for_period(data, period)
                
                # Validate model
                period_results = self.validate_model_for_period(period, model_directory)
                
                # Combine results
                combined_results = {
                    **training_metadata,
                    'validation_start': period['validation_start'].isoformat(),
                    'validation_end': period['validation_end'].isoformat(),
                    'validation_samples': period['validation_samples'],
                    'backtest_results': period_results
                }
                
                validation_results.append(combined_results)
                
            except Exception as e:
                logger.error(f"Error in period {period['period_id']}: {str(e)}")
                continue
        
        # Calculate aggregate metrics
        aggregate_results = self.calculate_aggregate_metrics(validation_results)
        
        # Save aggregate results
        aggregate_path = os.path.join(self.validation_directory, 'aggregate_results.json')
        with open(aggregate_path, 'w') as f:
            json.dump(aggregate_results, f, indent=2, default=str)
        
        logger.info(f"Walk-Forward Validation completed. Results saved to: {self.validation_directory}")
        return aggregate_results
    
    def calculate_aggregate_metrics(self, validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate performance metrics across all validation periods."""
        if not validation_results:
            return {
                'validation_summary': {
                    'total_periods': 0,
                    'successful_periods': 0
                },
                'performance_metrics': {
                    'mean_return': 0,
                    'median_return': 0,
                    'std_return': 0,
                    'mean_win_rate': 0,
                    'total_trades': 0,
                    'mean_sharpe_ratio': 0,
                    'worst_drawdown': 0
                },
                'consistency_metrics': {
                    'positive_periods': 0,
                    'negative_periods': 0,
                    'consistency_ratio': 0
                },
                'period_results': [],
                'error': 'No validation results to aggregate'
            }
        
        # Extract key metrics from each period
        returns = []
        win_rates = []
        trade_counts = []
        sharpe_ratios = []
        max_drawdowns = []
        
        for result in validation_results:
            backtest = result.get('backtest_results', {})
            
            if 'total_return' in backtest:
                returns.append(backtest['total_return'])
            if 'win_rate' in backtest:
                win_rates.append(backtest['win_rate'])
            if 'total_trades' in backtest:
                trade_counts.append(backtest['total_trades'])
            if 'sharpe_ratio' in backtest:
                sharpe_ratios.append(backtest['sharpe_ratio'])
            if 'max_drawdown' in backtest:
                max_drawdowns.append(backtest['max_drawdown'])
        
        aggregate_metrics = {
            'validation_summary': {
                'total_periods': len(validation_results),
                'successful_periods': len([r for r in validation_results if 'backtest_results' in r])
            },
            'performance_metrics': {
                'mean_return': np.mean(returns) if returns else 0,
                'median_return': np.median(returns) if returns else 0,
                'std_return': np.std(returns) if returns else 0,
                'mean_win_rate': np.mean(win_rates) if win_rates else 0,
                'total_trades': sum(trade_counts) if trade_counts else 0,
                'mean_sharpe_ratio': np.mean(sharpe_ratios) if sharpe_ratios else 0,
                'worst_drawdown': min(max_drawdowns) if max_drawdowns else 0
            },
            'consistency_metrics': {
                'positive_periods': len([r for r in returns if r > 0]) if returns else 0,
                'negative_periods': len([r for r in returns if r < 0]) if returns else 0,
                'consistency_ratio': len([r for r in returns if r > 0]) / len(returns) if returns else 0
            },
            'period_results': validation_results
        }
        
        logger.info(f"Aggregate Results: Mean Return: {aggregate_metrics['performance_metrics']['mean_return']:.4f}, "
                   f"Consistency: {aggregate_metrics['consistency_metrics']['consistency_ratio']:.2%}")
        
        return aggregate_metrics


def create_walk_forward_config(
    initial_training_months: int = 6,
    validation_period_months: int = 1,
    step_forward_months: int = 1,
    min_training_samples: int = 50000
) -> Dict[str, Any]:
    """
    Helper function to create walk-forward validation configuration.
    
    Args:
        initial_training_months: Initial training period in months
        validation_period_months: Length of each validation period in months
        step_forward_months: Step size between validation periods in months
        min_training_samples: Minimum number of training samples required
        
    Returns:
        Walk-forward validation configuration dictionary
    """
    return {
        'initial_training_months': initial_training_months,
        'validation_period_months': validation_period_months,
        'step_forward_months': step_forward_months,
        'min_training_samples': min_training_samples
    }