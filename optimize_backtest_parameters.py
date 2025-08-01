#!/usr/bin/env python3
"""
Systematic Parameter Optimization for Backtesting Configuration

This script performs grid search optimization on backtesting parameters to find
the optimal combination for maximum risk-adjusted returns.

Usage: python optimize_backtest_parameters.py
"""

import os
import json
import yaml
import itertools
import subprocess
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BacktestOptimizer:
    def __init__(self, base_config_path: str, model_path: str):
        self.base_config_path = base_config_path
        self.model_path = model_path
        self.results = []
        self.optimization_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load base configuration
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
    
    def define_parameter_grid(self) -> Dict[str, List]:
        """Define the parameter search space"""
        return {
            # Signal Processing Parameters
            'confidence_threshold': [0.06, 0.08, 0.10, 0.12, 0.15],
            'cooldown_minutes': [0.25, 0.5, 1.0, 2.0],
            'max_signals_per_hour': [120, 180, 240, 300],
            
            # Risk Management Parameters
            'stop_loss_pct': [0.008, 0.010, 0.012, 0.015],
            'take_profit_pct': [0.015, 0.020, 0.025, 0.030],
            'max_position_size': [0.04, 0.06, 0.08, 0.10],
            
            # Position Management
            'max_concurrent_positions': [2, 3, 4],
        }
    
    def create_config_variant(self, params: Dict[str, Any]) -> str:
        """Create a configuration file with specified parameters"""
        config = self.base_config.copy()
        
        # Update signal processing parameters
        config['backtesting']['signal_processing']['confidence_threshold'] = params['confidence_threshold']
        config['backtesting']['signal_processing']['cooldown_minutes'] = params['cooldown_minutes']
        config['backtesting']['signal_processing']['max_signals_per_hour'] = params['max_signals_per_hour']
        config['backtesting']['signal_processing']['min_signal_strength'] = params['confidence_threshold']  # Match threshold
        
        # Update risk management parameters
        config['backtesting']['risk_management']['stop_loss_pct'] = params['stop_loss_pct']
        config['backtesting']['risk_management']['take_profit_pct'] = params['take_profit_pct']
        config['backtesting']['risk_management']['max_position_size'] = params['max_position_size']
        config['backtesting']['risk_management']['max_concurrent_positions'] = params['max_concurrent_positions']
        
        # Update position management to match
        config['backtesting']['position_management']['max_concurrent_positions'] = params['max_concurrent_positions']
        
        # Create temporary config file
        temp_config_path = f"temp_config_{self.optimization_timestamp}_{len(self.results)}.yml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return temp_config_path
    
    def run_backtest(self, config_path: str) -> Dict[str, Any]:
        """Run a single backtest and extract results"""
        try:
            # Run backtest
            result = subprocess.run(
                ['python', 'main.py', '--hyp', config_path],
                capture_output=True,
                text=True,
                timeout=300  # 5-minute timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Backtest failed: {result.stderr}")
                return None
            
            # Parse results from output
            output_lines = result.stdout.split('\n')
            for line in reversed(output_lines):  # Search from end
                if "Final Results:" in line:
                    # Extract return, trades, win rate
                    parts = line.split('|')
                    return_str = [p.strip() for p in parts if 'Return:' in p][0]
                    trades_str = [p.strip() for p in parts if 'Trades:' in p][0]
                    winrate_str = [p.strip() for p in parts if 'Win Rate:' in p][0]
                    
                    return {
                        'return_pct': float(return_str.split('Return: ')[1].replace('%', '').replace(',', '')),
                        'total_trades': int(trades_str.split('Trades: ')[1].replace(',', '')),
                        'win_rate': float(winrate_str.split('Win Rate: ')[1].replace('%', ''))
                    }
            
            logger.warning("Could not parse backtest results")
            return None
            
        except subprocess.TimeoutExpired:
            logger.error("Backtest timed out")
            return None
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return None
    
    def calculate_fitness_score(self, results: Dict[str, Any]) -> float:
        """Calculate fitness score for parameter combination"""
        if not results or results['total_trades'] == 0:
            return -1000  # Heavily penalize no trading
        
        return_pct = results['return_pct']
        win_rate = results['win_rate']
        total_trades = results['total_trades']
        
        # Fitness formula: prioritize positive returns, good win rate, adequate trading
        fitness = (
            return_pct * 10 +  # Return weight
            (win_rate - 33) * 0.5 +  # Win rate bonus above 33%
            min(total_trades / 10, 5) * 2 -  # Trading activity (capped)
            max(0, 50 - total_trades) * 0.1  # Penalty for too few trades
        )
        
        return fitness
    
    def optimize(self, max_iterations: int = 50) -> pd.DataFrame:
        """Run parameter optimization"""
        logger.info("Starting parameter optimization...")
        
        param_grid = self.define_parameter_grid()
        param_names = list(param_grid.keys())
        param_combinations = list(itertools.product(*param_grid.values()))
        
        # Limit combinations if too many
        if len(param_combinations) > max_iterations:
            logger.info(f"Limiting to {max_iterations} combinations from {len(param_combinations)} total")
            param_combinations = param_combinations[:max_iterations]
        
        logger.info(f"Testing {len(param_combinations)} parameter combinations")
        
        for i, param_values in enumerate(param_combinations):
            params = dict(zip(param_names, param_values))
            logger.info(f"Testing combination {i+1}/{len(param_combinations)}: {params}")
            
            # Create config and run backtest
            temp_config = self.create_config_variant(params)
            
            try:
                backtest_results = self.run_backtest(temp_config)
                
                if backtest_results:
                    fitness = self.calculate_fitness_score(backtest_results)
                    
                    result = {
                        'iteration': i + 1,
                        'fitness_score': fitness,
                        **params,
                        **backtest_results
                    }
                    
                    self.results.append(result)
                    logger.info(f"Result: Return={backtest_results['return_pct']:.2f}%, "
                              f"Trades={backtest_results['total_trades']}, "
                              f"WinRate={backtest_results['win_rate']:.1f}%, "
                              f"Fitness={fitness:.2f}")
                else:
                    logger.warning(f"Failed to get results for combination {i+1}")
                
            finally:
                # Clean up temp config
                if os.path.exists(temp_config):
                    os.remove(temp_config)
        
        # Convert to DataFrame and sort by fitness
        results_df = pd.DataFrame(self.results)
        if not results_df.empty:
            results_df = results_df.sort_values('fitness_score', ascending=False)
        
        return results_df
    
    def save_results(self, results_df: pd.DataFrame):
        """Save optimization results"""
        timestamp = self.optimization_timestamp
        
        # Save to CSV
        csv_path = f"optimization_results_{timestamp}.csv"
        results_df.to_csv(csv_path, index=False)
        
        # Save best parameters to YAML
        if not results_df.empty:
            best_params = results_df.iloc[0].to_dict()
            best_config_path = f"optimal_config_{timestamp}.yml"
            
            # Create optimal configuration
            optimal_config = self.create_config_variant(best_params)
            os.rename(optimal_config, best_config_path)
            
            logger.info(f"Results saved to: {csv_path}")
            logger.info(f"Best configuration saved to: {best_config_path}")
            logger.info(f"Best fitness score: {best_params['fitness_score']:.2f}")
            logger.info(f"Best return: {best_params['return_pct']:.2f}%")
        
        return results_df

def main():
    """Main optimization execution"""
    # Configuration
    base_config = "hypothesis/spread_compression_patterns/config_backtest_v4_enhanced.yml"
    model_path = "storage/models/2025-08-01_15:16:41"
    
    if not os.path.exists(base_config):
        logger.error(f"Base configuration not found: {base_config}")
        return
    
    # Run optimization
    optimizer = BacktestOptimizer(base_config, model_path)
    results_df = optimizer.optimize(max_iterations=30)  # Start with 30 combinations
    
    if results_df.empty:
        logger.error("No successful backtests completed")
        return
    
    # Save and display results
    optimizer.save_results(results_df)
    
    # Display top 5 results
    print("\n" + "="*80)
    print("TOP 5 PARAMETER COMBINATIONS")
    print("="*80)
    print(results_df.head().to_string(index=False))
    
    print(f"\n📊 OPTIMIZATION COMPLETE")
    print(f"Best Return: {results_df.iloc[0]['return_pct']:.2f}%")
    print(f"Best Win Rate: {results_df.iloc[0]['win_rate']:.1f}%")
    print(f"Best Trades: {results_df.iloc[0]['total_trades']}")

if __name__ == "__main__":
    main()