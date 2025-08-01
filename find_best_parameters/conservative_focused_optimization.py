#!/usr/bin/env python3
"""
Option B: Focused Optimization Around Conservative Parameters

Runs systematic optimization in a tight range around the best-performing 
Conservative parameters to find the optimal fine-tuned configuration.
"""

import os
import yaml
import subprocess
import pandas as pd
import numpy as np
import itertools
from datetime import datetime
from typing import Dict, List, Any
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConservativeFocusedOptimizer:
    """Focused optimization around Conservative baseline"""
    
    def __init__(self, base_config_path: str):
        self.base_config_path = base_config_path
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
    
    def define_conservative_parameter_grid(self) -> Dict[str, List]:
        """Define tight parameter grid around Conservative baseline"""
        
        # Conservative baseline: confidence_threshold=0.12, cooldown=1.0, etc.
        return {
            # Fine-tune confidence threshold around 0.12
            'confidence_threshold': [0.10, 0.11, 0.12, 0.13, 0.14],
            
            # Fine-tune cooldown around 1.0 minutes
            'cooldown_minutes': [0.75, 1.0, 1.25, 1.5],
            
            # Fine-tune signals per hour around 120
            'max_signals_per_hour': [100, 120, 140, 160],
            
            # Fine-tune stop loss around 0.010
            'stop_loss_pct': [0.008, 0.009, 0.010, 0.011, 0.012],
            
            # Fine-tune take profit around 0.025
            'take_profit_pct': [0.022, 0.024, 0.025, 0.026, 0.028],
            
            # Fine-tune position size around 0.06
            'max_position_size': [0.05, 0.06, 0.07],
            
            # Keep max concurrent positions conservative
            'max_concurrent_positions': [2, 3]
        }
    
    def create_test_config(self, params: Dict[str, Any]) -> str:
        """Create test configuration with specified parameters"""
        config = self.base_config.copy()
        
        # Update parameters
        config['backtesting']['signal_processing'].update({
            'confidence_threshold': params['confidence_threshold'],
            'cooldown_minutes': params['cooldown_minutes'],
            'max_signals_per_hour': params['max_signals_per_hour'],
            'min_signal_strength': params['confidence_threshold']
        })
        
        config['backtesting']['risk_management'].update({
            'stop_loss_pct': params['stop_loss_pct'],
            'take_profit_pct': params['take_profit_pct'],
            'max_position_size': params['max_position_size'],
            'max_concurrent_positions': params['max_concurrent_positions']
        })
        
        config['backtesting']['position_management']['max_concurrent_positions'] = params['max_concurrent_positions']
        
        # Save test config
        test_config_path = f"conservative_test_{self.timestamp}_{len(self.results)}.yml"
        with open(test_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return test_config_path
    
    def run_backtest(self, config_path: str) -> Dict[str, Any]:
        """Run single backtest and extract results"""
        start_time = time.time()
        
        try:
            result = subprocess.run(
                ['python', 'main.py', '--hyp', config_path],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                # Extract results from output
                output = result.stdout
                for line in reversed(output.split('\n')):
                    if "Final Results:" in line:
                        import re
                        return_match = re.search(r'Return:\s*([-+]?\d*\.?\d+)%', line)
                        trades_match = re.search(r'Trades:\s*(\d+)', line)
                        winrate_match = re.search(r'Win Rate:\s*([-+]?\d*\.?\d+)%', line)
                        
                        if all([return_match, trades_match, winrate_match]):
                            return {
                                'return_pct': float(return_match.group(1)),
                                'trades': int(trades_match.group(1)),
                                'win_rate': float(winrate_match.group(1)),
                                'runtime': time.time() - start_time,
                                'success': True
                            }
            
            return {'success': False, 'error': 'Could not parse results'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def calculate_optimization_score(self, result: Dict[str, Any]) -> float:
        """Calculate optimization score for Conservative strategy"""
        if not result['success'] or result['trades'] == 0:
            return -1000
        
        return_pct = result['return_pct']
        win_rate = result['win_rate']
        trades = result['trades']
        
        # Conservative strategy scoring: prioritize consistency and positive returns
        score = (
            return_pct * 15 +  # Heavy weight on returns
            (win_rate - 30) * 0.8 +  # Bonus for win rate above 30%
            min(trades / 5, 2) * 3 -  # Moderate trading activity reward
            max(0, 2 - trades) * 10  # Penalty for too few trades (<2)
        )
        
        # Bonus for positive returns
        if return_pct > 0:
            score += 5
        
        return score
    
    def optimize(self, max_combinations: int = 25) -> pd.DataFrame:
        """Run focused optimization"""
        logger.info("Starting Conservative focused optimization...")
        
        param_grid = self.define_conservative_parameter_grid()
        param_names = list(param_grid.keys())
        param_combinations = list(itertools.product(*param_grid.values()))
        
        # Limit combinations for efficiency
        if len(param_combinations) > max_combinations:
            logger.info(f"Sampling {max_combinations} from {len(param_combinations)} combinations")
            # Sample evenly across the space
            indices = np.linspace(0, len(param_combinations)-1, max_combinations, dtype=int)
            param_combinations = [param_combinations[i] for i in indices]
        
        logger.info(f"Testing {len(param_combinations)} focused parameter combinations")
        
        for i, param_values in enumerate(param_combinations):
            params = dict(zip(param_names, param_values))
            logger.info(f"Testing {i+1}/{len(param_combinations)}: {params}")
            
            config_path = self.create_test_config(params)
            
            try:
                result = self.run_backtest(config_path)
                
                if result['success']:
                    score = self.calculate_optimization_score(result)
                    
                    self.results.append({
                        'iteration': i + 1,
                        'score': score,
                        **params,
                        **{k: v for k, v in result.items() if k != 'success'}
                    })
                    
                    logger.info(f"Result: {result['return_pct']:.2f}%, {result['trades']} trades, "
                              f"{result['win_rate']:.1f}% win rate, Score: {score:.2f}")
                else:
                    logger.warning(f"Failed: {result.get('error', 'Unknown error')}")
                    
            finally:
                # Cleanup
                if os.path.exists(config_path):
                    os.remove(config_path)
        
        # Convert to DataFrame and sort by score
        results_df = pd.DataFrame(self.results)
        if not results_df.empty:
            results_df = results_df.sort_values('score', ascending=False)
        
        return results_df
    
    def save_optimal_config(self, best_params: Dict[str, Any]) -> str:
        """Save the optimal configuration"""
        optimal_config_path = self.create_test_config(best_params)
        final_path = f"conservative_optimal_{self.timestamp}.yml"
        os.rename(optimal_config_path, final_path)
        
        logger.info(f"Optimal Conservative config saved to: {final_path}")
        return final_path

def main():
    """Run Conservative focused optimization"""
    base_config = "hypothesis/spread_compression_patterns/config_backtest_v4_enhanced.yml"
    
    if not os.path.exists(base_config):
        logger.error(f"Base configuration not found: {base_config}")
        return
    
    print("🎯 CONSERVATIVE FOCUSED OPTIMIZATION")
    print("=" * 50)
    print("Fine-tuning around Conservative baseline parameters...")
    
    optimizer = ConservativeFocusedOptimizer(base_config)
    results_df = optimizer.optimize(max_combinations=25)
    
    if results_df.empty:
        logger.error("No successful backtests completed")
        return
    
    # Save results
    results_path = f"conservative_optimization_results_{optimizer.timestamp}.csv"
    results_df.to_csv(results_path, index=False)
    
    # Save optimal config
    best_params = results_df.iloc[0].to_dict()
    optimal_config_path = optimizer.save_optimal_config(best_params)
    
    # Display results
    print(f"\n📊 TOP 5 OPTIMIZED CONSERVATIVE CONFIGURATIONS")
    print("=" * 70)
    
    # Show key columns
    display_cols = ['confidence_threshold', 'cooldown_minutes', 'stop_loss_pct', 
                   'take_profit_pct', 'max_position_size', 'return_pct', 'trades', 'win_rate', 'score']
    print(results_df[display_cols].head().to_string(index=False))
    
    best = results_df.iloc[0]
    print(f"\n🏆 OPTIMAL CONSERVATIVE CONFIGURATION:")
    print(f"   File: {optimal_config_path}")
    print(f"   Return: {best['return_pct']:.2f}%")
    print(f"   Trades: {best['trades']}")
    print(f"   Win Rate: {best['win_rate']:.1f}%")
    print(f"   Score: {best['score']:.2f}")
    
    print(f"\n📈 PERFORMANCE COMPARISON:")
    print(f"   Original Conservative: -0.41% (3 trades)")
    print(f"   Optimized Conservative: {best['return_pct']:.2f}% ({best['trades']} trades)")
    
    if best['return_pct'] > -0.41:
        print(f"   🎉 IMPROVEMENT: +{best['return_pct'] + 0.41:.2f}% better!")
    else:
        print(f"   📊 Result: {best['return_pct'] + 0.41:.2f}% vs baseline")

if __name__ == "__main__":
    main()