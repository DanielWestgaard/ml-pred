"""
Risk Management Parameter Optimization System
============================================

Systematic testing framework to optimize backtesting parameters and identify
why profitable signals are producing 0% win rates in backtesting.

This tests:
1. Stop-loss and take-profit combinations
2. Confidence threshold sensitivity
3. Signal processor parameters
4. Entry/exit timing logic
5. Market regime compatibility

Usage:
    python risk_management_optimizer.py <config_file> [--quick-test]
"""

import json
import os
import time
import pandas as pd
import numpy as np
import yaml
import itertools
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

from utils.logging_utils import get_logger
logger = get_logger()

class RiskManagementOptimizer:
    """
    Comprehensive risk management parameter optimization system
    """
    
    def __init__(self, base_config_path: str):
        """
        Initialize optimizer with base backtesting configuration
        
        Args:
            base_config_path: Path to base backtesting config file
        """
        self.base_config_path = base_config_path
        self.base_config = self._load_base_config()
        self.optimization_results = {}
        
        logger.info(f"RiskManagementOptimizer initialized with config: {base_config_path}")
    
    def _load_base_config(self) -> Dict:
        """Load base configuration file"""
        try:
            with open(self.base_config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Failed to load base config: {e}")
            raise
    
    def generate_test_parameter_grid(self, test_type: str = 'comprehensive') -> List[Dict]:
        """
        Generate parameter combinations to test
        
        Args:
            test_type: 'quick', 'comprehensive', or 'focused'
        
        Returns:
            List of parameter dictionaries to test
        """
        if test_type == 'quick':
            return self._generate_quick_grid()
        elif test_type == 'comprehensive':
            return self._generate_comprehensive_grid()
        elif test_type == 'focused':
            return self._generate_focused_grid()
        else:
            raise ValueError(f"Unknown test_type: {test_type}")
    
    def _generate_quick_grid(self) -> List[Dict]:
        """Generate minimal parameter grid for quick testing"""
        
        # Focus on confidence threshold issue - model producing 0.0000 confidence
        grid_params = {
            'confidence_threshold': [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],  # MUCH lower thresholds
            'stop_loss_pct': [0.05, 0.1, 0.2],          # Tighter stops for small margins
            'take_profit_pct': [0.1, 0.2, 0.4],         # Conservative ratios
            'min_confidence_delta': [0.0, 0.01],        # Current and slightly higher
            'cooldown_minutes': [5, 10],                 # Current and faster
            'max_signals_per_hour': [4, 8, 12]          # Allow more signals if confidence is very low
        }
        
        # Generate combinations (3*3*3*2*2*2 = 216 tests)
        combinations = []
        for params in itertools.product(*grid_params.values()):
            param_dict = dict(zip(grid_params.keys(), params))
            combinations.append(param_dict)
        
        logger.info(f"Generated {len(combinations)} quick test combinations")
        return combinations
    
    def _generate_comprehensive_grid(self) -> List[Dict]:
        """Generate comprehensive parameter grid"""
        
        grid_params = {
            'confidence_threshold': [0.40, 0.45, 0.48, 0.50, 0.52, 0.55],
            'stop_loss_pct': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
            'take_profit_pct': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0],
            'min_confidence_delta': [0.0, 0.01, 0.02, 0.03],
            'cooldown_minutes': [1, 5, 10, 15, 20],
            'max_signals_per_hour': [2, 4, 6, 8, 12, 20]
        }
        
        # This would generate ~46k combinations - too many for practical testing
        # Instead, use smart sampling
        combinations = self._smart_sample_grid(grid_params, max_combinations=500)
        
        logger.info(f"Generated {len(combinations)} comprehensive test combinations")
        return combinations
    
    def _generate_focused_grid(self) -> List[Dict]:
        """Generate focused grid based on profitability analysis insights"""
        
        # Based on your results: SELL slightly profitable, BUY slightly negative
        # Focus on asymmetric parameters and very tight risk management
        grid_params = {
            'confidence_threshold': [0.47, 0.485, 0.49, 0.495, 0.50],  # Around current
            'stop_loss_pct': [0.03, 0.05, 0.08, 0.1],                 # Very tight stops
            'take_profit_pct': [0.06, 0.1, 0.15, 0.2],                # Conservative targets
            'min_confidence_delta': [0.0, 0.005, 0.01],               # Fine-tuned deltas
            'cooldown_minutes': [5, 10, 15],                           # Moderate cooling
            'max_signals_per_hour': [4, 6, 8],                        # Quality over quantity
            
            # Additional focused parameters
            'risk_reward_ratio': [1.5, 2.0, 2.5, 3.0],               # R:R ratios
            'trailing_stop': [False, True],                            # Trailing stops
            'breakeven_after_pct': [0.1, 0.15, None]                 # Move to breakeven
        }
        
        combinations = self._smart_sample_grid(grid_params, max_combinations=300)
        
        logger.info(f"Generated {len(combinations)} focused test combinations")
        return combinations
    
    def _smart_sample_grid(self, grid_params: Dict, max_combinations: int) -> List[Dict]:
        """
        Intelligently sample parameter combinations using:
        1. Full factorial for key parameters
        2. Random sampling for secondary parameters
        3. Ensure coverage of extreme values
        """
        import random
        random.seed(42)  # Reproducible results
        
        # Identify key parameters (most likely to affect results)
        key_params = ['confidence_threshold', 'stop_loss_pct', 'take_profit_pct']
        secondary_params = {k: v for k, v in grid_params.items() if k not in key_params}
        
        combinations = []
        
        # Full factorial on key parameters
        key_combinations = list(itertools.product(*[grid_params[k] for k in key_params]))
        
        for key_combo in key_combinations:
            key_dict = dict(zip(key_params, key_combo))
            
            # Add random secondary parameter combinations
            secondary_combo = {}
            for param, values in secondary_params.items():
                secondary_combo[param] = random.choice(values)
            
            combined = {**key_dict, **secondary_combo}
            combinations.append(combined)
            
            if len(combinations) >= max_combinations:
                break
        
        # Ensure we have extreme value combinations
        extreme_combinations = self._generate_extreme_combinations(grid_params)
        combinations.extend(extreme_combinations[:max(50, max_combinations//10)])
        
        return combinations[:max_combinations]
    
    def _generate_extreme_combinations(self, grid_params: Dict) -> List[Dict]:
        """Generate combinations with extreme parameter values"""
        extremes = []
        
        # Very conservative (tight stops, low confidence, high R:R)
        conservative = {}
        for param, values in grid_params.items():
            if 'stop_loss' in param or 'confidence' in param:
                conservative[param] = min(values)
            elif 'take_profit' in param or 'risk_reward' in param:
                conservative[param] = max(values)
            else:
                conservative[param] = values[len(values)//2]  # Middle value
        extremes.append(conservative)
        
        # Very aggressive (loose stops, high confidence, low R:R)
        aggressive = {}
        for param, values in grid_params.items():
            if 'stop_loss' in param:
                aggressive[param] = max(values)
            elif 'confidence' in param:
                aggressive[param] = max(values)
            elif 'take_profit' in param:
                aggressive[param] = min(values)
            else:
                aggressive[param] = values[len(values)//2]
        extremes.append(aggressive)
        
        return extremes
    
    def create_test_config(self, base_config: Dict, param_overrides: Dict) -> Dict:
        """
        Create test configuration by overlaying parameter overrides
        
        Args:
            base_config: Base configuration
            param_overrides: Parameters to override
        
        Returns:
            Modified configuration for testing
        """
        import copy
        test_config = copy.deepcopy(base_config)
        
        # Map parameters to configuration paths (based on your config structure)
        param_mapping = {
            'confidence_threshold': ['backtesting', 'signal_processing', 'confidence_threshold'],
            'stop_loss_pct': ['backtesting', 'risk_management', 'stop_loss_pct'],
            'take_profit_pct': ['backtesting', 'risk_management', 'take_profit_pct'],
            'min_confidence_delta': ['backtesting', 'signal_processing', 'min_confidence_delta'],
            'cooldown_minutes': ['backtesting', 'signal_processing', 'cooldown_minutes'],
            'max_signals_per_hour': ['backtesting', 'signal_processing', 'max_signals_per_hour'],
            'risk_reward_ratio': ['backtesting', 'risk_management', 'risk_reward_ratio'],
            'trailing_stop': ['backtesting', 'risk_management', 'use_trailing_stop'],
            'breakeven_after_pct': ['backtesting', 'risk_management', 'breakeven_after_pct'],
            'min_signal_strength': ['backtesting', 'signal_processing', 'min_signal_strength'],
            'max_position_size': ['backtesting', 'risk_management', 'max_position_size'],
            'max_concurrent_positions': ['backtesting', 'risk_management', 'max_concurrent_positions'],
            'transaction_costs': ['backtesting', 'transaction_costs'],
            'slippage': ['backtesting', 'slippage']
        }
        
        # Apply parameter overrides
        for param, value in param_overrides.items():
            if param in param_mapping:
                config_path = param_mapping[param]
                
                # Navigate to the correct nested location
                current_dict = test_config
                for key in config_path[:-1]:
                    if key not in current_dict:
                        current_dict[key] = {}
                    current_dict = current_dict[key]
                
                # Set the value
                current_dict[config_path[-1]] = value
            else:
                logger.warning(f"Unknown parameter for mapping: {param}")
        
        return test_config
    
    def run_optimization_batch(self, parameter_combinations: List[Dict], 
                             max_concurrent: int = 4) -> Dict:
        """
        Run optimization batch with parameter combinations
        
        Args:
            parameter_combinations: List of parameter dictionaries to test
            max_concurrent: Maximum concurrent backtests to run
        
        Returns:
            Dictionary of results keyed by parameter combination
        """
        logger.info(f"Starting optimization batch with {len(parameter_combinations)} combinations")
        
        results = {}
        
        for i, params in enumerate(parameter_combinations):
            logger.info(f"Testing combination {i+1}/{len(parameter_combinations)}: {params}")
            
            try:
                # Create test configuration
                test_config = self.create_test_config(self.base_config, params)
                
                # Run backtest (you'll need to implement this based on your backtesting system)
                result = self._run_single_backtest(test_config, params)
                
                # Store results
                param_key = self._params_to_key(params)
                results[param_key] = {
                    'parameters': params,
                    'results': result,
                    'config': test_config
                }
                
                # Log key metrics
                if result:
                    win_rate = result.get('win_rate', 0)
                    total_return = result.get('total_return_pct', 0)
                    total_trades = result.get('total_trades', 0)
                    equity = result.get('equity', 0)
                    
                    logger.info(f"  Result: {win_rate:.1f}% win rate, "
                              f"{total_return:+.2f}% return, {total_trades} trades, "
                              f"${equity:.2f} equity")
                    
                    # Early stopping for clearly good results
                    if win_rate > 60 and total_return > 5 and total_trades > 50:
                        logger.info(f"🎯 EXCELLENT RESULT FOUND! Params: {params}")
                
            except Exception as e:
                logger.error(f"Failed to test combination {i+1}: {e}")
                results[self._params_to_key(params)] = {
                    'parameters': params,
                    'results': None,
                    'error': str(e)
                }
        
        self.optimization_results = results
        return results
    
    def _run_single_backtest(self, test_config: Dict, params: Dict) -> Optional[Dict]:
        """
        Run single backtest with test configuration using your backtesting system
        """
        import tempfile
        import subprocess
        import os
        
        try:
            # Create temporary config file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
                yaml.dump(test_config, f, default_flow_style=False)
                temp_config_path = f.name
            
            # Run backtesting using your main.py command
            cmd = [
                'python', 'main.py', 
                '--hyp', temp_config_path
            ]
            
            logger.debug(f"Running backtest command: {' '.join(cmd)}")
            
            # Execute backtest with timeout
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=300,  # 5 minute timeout per backtest
                cwd='/Users/dmartinsen/pers-prog/ml-pred'
            )
            
            # Clean up temporary file
            try:
                os.unlink(temp_config_path)
            except:
                pass
            
            if result.returncode != 0:
                logger.error(f"Backtest failed with return code {result.returncode}")
                logger.error(f"STDERR: {result.stderr}")
                return None
            
            # Parse results from output or find results file
            backtest_results = self._parse_backtest_results(result.stdout, result.stderr, params)
            
            return backtest_results
            
        except subprocess.TimeoutExpired:
            logger.error(f"Backtest timed out for params: {params}")
            return None
        except Exception as e:
            logger.error(f"Failed to run backtest: {e}")
            return None
    
    def _parse_backtest_results(self, stdout: str, stderr: str, params: Dict) -> Optional[Dict]:
        """
        Parse backtest results from output or results files
        """
        try:
            # Method 1: Parse from stdout/stderr
            results = self._parse_output_results(stdout, stderr)
            if results:
                return results
            
            # Method 2: Look for results files (your system might save results to files)
            results = self._parse_results_files(params)
            if results:
                return results
            
            # Method 3: Parse log files for results
            results = self._parse_log_results()
            if results:
                return results
                
            logger.warning("Could not parse backtest results from any source")
            return None
            
        except Exception as e:
            logger.error(f"Failed to parse backtest results: {e}")
            return None
    
    def _parse_output_results(self, stdout: str, stderr: str) -> Optional[Dict]:
        """Parse results from command output"""
        try:
            # Look for key metrics in the output
            import re
            
            # Patterns for your backtesting system output
            patterns = {
                'total_return_pct': r'Return:\s*([+-]?\d+\.?\d*)%',  # "Return: -0.01%"
                'win_rate': r'Win Rate:\s*(\d+\.?\d*)%',            # "Win Rate: 0.0%"
                'total_trades': r'Trades:\s*(\d+)',                # "Trades: 2"
                'sharpe_ratio': r'Sharpe.*?([+-]?\d+\.?\d*)',
                'max_drawdown_pct': r'Max Drawdown.*?([+-]?\d+\.?\d*)%',
                'profitable_trades': r'Profitable.*?(\d+)',
                'losing_trades': r'Losing.*?(\d+)',
                'iterations': r'Iterations:\s*(\d+)/(\d+)',        # "Iterations: 3000/3000"
                'equity': r'Equity:\s*\$([+-]?\d+\.?\d*)'          # "Equity: $9999.11"
            }
            
            results = {}
            full_output = stdout + "\n" + stderr
            
            for metric, pattern in patterns.items():
                match = re.search(pattern, full_output, re.IGNORECASE)
                if match:
                    try:
                        results[metric] = float(match.group(1))
                    except ValueError:
                        results[metric] = match.group(1)
            
            # Calculate derived metrics
            if 'profitable_trades' in results and 'losing_trades' in results:
                total = results['profitable_trades'] + results['losing_trades']
                if total > 0:
                    results['win_rate'] = results['profitable_trades'] / total * 100
                    results['total_trades'] = total
            
            # Return results if we found key metrics
            if len(results) >= 2:  # Need at least 2 metrics
                logger.debug(f"Parsed results from output: {results}")
                return results
                
            return None
            
        except Exception as e:
            logger.error(f"Failed to parse output results: {e}")
            return None
    
    def _parse_results_files(self, params: Dict) -> Optional[Dict]:
        """Parse results from saved results files"""
        try:
            # Look for recent results files in common locations
            import glob
            from pathlib import Path
            
            # Common result file locations
            search_paths = [
                'storage/backtesting/results/*.json',
                'storage/backtesting/results/*.csv',
                'logs/backtesting/*.json',
                'results/*.json'
            ]
            
            latest_file = None
            latest_time = 0
            
            for pattern in search_paths:
                files = glob.glob(pattern)
                for file_path in files:
                    file_time = os.path.getmtime(file_path)
                    if file_time > latest_time:
                        latest_time = file_time
                        latest_file = file_path
            
            if latest_file and (time.time() - latest_time) < 60:  # File created in last minute
                logger.debug(f"Found recent results file: {latest_file}")
                
                if latest_file.endswith('.json'):
                    with open(latest_file, 'r') as f:
                        data = json.load(f)
                    return self._extract_metrics_from_json(data)
                
                elif latest_file.endswith('.csv'):
                    df = pd.read_csv(latest_file)
                    return self._extract_metrics_from_dataframe(df)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to parse results files: {e}")
            return None
    
    def _parse_log_results(self) -> Optional[Dict]:
        """Parse results from recent log files"""
        try:
            # Look for recent log files
            log_patterns = [
                'logs/logging_*.log',
                'logs/backtesting_*.log'
            ]
            
            latest_log = None
            latest_time = 0
            
            for pattern in log_patterns:
                import glob
                files = glob.glob(pattern)
                for file_path in files:
                    file_time = os.path.getmtime(file_path)
                    if file_time > latest_time:
                        latest_time = file_time
                        latest_log = file_path
            
            if latest_log and (time.time() - latest_time) < 120:  # Log updated in last 2 minutes
                with open(latest_log, 'r') as f:
                    log_content = f.read()
                
                # Parse log content for results
                return self._parse_output_results(log_content, "")
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to parse log results: {e}")
            return None
    
    def _extract_metrics_from_json(self, data: Dict) -> Dict:
        """Extract standard metrics from JSON results"""
        metrics = {}
        
        # Common field mappings
        field_mappings = {
            'total_return_pct': ['total_return', 'total_return_pct', 'return_pct', 'pnl_pct'],
            'win_rate': ['win_rate', 'win_percentage', 'success_rate'],
            'total_trades': ['total_trades', 'trade_count', 'num_trades'],
            'sharpe_ratio': ['sharpe_ratio', 'sharpe'],
            'max_drawdown_pct': ['max_drawdown', 'max_drawdown_pct', 'drawdown']
        }
        
        for metric, possible_fields in field_mappings.items():
            for field in possible_fields:
                if field in data:
                    metrics[metric] = data[field]
                    break
        
        return metrics if metrics else None
    
    def _extract_metrics_from_dataframe(self, df: pd.DataFrame) -> Dict:
        """Extract metrics from results DataFrame"""
        if len(df) == 0:
            return None
        
        metrics = {}
        
        # Calculate metrics from trade-level data if available
        if 'pnl' in df.columns:
            total_return = df['pnl'].sum()
            metrics['total_return_pct'] = total_return
            
            profitable_trades = (df['pnl'] > 0).sum()
            total_trades = len(df)
            metrics['win_rate'] = profitable_trades / total_trades * 100 if total_trades > 0 else 0
            metrics['total_trades'] = total_trades
            
            # Calculate Sharpe ratio if we have enough trades
            if total_trades > 10:
                returns = df['pnl']
                if returns.std() > 0:
                    metrics['sharpe_ratio'] = returns.mean() / returns.std() * np.sqrt(252)  # Annualized
        
        return metrics if metrics else None
    
    def _params_to_key(self, params: Dict) -> str:
        """Convert parameters dictionary to string key"""
        sorted_items = sorted(params.items())
        return "_".join([f"{k}={v}" for k, v in sorted_items])
    
    def analyze_results(self) -> Dict:
        """
        Analyze optimization results to identify best parameters
        """
        if not self.optimization_results:
            logger.error("No optimization results available")
            return {}
        
        logger.info("Analyzing optimization results...")
        
        # Extract successful results
        successful_results = {}
        for key, result_data in self.optimization_results.items():
            if result_data.get('results') and result_data['results'].get('total_trades', 0) > 10:
                successful_results[key] = result_data
        
        if not successful_results:
            logger.warning("No successful results found")
            return {'successful_count': 0}
        
        # Find best performers by different metrics
        analysis = {
            'total_tested': len(self.optimization_results),
            'successful_count': len(successful_results),
            'best_win_rate': self._find_best_by_metric(successful_results, 'win_rate'),
            'best_return': self._find_best_by_metric(successful_results, 'total_return_pct'),
            'best_sharpe': self._find_best_by_metric(successful_results, 'sharpe_ratio'),
            'parameter_sensitivity': self._analyze_parameter_sensitivity(successful_results)
        }
        
        return analysis
    
    def _find_best_by_metric(self, results: Dict, metric: str) -> Dict:
        """Find best result by specific metric"""
        best_key = None
        best_value = float('-inf')
        
        for key, result_data in results.items():
            value = result_data['results'].get(metric, float('-inf'))
            if value > best_value:
                best_value = value
                best_key = key
        
        if best_key:
            return {
                'key': best_key,
                'value': best_value,
                'parameters': results[best_key]['parameters'],
                'full_results': results[best_key]['results']
            }
        return {}
    
    def _analyze_parameter_sensitivity(self, results: Dict) -> Dict:
        """Analyze which parameters have the biggest impact"""
        
        # Group results by parameter values
        param_impacts = {}
        
        for key, result_data in results.items():
            params = result_data['parameters']
            win_rate = result_data['results'].get('win_rate', 0)
            
            for param_name, param_value in params.items():
                if param_name not in param_impacts:
                    param_impacts[param_name] = {}
                
                if param_value not in param_impacts[param_name]:
                    param_impacts[param_name][param_value] = []
                
                param_impacts[param_name][param_value].append(win_rate)
        
        # Calculate average impact for each parameter value
        sensitivity_analysis = {}
        for param_name, value_results in param_impacts.items():
            avg_by_value = {}
            for value, win_rates in value_results.items():
                avg_by_value[value] = np.mean(win_rates)
            
            # Calculate sensitivity (range of averages)
            if len(avg_by_value) > 1:
                sensitivity = max(avg_by_value.values()) - min(avg_by_value.values())
                sensitivity_analysis[param_name] = {
                    'sensitivity': sensitivity,
                    'avg_by_value': avg_by_value,
                    'best_value': max(avg_by_value.keys(), key=lambda k: avg_by_value[k])
                }
        
        return sensitivity_analysis
    
    def generate_report(self, output_path: str = None) -> str:
        """Generate comprehensive optimization report"""
        
        analysis = self.analyze_results()
        
        if not analysis.get('successful_count', 0):
            return "No successful optimization results to report."
        
        report = []
        report.append("=" * 80)
        report.append("RISK MANAGEMENT OPTIMIZATION REPORT")
        report.append("=" * 80)
        report.append(f"Total combinations tested: {analysis['total_tested']}")
        report.append(f"Successful combinations: {analysis['successful_count']}")
        report.append(f"Success rate: {analysis['successful_count']/analysis['total_tested']*100:.1f}%")
        report.append("")
        
        # Best performers
        if analysis.get('best_win_rate'):
            best_wr = analysis['best_win_rate']
            report.append("BEST WIN RATE:")
            report.append(f"  Win Rate: {best_wr['value']:.1f}%")
            report.append(f"  Parameters: {best_wr['parameters']}")
            report.append("")
        
        if analysis.get('best_return'):
            best_ret = analysis['best_return']
            report.append("BEST TOTAL RETURN:")
            report.append(f"  Return: {best_ret['value']:+.2f}%")
            report.append(f"  Parameters: {best_ret['parameters']}")
            report.append("")
        
        # Parameter sensitivity
        if analysis.get('parameter_sensitivity'):
            report.append("PARAMETER SENSITIVITY ANALYSIS:")
            sensitivity = analysis['parameter_sensitivity']
            sorted_params = sorted(sensitivity.items(), 
                                 key=lambda x: x[1]['sensitivity'], 
                                 reverse=True)
            
            for param_name, param_data in sorted_params:
                report.append(f"  {param_name}:")
                report.append(f"    Sensitivity: {param_data['sensitivity']:.2f}")
                report.append(f"    Best value: {param_data['best_value']}")
                report.append(f"    Average by value: {param_data['avg_by_value']}")
                report.append("")
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to: {output_path}")
        
        return report_text


def optimize_risk_management(config_path: str, test_type: str = 'quick') -> Dict:
    """
    Convenience function to run complete risk management optimization
    
    Args:
        config_path: Path to base backtesting configuration
        test_type: 'quick', 'comprehensive', or 'focused'
    
    Returns:
        Optimization results and analysis
    """
    optimizer = RiskManagementOptimizer(config_path)
    
    # Generate parameter combinations
    combinations = optimizer.generate_test_parameter_grid(test_type)
    
    # Run optimization
    results = optimizer.run_optimization_batch(combinations)
    
    # Analyze results
    analysis = optimizer.analyze_results()
    
    # Generate report
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = f"storage/optimization/risk_management_report_{timestamp}.txt"
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    
    report = optimizer.generate_report(report_path)
    logger.info(report)
    
    return {
        'optimizer': optimizer,
        'results': results,
        'analysis': analysis,
        'report': report,
        'report_path': report_path
    }


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Risk Management Parameter Optimization')
    parser.add_argument('config_file', help='Path to backtesting configuration file')
    parser.add_argument('--test-type', choices=['quick', 'focused', 'comprehensive'], 
                       default='quick', help='Type of optimization test to run')
    parser.add_argument('--max-concurrent', type=int, default=1, 
                       help='Maximum concurrent backtests (be careful with system resources)')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Generate parameter combinations without running backtests')
    
    args = parser.parse_args()
    
    logger.info(f"🚀 Starting Risk Management Optimization")
    logger.info(f"   Config: {args.config_file}")
    logger.info(f"   Test Type: {args.test_type}")
    logger.info(f"   Max Concurrent: {args.max_concurrent}")
    logger.info("=" * 60)
    
    if args.dry_run:
        logger.info("🔍 DRY RUN MODE - Generating parameters only")
        optimizer = RiskManagementOptimizer(args.config_file)
        combinations = optimizer.generate_test_parameter_grid(args.test_type)
        
        logger.info(f"\n📊 Generated {len(combinations)} parameter combinations:")
        for i, params in enumerate(combinations[:10]):  # Show first 10
            logger.info(f"  {i+1:3d}. {params}")
        
        if len(combinations) > 10:
            logger.info(f"  ... and {len(combinations) - 10} more combinations")
        
        logger.info(f"\n💡 Remove --dry-run to execute actual optimization")
    else:
        logger.info("🔥 LIVE MODE - Running actual backtests")
        results = optimize_risk_management(args.config_file, args.test_type)
        logger.info(f"\n✅ Optimization completed!")
        logger.info(f"📊 Report saved to: {results['report_path']}")
        
        # Show quick summary
        analysis = results['analysis']
        if analysis.get('successful_count', 0) > 0:
            logger.info(f"\n🎯 Quick Results Summary:")
            logger.info(f"   Successful tests: {analysis['successful_count']}/{analysis['total_tested']}")
            
            if analysis.get('best_win_rate'):
                best_wr = analysis['best_win_rate']
                logger.info(f"   Best win rate: {best_wr['value']:.1f}%")
                logger.info(f"   Best parameters: {best_wr['parameters']}")
                
            if analysis.get('parameter_sensitivity'):
                sens = analysis['parameter_sensitivity']
                most_sensitive = max(sens.items(), key=lambda x: x[1]['sensitivity'])
                logger.info(f"   Most sensitive parameter: {most_sensitive[0]} (impact: {most_sensitive[1]['sensitivity']:.2f})")
        else:
            logger.info("⚠️  No successful results found. Check your configuration and system setup.")