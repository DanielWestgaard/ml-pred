"""
Enhanced Risk Management Optimizer for Exit Strategy & Model Performance Tuning
=============================================================================

Building on the existing risk_management_optimizer.py to focus on:
1. Confidence threshold optimization (0.50-0.65 range) 
2. Exit strategy parameters
3. Position duration settings
4. Stop-loss/take-profit ratios

This runs quickly with targeted parameter ranges based on our breakthrough results.
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

from risk_management_optimizer import RiskManagementOptimizer
from utils.logging_utils import get_logger
logger = get_logger()

class EnhancedRiskOptimizer(RiskManagementOptimizer):
    """
    Enhanced optimizer focused on our specific breakthrough parameters
    """
    
    def generate_breakthrough_grid(self, test_type: str = 'focused') -> List[Dict]:
        """
        Generate parameter grid focused on our breakthrough at 0.54 confidence
        
        Args:
            test_type: 'quick' (30 combos), 'focused' (100 combos), 'comprehensive' (300 combos)
        """
        if test_type == 'quick':
            return self._generate_breakthrough_quick()
        elif test_type == 'focused':
            return self._generate_breakthrough_focused()
        elif test_type == 'comprehensive':
            return self._generate_breakthrough_comprehensive()
        else:
            raise ValueError(f"Unknown test_type: {test_type}")
    
    def _generate_breakthrough_quick(self) -> List[Dict]:
        """Quick 30-combo test around breakthrough parameters"""
        
        grid_params = {
            # Confidence threshold around breakthrough point
            'confidence_threshold': [0.52, 0.54, 0.56, 0.58, 0.60],
            
            # Stop-loss optimization (current 5.0% seems wide)
            'stop_loss_pct': [0.03, 0.05, 0.08],
            
            # Take-profit optimization (current 1.5% seems tight)  
            'take_profit_pct': [0.015, 0.025, 0.035],
            
            # Minimum position duration (current 10min)
            'minimum_signal_life': [8, 10, 12]
        }
        
        # 5*3*3*3 = 135 combinations - sample to 30
        all_combinations = list(itertools.product(*grid_params.values()))
        param_keys = list(grid_params.keys())
        
        # Select evenly spaced combinations
        step = len(all_combinations) // 30
        selected_combinations = all_combinations[::step][:30]
        
        combinations = []
        for combo in selected_combinations:
            param_dict = dict(zip(param_keys, combo))
            combinations.append(param_dict)
        
        logger.info(f"Generated {len(combinations)} breakthrough quick test combinations")
        return combinations
    
    def _generate_breakthrough_focused(self) -> List[Dict]:
        """Focused 100-combo test with detailed parameter sweep"""
        
        grid_params = {
            # Fine-grained confidence threshold sweep
            'confidence_threshold': [0.50, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.60],
            
            # Stop-loss range (wider to tighter)
            'stop_loss_pct': [0.02, 0.03, 0.04, 0.05, 0.06, 0.08],
            
            # Take-profit range  
            'take_profit_pct': [0.01, 0.015, 0.02, 0.025, 0.03, 0.04],
            
            # Position duration
            'minimum_signal_life': [5, 8, 10, 12, 15],
            
            # Market change exit threshold (current 98% HOLD + 10% signal)
            'market_change_hold_threshold': [0.95, 0.97, 0.98, 0.99],
            'market_change_signal_threshold': [0.05, 0.10, 0.15]
        }
        
        # Smart sampling to get ~100 combinations
        combinations = self._smart_sample_grid(grid_params, max_combinations=100)
        
        logger.info(f"Generated {len(combinations)} breakthrough focused test combinations")
        return combinations
    
    def _generate_breakthrough_comprehensive(self) -> List[Dict]:
        """Comprehensive 300-combo test with all parameter variations"""
        
        grid_params = {
            # Very fine-grained confidence threshold
            'confidence_threshold': [0.49, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60, 0.61],
            
            # Full stop-loss range
            'stop_loss_pct': [0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.06, 0.08, 0.10],
            
            # Full take-profit range
            'take_profit_pct': [0.008, 0.01, 0.012, 0.015, 0.018, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05],
            
            # Position duration variations
            'minimum_signal_life': [3, 5, 8, 10, 12, 15, 20],
            
            # Exit strategy parameters
            'market_change_hold_threshold': [0.93, 0.95, 0.97, 0.98, 0.99],
            'market_change_signal_threshold': [0.05, 0.08, 0.10, 0.12, 0.15, 0.20],
            
            # Signal strength decay threshold
            'signal_strength_decay': [0.5, 0.6, 0.7, 0.8],
            
            # Position size variations
            'max_position_size': [0.06, 0.08, 0.10]
        }
        
        # Smart sampling to get ~300 combinations
        combinations = self._smart_sample_grid(grid_params, max_combinations=300)
        
        logger.info(f"Generated {len(combinations)} breakthrough comprehensive test combinations")
        return combinations
    
    def create_test_config(self, base_config: Dict, param_overrides: Dict) -> Dict:
        """
        Enhanced config creation with our new parameters
        """
        import copy
        test_config = copy.deepcopy(base_config)
        
        # Enhanced parameter mapping for our new exit strategy parameters
        param_mapping = {
            'confidence_threshold': ['backtesting', 'signal_processing', 'confidence_threshold'],
            'stop_loss_pct': ['backtesting', 'risk_management', 'stop_loss_pct'],
            'take_profit_pct': ['backtesting', 'risk_management', 'take_profit_pct'],
            'minimum_signal_life': ['backtesting', 'exit_strategy', 'minimum_signal_life'],
            'signal_strength_decay': ['backtesting', 'exit_strategy', 'signal_strength_decay'],
            'market_change_hold_threshold': ['backtesting', 'exit_strategy', 'market_change_hold_threshold'],
            'market_change_signal_threshold': ['backtesting', 'exit_strategy', 'market_change_signal_threshold'],
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
                logger.debug(f"Set {'.'.join(config_path)} = {value}")
            else:
                logger.warning(f"Unknown parameter for mapping: {param}")
        
        return test_config
    
    def _parse_output_results(self, stdout: str, stderr: str) -> Optional[Dict]:
        """
        Enhanced result parsing for our specific log format
        """
        try:
            import re
            
            # Patterns matching your specific log format
            patterns = {
                'total_return_pct': r'Return:\s*([+-]?\d+\.?\d*)%',
                'win_rate': r'Win Rate:\s*(\d+\.?\d*)%', 
                'total_trades': r'Trades:\s*(\d+)',
                'final_equity': r'Equity:\s*\$([+-]?\d+\.?\d*)',
                'iterations': r'Iterations:\s*(\d+)/(\d+)',
                'execution_time': r'completed in\s*(\d+\.?\d*)\s*seconds'
            }
            
            results = {}
            full_output = stdout + "\n" + stderr
            
            for metric, pattern in patterns.items():
                match = re.search(pattern, full_output, re.IGNORECASE)
                if match:
                    try:
                        if metric == 'iterations':
                            results['iterations_completed'] = int(match.group(1))
                            results['iterations_total'] = int(match.group(2))
                        else:
                            results[metric] = float(match.group(1))
                    except (ValueError, IndexError):
                        results[metric] = match.group(1)
            
            # Parse final results line format: "Return: -0.48%, Trades: 87, Win Rate: 0.1%"
            final_results_pattern = r'Return:\s*([+-]?\d+\.?\d*)%.*?Trades:\s*(\d+).*?Win Rate:\s*(\d+\.?\d*)%'
            final_match = re.search(final_results_pattern, full_output, re.IGNORECASE)
            if final_match:
                results['total_return_pct'] = float(final_match.group(1))
                results['total_trades'] = int(final_match.group(2))
                results['win_rate'] = float(final_match.group(3))
            
            # Return results if we found key metrics
            if len(results) >= 2:
                logger.debug(f"Parsed results from output: {results}")
                return results
                
            logger.warning(f"Could not parse sufficient results. Found: {results}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to parse output results: {e}")
            return None


def test_enhanced_optimizer(config_path: str) -> bool:
    """
    Test the enhanced optimizer to make sure it works before running hours of tests
    
    Args:
        config_path: Path to your backtesting config file
        
    Returns:
        True if tests pass, False if issues found
    """
    logger.info("🧪 Testing Enhanced Risk Optimizer")
    logger.info("=" * 60)
    
    try:
        # Test 1: Initialize optimizer
        optimizer = EnhancedRiskOptimizer(config_path)
        logger.info(f"✅ Test 1 PASS: Successfully loaded config: {config_path}")
        
        # Test 2: Generate parameter combinations
        quick_params = optimizer.generate_breakthrough_grid('quick')
        logger.info(f"✅ Test 2 PASS: Generated {len(quick_params)} parameter combinations")
        
        # Test 3: Create test configuration
        test_params = quick_params[0]
        test_config = optimizer.create_test_config(optimizer.base_config, test_params)
        logger.info(f"✅ Test 3 PASS: Created test config with params: {test_params}")
        
        # Test 4: Verify config modifications
        original_conf = optimizer.base_config['backtesting']['signal_processing']['confidence_threshold']
        modified_conf = test_config['backtesting']['signal_processing']['confidence_threshold']
        
        if original_conf != modified_conf:
            logger.info(f"✅ Test 4 PASS: Config modification working (confidence: {original_conf} → {modified_conf})")
        else:
            logger.info(f"⚠️  Test 4 WARNING: Config not modified (both: {original_conf})")
        
        # Test 5: Test result parsing
        sample_output = """
        Final Results: Iterations: 3000/3000 | Return: -0.48%, Trades: 87, Win Rate: 0.1%
        Backtest completed in 96.2 seconds
        """
        
        parsed = optimizer._parse_output_results(sample_output, "")
        if parsed and 'win_rate' in parsed:
            logger.info(f"✅ Test 5 PASS: Result parsing working: {parsed}")
        else:
            logger.info(f"❌ Test 5 FAIL: Result parsing failed: {parsed}")
            return False
        
        logger.info("\n🎯 All Tests PASSED!")
        logger.info("✅ Enhanced optimizer is ready for production use")
        return True
        
    except Exception as e:
        logger.info(f"❌ Tests FAILED: {e}")
        import traceback
        traceback.logger.info_exc()
        return False


def run_quick_optimization(config_path: str) -> Dict:
    """
    Run a quick 30-combination optimization to verify everything works
    """
    logger.info("🚀 Running Quick Optimization (30 combinations)")
    logger.info("=" * 60)
    
    optimizer = EnhancedRiskOptimizer(config_path)
    
    # Generate quick test parameters
    combinations = optimizer.generate_breakthrough_grid('quick')
    logger.info(f"Generated {len(combinations)} parameter combinations")
    
    # Show first few combinations
    logger.info("\nFirst 5 parameter combinations:")
    for i, params in enumerate(combinations[:5]):
        logger.info(f"  {i+1}. {params}")
    
    # Run optimization
    start_time = time.time()
    results = optimizer.run_optimization_batch(combinations)
    end_time = time.time()
    
    logger.info(f"\n⏱️  Optimization completed in {end_time - start_time:.1f} seconds")
    
    # Analyze results
    analysis = optimizer.analyze_results()
    
    if analysis.get('successful_count', 0) > 0:
        logger.info(f"✅ Success! {analysis['successful_count']}/{analysis['total_tested']} combinations worked")
        
        if analysis.get('best_win_rate'):
            best = analysis['best_win_rate']
            logger.info(f"🏆 Best win rate: {best['value']:.1f}%")
            logger.info(f"🏆 Best parameters: {best['parameters']}")
        
        # Generate report
        report = optimizer.generate_report()
        
        # Save results
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"storage/optimization/quick_test_{timestamp}.json"
        Path(results_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump({
                'analysis': analysis,
                'results': results,
                'report': report
            }, f, indent=2, default=str)
        
        logger.info(f"💾 Results saved to: {results_file}")
        
    else:
        logger.info("❌ No successful combinations found - check your configuration")
        
    return {
        'optimizer': optimizer,
        'results': results,
        'analysis': analysis
    }


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Risk Management Optimizer')
    parser.add_argument('config_file', help='Path to backtesting configuration file')
    parser.add_argument('--test-only', action='store_true', help='Run tests only, no optimization')
    parser.add_argument('--quick', action='store_true', help='Run quick 30-combo optimization')
    parser.add_argument('--test-type', choices=['quick', 'focused', 'comprehensive'], 
                       default='focused', help='Type of optimization')
    
    args = parser.parse_args()
    
    if args.test_only:
        logger.info("🧪 TESTING MODE")
        success = test_enhanced_optimizer(args.config_file)
        if success:
            logger.info("\n🎉 All tests passed! Ready for optimization.")
        else:
            logger.info("\n❌ Tests failed! Fix issues before running optimization.")
            sys.exit(1)
    
    elif args.quick:
        logger.info("⚡ QUICK OPTIMIZATION MODE")
        results = run_quick_optimization(args.config_file)
        logger.info("\n✅ Quick optimization completed!")
    
    else:
        logger.info("🔥 FULL OPTIMIZATION MODE")
        logger.info(f"Running {args.test_type} optimization...")
        
        optimizer = EnhancedRiskOptimizer(args.config_file)
        combinations = optimizer.generate_breakthrough_grid(args.test_type)
        
        logger.info(f"Will test {len(combinations)} parameter combinations")
        logger.info("This may take several hours...")
        
        confirm = input("Continue? (y/N): ")
        if confirm.lower() == 'y':
            results = optimizer.run_optimization_batch(combinations)
            analysis = optimizer.analyze_results()
            
            # Generate and save report
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"storage/optimization/enhancement_report_{timestamp}.txt"
            Path(report_path).parent.mkdir(parents=True, exist_ok=True)
            
            report = optimizer.generate_report(report_path)
            logger.info(f"\n📊 Optimization complete! Report: {report_path}")
        else:
            logger.info("Cancelled.")