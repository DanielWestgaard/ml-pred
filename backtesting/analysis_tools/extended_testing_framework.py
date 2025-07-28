#!/usr/bin/env python3
"""
Extended Testing Framework
==========================

Run comprehensive testing across multiple periods and parameter combinations
to optimize the intelligent exit strategy.
"""

import os
import yaml
import subprocess
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import itertools
import time

def get_optimal_periods():
    """Get the optimal periods identified by intelligent period selection."""
    return [
        {'start': '2025-04-07', 'end': '2025-04-14', 'score': 40.4, 'signals': 975},
        {'start': '2025-04-14', 'end': '2025-04-21', 'score': 35.2, 'signals': 842},
        {'start': '2025-03-31', 'end': '2025-04-07', 'score': 32.8, 'signals': 756},
        {'start': '2025-04-21', 'end': '2025-04-28', 'score': 28.9, 'signals': 689},
        {'start': '2025-03-24', 'end': '2025-03-31', 'score': 26.1, 'signals': 634},
    ]

def create_test_config(base_config_path, period, parameters, output_path):
    """Create a test configuration with specific period and parameters."""
    
    # Load base config
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update period
    config['backtesting']['start_date'] = period['start']
    config['backtesting']['end_date'] = period['end']
    
    # Update exit strategy parameters
    if 'exit_strategy' not in config['backtesting']:
        config['backtesting']['exit_strategy'] = {}
    
    config['backtesting']['exit_strategy']['confidence_exit_threshold'] = parameters['confidence_threshold']
    config['backtesting']['exit_strategy']['signal_strength_decay'] = parameters['signal_decay']
    config['backtesting']['exit_strategy']['minimum_signal_life'] = parameters['min_signal_life']
    
    # Ensure signal tracking is configured
    if 'signal_tracking' not in config['backtesting']:
        config['backtesting']['signal_tracking'] = {
            'signal_strength_features': [
                'compression_ratio_micro',
                'volume_ratio',
                'signal_strength_v2',
                'compression_consistency'
            ]
        }
    
    # Save test config
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    return output_path

def run_backtest(config_path):
    """Run a single backtest and return results."""
    try:
        print(f"🔄 Running backtest: {config_path}")
        start_time = time.time()
        
        # Run backtest with timeout
        result = subprocess.run(
            ['python', 'main.py', '-hyp', config_path],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd='/Users/danielwestgaard/progRel/ml-pred'
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Backtest completed in {duration:.1f}s")
            return {
                'success': True,
                'duration': duration,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        else:
            print(f"❌ Backtest failed: {result.stderr}")
            return {
                'success': False,
                'duration': duration,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
    except subprocess.TimeoutExpired:
        print(f"⏰ Backtest timed out after 5 minutes")
        return {
            'success': False,
            'duration': 300,
            'stdout': '',
            'stderr': 'Timeout after 5 minutes'
        }
    except Exception as e:
        print(f"💥 Backtest error: {e}")
        return {
            'success': False,
            'duration': 0,
            'stdout': '',
            'stderr': str(e)
        }

def extract_backtest_metrics(stdout):
    """Extract key metrics from backtest output."""
    metrics = {
        'total_return': 0.0,
        'total_trades': 0,
        'win_rate': 0.0,
        'avg_holding_time': 0.0,
        'confidence_exits': 0,
        'signal_decay_exits': 0,
        'volatility_exits': 0,
        'compression_reversal_exits': 0,
        'max_drawdown': 0.0
    }
    
    try:
        lines = stdout.split('\n')
        for line in lines:
            if 'Final Portfolio Value:' in line:
                # Extract final value and calculate return
                final_value = float(line.split('$')[1].replace(',', ''))
                metrics['total_return'] = (final_value - 10000) / 10000 * 100
            elif 'Total Trades:' in line:
                metrics['total_trades'] = int(line.split(':')[1].strip())
            elif 'confidence_drop' in line:
                metrics['confidence_exits'] += 1
            elif 'signal_decay' in line:
                metrics['signal_decay_exits'] += 1
            elif 'volatility_spike' in line:
                metrics['volatility_exits'] += 1
            elif 'compression_reversal' in line:
                metrics['compression_reversal_exits'] += 1
    except Exception as e:
        print(f"⚠️  Error extracting metrics: {e}")
    
    return metrics

def run_extended_testing():
    """Run comprehensive testing across periods and parameters."""
    
    print("🚀 Starting Extended Testing Framework")
    print("=" * 50)
    
    # Test parameters
    parameter_combinations = [
        # Baseline (current)
        {'confidence_threshold': 0.30, 'signal_decay': 0.25, 'min_signal_life': 3},
        
        # Confidence threshold variations
        {'confidence_threshold': 0.25, 'signal_decay': 0.25, 'min_signal_life': 3},
        {'confidence_threshold': 0.35, 'signal_decay': 0.25, 'min_signal_life': 3},
        
        # Signal decay variations
        {'confidence_threshold': 0.30, 'signal_decay': 0.20, 'min_signal_life': 3},
        {'confidence_threshold': 0.30, 'signal_decay': 0.30, 'min_signal_life': 3},
        
        # Minimum signal life variations
        {'confidence_threshold': 0.30, 'signal_decay': 0.25, 'min_signal_life': 2},
        {'confidence_threshold': 0.30, 'signal_decay': 0.25, 'min_signal_life': 5},
        
        # Optimized combinations
        {'confidence_threshold': 0.25, 'signal_decay': 0.20, 'min_signal_life': 2},
        {'confidence_threshold': 0.35, 'signal_decay': 0.30, 'min_signal_life': 5},
    ]
    
    # Get optimal periods
    periods = get_optimal_periods()[:3]  # Test top 3 periods initially
    
    # Results storage
    results = []
    
    # Base config path
    base_config = '/Users/danielwestgaard/progRel/ml-pred/hypothesis/spread_compression_patterns/config_backtest_improved_simple.yml'
    
    # Create results directory
    results_dir = Path('/Users/danielwestgaard/progRel/ml-pred/results/extended_testing')
    results_dir.mkdir(exist_ok=True)
    
    # Test each combination
    test_count = 0
    total_tests = len(periods) * len(parameter_combinations)
    
    for period_idx, period in enumerate(periods):
        for param_idx, params in enumerate(parameter_combinations):
            test_count += 1
            print(f"\n📊 Test {test_count}/{total_tests}")
            print(f"Period: {period['start']} to {period['end']} (Score: {period['score']})")
            print(f"Parameters: {params}")
            
            # Create test config
            config_name = f"test_p{period_idx}_param{param_idx}.yml"
            config_path = results_dir / config_name
            
            create_test_config(base_config, period, params, config_path)
            
            # Run backtest
            result = run_backtest(str(config_path))
            
            # Extract metrics
            metrics = extract_backtest_metrics(result.get('stdout', ''))
            
            # Store results
            test_result = {
                'test_id': f"p{period_idx}_param{param_idx}",
                'period_start': period['start'],
                'period_end': period['end'],
                'period_score': period['score'],
                'confidence_threshold': params['confidence_threshold'],
                'signal_decay': params['signal_decay'],
                'min_signal_life': params['min_signal_life'],
                'success': result['success'],
                'duration': result['duration'],
                **metrics
            }
            
            results.append(test_result)
            
            # Save intermediate results
            results_df = pd.DataFrame(results)
            results_df.to_csv(results_dir / 'extended_testing_results.csv', index=False)
            
            print(f"Return: {metrics['total_return']:.2f}%, Trades: {metrics['total_trades']}")
            
            # Small delay between tests
            time.sleep(2)
    
    print(f"\n✅ Extended testing completed!")
    print(f"📊 Results saved to: {results_dir / 'extended_testing_results.csv'}")
    
    return results

def analyze_results():
    """Analyze extended testing results and find optimal parameters."""
    
    results_file = Path('/Users/danielwestgaard/progRel/ml-pred/results/extended_testing/extended_testing_results.csv')
    
    if not results_file.exists():
        print("❌ No results file found. Run extended testing first.")
        return
    
    df = pd.read_csv(results_file)
    
    print("\n📈 EXTENDED TESTING ANALYSIS")
    print("=" * 50)
    
    # Filter successful tests
    successful_tests = df[df['success'] == True]
    
    if successful_tests.empty:
        print("❌ No successful tests found.")
        return
    
    print(f"✅ Successful tests: {len(successful_tests)}/{len(df)}")
    
    # Best performing configurations
    print("\n🏆 TOP 5 CONFIGURATIONS BY RETURN:")
    top_configs = successful_tests.nlargest(5, 'total_return')
    for idx, row in top_configs.iterrows():
        print(f"  {row['test_id']}: {row['total_return']:.2f}% return")
        print(f"    Confidence: {row['confidence_threshold']}, Decay: {row['signal_decay']}, Min Life: {row['min_signal_life']}")
        print(f"    Trades: {row['total_trades']}, Period: {row['period_start']} to {row['period_end']}")
        print()
    
    # Parameter analysis
    print("\n📊 PARAMETER ANALYSIS:")
    
    # Group by parameters and analyze
    param_analysis = successful_tests.groupby(['confidence_threshold', 'signal_decay', 'min_signal_life']).agg({
        'total_return': ['mean', 'std', 'count'],
        'total_trades': 'mean',
        'confidence_exits': 'mean',
        'signal_decay_exits': 'mean'
    }).round(3)
    
    print(param_analysis)
    
    # Find optimal parameters
    best_params = successful_tests.loc[successful_tests['total_return'].idxmax()]
    
    print(f"\n🎯 OPTIMAL PARAMETERS:")
    print(f"  Confidence threshold: {best_params['confidence_threshold']}")
    print(f"  Signal decay: {best_params['signal_decay']}")
    print(f"  Minimum signal life: {best_params['min_signal_life']}")
    print(f"  Best return: {best_params['total_return']:.2f}%")
    print(f"  Total trades: {best_params['total_trades']}")
    
    return best_params

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'analyze':
        analyze_results()
    else:
        run_extended_testing()