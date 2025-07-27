#!/usr/bin/env python3
"""
Quick Parameter Tuning
======================

Fast parameter optimization for the intelligent exit strategy.
"""

import os
import yaml
import subprocess
import pandas as pd
from datetime import datetime
from pathlib import Path
import time

def create_tuning_configs():
    """Create configurations for parameter tuning."""
    
    base_config_path = '/Users/danielwestgaard/progRel/ml-pred/hypothesis/spread_compression_patterns/config_backtest_improved_simple.yml'
    
    # Load base config
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Parameter variations to test
    test_configs = [
        {
            'name': 'baseline',
            'confidence_threshold': 0.30,
            'signal_decay': 0.25,
            'min_signal_life': 3,
            'description': 'Current baseline configuration'
        },
        {
            'name': 'conservative',
            'confidence_threshold': 0.35,
            'signal_decay': 0.30,
            'min_signal_life': 5,
            'description': 'More conservative exits'
        },
        {
            'name': 'aggressive',
            'confidence_threshold': 0.25,
            'signal_decay': 0.20,
            'min_signal_life': 2,
            'description': 'More aggressive exits'
        },
        {
            'name': 'balanced',
            'confidence_threshold': 0.28,
            'signal_decay': 0.23,
            'min_signal_life': 4,
            'description': 'Balanced approach'
        }
    ]
    
    # Create configs directory
    configs_dir = Path('/Users/danielwestgaard/progRel/ml-pred/hypothesis/spread_compression_patterns/tuning_configs')
    configs_dir.mkdir(exist_ok=True)
    
    created_configs = []
    
    for config in test_configs:
        # Copy base config
        tuning_config = base_config.copy()
        
        # Update exit strategy parameters
        if 'exit_strategy' not in tuning_config['backtesting']:
            tuning_config['backtesting']['exit_strategy'] = {}
        
        tuning_config['backtesting']['exit_strategy']['confidence_exit_threshold'] = config['confidence_threshold']
        tuning_config['backtesting']['exit_strategy']['signal_strength_decay'] = config['signal_decay']
        tuning_config['backtesting']['exit_strategy']['minimum_signal_life'] = config['min_signal_life']
        
        # Use the high-scoring period from previous analysis
        tuning_config['backtesting']['start_date'] = '2025-04-07'
        tuning_config['backtesting']['end_date'] = '2025-04-10'  # Shorter 3-day test
        
        # Reduce max_loops for faster testing
        tuning_config['backtesting']['execution']['max_loops'] = 3000
        
        # Save config
        config_path = configs_dir / f"config_tuning_{config['name']}.yml"
        with open(config_path, 'w') as f:
            yaml.dump(tuning_config, f, default_flow_style=False, indent=2)
        
        created_configs.append({
            'path': str(config_path),
            'name': config['name'],
            'description': config['description'],
            **config
        })
        
        print(f"✅ Created config: {config['name']} - {config['description']}")
    
    return created_configs

def run_quick_test(config_info):
    """Run a single quick test."""
    
    print(f"\n🔄 Testing: {config_info['name']} - {config_info['description']}")
    print(f"   Confidence: {config_info['confidence_threshold']}")
    print(f"   Signal decay: {config_info['signal_decay']}")
    print(f"   Min signal life: {config_info['min_signal_life']}")
    
    try:
        start_time = time.time()
        
        # Run backtest with shorter timeout
        result = subprocess.run(
            ['python', 'main.py', '-hyp', config_info['path']],
            capture_output=True,
            text=True,
            timeout=180,  # 3 minute timeout
            cwd='/Users/danielwestgaard/progRel/ml-pred'
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Test completed in {duration:.1f}s")
            
            # Extract key metrics from output
            metrics = extract_quick_metrics(result.stdout)
            
            return {
                'config': config_info['name'],
                'success': True,
                'duration': duration,
                **metrics
            }
        else:
            print(f"❌ Test failed: {result.stderr[:200]}...")
            return {
                'config': config_info['name'],
                'success': False,
                'duration': duration,
                'error': result.stderr[:200]
            }
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Test timed out after 3 minutes")
        return {
            'config': config_info['name'],
            'success': False,
            'duration': 180,
            'error': 'Timeout'
        }
    except Exception as e:
        print(f"💥 Test error: {e}")
        return {
            'config': config_info['name'],
            'success': False,
            'duration': 0,
            'error': str(e)
        }

def extract_quick_metrics(stdout):
    """Extract key metrics from backtest output."""
    
    metrics = {
        'final_value': 10000,
        'total_return': 0.0,
        'total_trades': 0,
        'confidence_exits': 0,
        'signal_decay_exits': 0,
        'volatility_exits': 0,
        'stop_loss_exits': 0,
        'take_profit_exits': 0
    }
    
    try:
        lines = stdout.split('\n')
        
        for line in lines:
            # Extract final portfolio value
            if 'Final Portfolio Value:' in line or 'Portfolio value:' in line:
                try:
                    value_str = line.split('$')[1].replace(',', '').replace(')', '').strip()
                    metrics['final_value'] = float(value_str)
                    metrics['total_return'] = (metrics['final_value'] - 10000) / 10000 * 100
                except:
                    pass
            
            # Count trades
            elif 'Position entered' in line or 'Order executed: BUY' in line:
                metrics['total_trades'] += 1
            
            # Count exit types
            elif 'confidence_drop' in line:
                metrics['confidence_exits'] += 1
            elif 'signal_decay' in line:
                metrics['signal_decay_exits'] += 1
            elif 'volatility_spike' in line:
                metrics['volatility_exits'] += 1
            elif 'Stop loss triggered' in line:
                metrics['stop_loss_exits'] += 1
            elif 'Take profit triggered' in line:
                metrics['take_profit_exits'] += 1
        
        # Calculate exit percentages
        if metrics['total_trades'] > 0:
            total_exits = (metrics['confidence_exits'] + metrics['signal_decay_exits'] + 
                          metrics['volatility_exits'] + metrics['stop_loss_exits'] + metrics['take_profit_exits'])
            
            if total_exits > 0:
                metrics['confidence_exit_pct'] = metrics['confidence_exits'] / total_exits * 100
                metrics['signal_decay_exit_pct'] = metrics['signal_decay_exits'] / total_exits * 100
    
    except Exception as e:
        print(f"⚠️  Error extracting metrics: {e}")
    
    return metrics

def run_parameter_tuning():
    """Run quick parameter tuning tests."""
    
    print("🚀 Quick Parameter Tuning")
    print("=" * 40)
    
    # Create test configurations
    configs = create_tuning_configs()
    
    # Run tests
    results = []
    
    for config_info in configs:
        result = run_quick_test(config_info)
        results.append(result)
        
        if result['success']:
            print(f"   📊 Return: {result.get('total_return', 0):.2f}%")
            print(f"   📈 Trades: {result.get('total_trades', 0)}")
            print(f"   🚪 Confidence exits: {result.get('confidence_exits', 0)}")
        
        time.sleep(1)  # Brief pause between tests
    
    # Analyze results
    print(f"\n📊 PARAMETER TUNING RESULTS")
    print("=" * 40)
    
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print("❌ No successful tests completed")
        return
    
    # Sort by return
    successful_results.sort(key=lambda x: x.get('total_return', 0), reverse=True)
    
    print("\n🏆 RANKING BY PERFORMANCE:")
    for i, result in enumerate(successful_results, 1):
        print(f"{i}. {result['config'].upper()}")
        print(f"   Return: {result.get('total_return', 0):.2f}%")
        print(f"   Trades: {result.get('total_trades', 0)}")
        print(f"   Confidence exits: {result.get('confidence_exits', 0)} ({result.get('confidence_exit_pct', 0):.1f}%)")
        print(f"   Signal decay exits: {result.get('signal_decay_exits', 0)} ({result.get('signal_decay_exit_pct', 0):.1f}%)")
        print()
    
    # Save results
    results_df = pd.DataFrame(results)
    results_path = Path('/Users/danielwestgaard/progRel/ml-pred/results/quick_parameter_tuning_results.csv')
    results_df.to_csv(results_path, index=False)
    
    print(f"💾 Results saved to: {results_path}")
    
    # Recommend best configuration
    if successful_results:
        best_config = successful_results[0]
        print(f"\n🎯 RECOMMENDED CONFIGURATION: {best_config['config'].upper()}")
        print(f"   Best return: {best_config.get('total_return', 0):.2f}%")
        print(f"   Total trades: {best_config.get('total_trades', 0)}")
        
        return best_config

if __name__ == "__main__":
    run_parameter_tuning()