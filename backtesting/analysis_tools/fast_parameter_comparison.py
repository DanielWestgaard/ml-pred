#!/usr/bin/env python3
"""
Fast Parameter Comparison
=========================

Quick parameter testing with optimized execution times.
"""

import os
import yaml
import subprocess
import pandas as pd
from datetime import datetime
from pathlib import Path
import time

def create_fast_configs():
    """Create fast test configurations for parameter comparison."""
    
    # Base template with fast settings
    base_template = {
        'hypothesis': {'name': 'spread_compression_patterns'},
        'backtesting': {
            'enabled': True,
            'model_folder': 'storage/models/2025-07-27_12:53:02',
            'data_source': 'storage/data/capital_com/GBPUSD/processed/1m/2024-06-01_2025-05-23.csv',
            'start_date': '2025-04-07',
            'end_date': '2025-04-08',                      # Single day for speed
            'buffer_size': 50,
            'initial_capital': 10000,
            'transaction_costs': 0.0013,
            'slippage': 0.0002,
            'execution': {
                'max_loops': 720,                          # ~12 hours at 1min intervals
                'loop_interval': 0
            },
            'signal_processing': {
                'confidence_threshold': 0.75,
                'signal_filters': ['session_filter'],
                'min_signal_strength': 0.3,
                'volume_confirmation_required': True
            },
            'risk_management': {
                'max_position_size': 0.015,
                'stop_loss_pct': 0.015,
                'take_profit_pct': 0.045,
                'max_drawdown': 0.04,
                'max_concurrent_positions': 2,
                'position_size_scaling': 'volatility'
            },
            'position_management': {
                'max_concurrent_positions': 2,
                'position_tracking': 'detailed'
            },
            'signal_tracking': {
                'signal_strength_features': [
                    'compression_ratio_micro',
                    'volume_ratio',
                    'signal_strength_v2',
                    'compression_consistency'
                ]
            }
        }
    }
    
    # Parameter variations
    test_configs = [
        {
            'name': 'baseline',
            'params': {
                'confidence_exit_threshold': 0.30,
                'signal_strength_decay': 0.25,
                'minimum_signal_life': 3
            },
            'description': 'Current baseline'
        },
        {
            'name': 'conservative',
            'params': {
                'confidence_exit_threshold': 0.35,
                'signal_strength_decay': 0.30,
                'minimum_signal_life': 5
            },
            'description': 'Conservative exits'
        },
        {
            'name': 'aggressive',
            'params': {
                'confidence_exit_threshold': 0.25,
                'signal_strength_decay': 0.20,
                'minimum_signal_life': 2
            },
            'description': 'Aggressive exits'
        },
        {
            'name': 'balanced',
            'params': {
                'confidence_exit_threshold': 0.28,
                'signal_strength_decay': 0.23,
                'minimum_signal_life': 4
            },
            'description': 'Balanced approach'
        }
    ]
    
    # Create configs directory
    configs_dir = Path('/Users/danielwestgaard/progRel/ml-pred/hypothesis/spread_compression_patterns/fast_configs')
    configs_dir.mkdir(exist_ok=True)
    
    created_configs = []
    
    for config in test_configs:
        # Create config from template
        test_config = base_template.copy()
        
        # Add exit strategy parameters
        test_config['backtesting']['exit_strategy'] = {
            'volatility_exit_multiplier': 1.5,
            **config['params']
        }
        
        # Save config
        config_path = configs_dir / f"config_fast_{config['name']}.yml"
        with open(config_path, 'w') as f:
            yaml.dump(test_config, f, default_flow_style=False, indent=2)
        
        created_configs.append({
            'path': str(config_path),
            'name': config['name'],
            'description': config['description'],
            **config['params']
        })
        
        print(f"✅ Created: {config['name']} - {config['description']}")
    
    return created_configs

def run_fast_comparison():
    """Run fast parameter comparison."""
    
    print("🚀 Fast Parameter Comparison")
    print("=" * 40)
    
    # Create configs
    configs = create_fast_configs()
    
    results = []
    
    for i, config in enumerate(configs, 1):
        print(f"\n🔄 Test {i}/{len(configs)}: {config['name'].upper()}")
        print(f"   Confidence threshold: {config['confidence_exit_threshold']}")
        print(f"   Signal decay: {config['signal_strength_decay']}")
        print(f"   Min signal life: {config['minimum_signal_life']}min")
        
        try:
            start_time = time.time()
            
            result = subprocess.run(
                ['python', 'main.py', '-hyp', config['path']],
                capture_output=True,
                text=True,
                timeout=60,  # 1 minute timeout
                cwd='/Users/danielwestgaard/progRel/ml-pred'
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                print(f"   ✅ Completed in {duration:.1f}s")
                
                # Extract metrics
                metrics = extract_metrics(result.stdout)
                
                test_result = {
                    'config': config['name'],
                    'success': True,
                    'duration': duration,
                    'confidence_threshold': config['confidence_exit_threshold'],
                    'signal_decay': config['signal_strength_decay'],
                    'min_signal_life': config['minimum_signal_life'],
                    **metrics
                }
                
                print(f"   📊 Return: {metrics.get('return_pct', 0):.2f}%")
                print(f"   📈 Trades: {metrics.get('total_trades', 0)}")
                print(f"   🚪 Exits: {metrics.get('exit_summary', 'N/A')}")
                
            else:
                print(f"   ❌ Failed: {result.stderr[:100]}...")
                test_result = {
                    'config': config['name'],
                    'success': False,
                    'duration': duration,
                    'error': result.stderr[:200]
                }
            
            results.append(test_result)
            
        except subprocess.TimeoutExpired:
            print(f"   ⏰ Timed out after 60s")
            results.append({
                'config': config['name'],
                'success': False,
                'duration': 60,
                'error': 'Timeout'
            })
        except Exception as e:
            print(f"   💥 Error: {e}")
            results.append({
                'config': config['name'],
                'success': False,
                'duration': 0,
                'error': str(e)
            })
    
    # Analyze results
    print(f"\n📊 COMPARISON RESULTS")
    print("=" * 40)
    
    successful = [r for r in results if r['success']]
    
    if not successful:
        print("❌ No successful tests")
        return
    
    # Sort by return
    successful.sort(key=lambda x: x.get('return_pct', 0), reverse=True)
    
    print(f"\n🏆 PERFORMANCE RANKING:")
    for i, result in enumerate(successful, 1):
        print(f"{i}. {result['config'].upper()}")
        print(f"   Return: {result.get('return_pct', 0):.2f}%")
        print(f"   Trades: {result.get('total_trades', 0)}")
        print(f"   Duration: {result.get('duration', 0):.1f}s")
        print(f"   Confidence exits: {result.get('confidence_exits', 0)}")
        print(f"   Signal decay exits: {result.get('signal_decay_exits', 0)}")
        print()
    
    # Save results
    results_df = pd.DataFrame(results)
    results_path = Path('/Users/danielwestgaard/progRel/ml-pred/results/fast_parameter_comparison.csv')
    results_df.to_csv(results_path, index=False)
    
    print(f"💾 Results saved to: {results_path}")
    
    if successful:
        best = successful[0]
        print(f"\n🎯 BEST CONFIGURATION: {best['config'].upper()}")
        print(f"   Return: {best.get('return_pct', 0):.2f}%")
        print(f"   Parameters: confidence={best.get('confidence_threshold', 0)}, decay={best.get('signal_decay', 0)}, life={best.get('min_signal_life', 0)}min")

def extract_metrics(stdout):
    """Extract metrics from backtest output."""
    
    metrics = {
        'return_pct': 0.0,
        'total_trades': 0,
        'confidence_exits': 0,
        'signal_decay_exits': 0,
        'volatility_exits': 0,
        'stop_loss_exits': 0,
        'take_profit_exits': 0,
        'final_value': 10000
    }
    
    try:
        lines = stdout.split('\n')
        
        for line in lines:
            # Extract return percentage
            if 'Return:' in line and '%' in line:
                try:
                    return_str = line.split('Return:')[1].split('%')[0].strip()
                    metrics['return_pct'] = float(return_str)
                except:
                    pass
            
            # Extract final value  
            elif 'Final Portfolio Value:' in line or 'Portfolio value:' in line:
                try:
                    value_str = line.split('$')[1].replace(',', '').replace(')', '').strip()
                    metrics['final_value'] = float(value_str)
                    metrics['return_pct'] = (metrics['final_value'] - 10000) / 10000 * 100
                except:
                    pass
            
            # Count trades and exits
            elif 'Trades:' in line:
                try:
                    trades_str = line.split('Trades:')[1].split(',')[0].strip()
                    metrics['total_trades'] = int(trades_str)
                except:
                    pass
            elif 'Position entered' in line or 'Order executed: BUY' in line:
                metrics['total_trades'] += 1
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
        
        # Create exit summary
        total_exits = (metrics['confidence_exits'] + metrics['signal_decay_exits'] + 
                      metrics['volatility_exits'] + metrics['stop_loss_exits'] + metrics['take_profit_exits'])
        
        if total_exits > 0:
            exit_types = []
            if metrics['confidence_exits'] > 0:
                exit_types.append(f"Conf:{metrics['confidence_exits']}")
            if metrics['signal_decay_exits'] > 0:
                exit_types.append(f"Decay:{metrics['signal_decay_exits']}")
            if metrics['volatility_exits'] > 0:
                exit_types.append(f"Vol:{metrics['volatility_exits']}")
            if metrics['stop_loss_exits'] > 0:
                exit_types.append(f"SL:{metrics['stop_loss_exits']}")
            if metrics['take_profit_exits'] > 0:
                exit_types.append(f"TP:{metrics['take_profit_exits']}")
            
            metrics['exit_summary'] = " | ".join(exit_types)
        else:
            metrics['exit_summary'] = "No exits"
    
    except Exception as e:
        print(f"⚠️  Error extracting metrics: {e}")
    
    return metrics

if __name__ == "__main__":
    run_fast_comparison()