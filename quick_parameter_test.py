#!/usr/bin/env python3
"""
Quick Parameter Testing - Test 3 promising configurations rapidly
"""

import yaml
import subprocess
import time
from typing import Dict, List

def create_test_config(base_config_path: str, params: Dict, name: str) -> str:
    """Create a test configuration with specific parameters"""
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
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
        'max_position_size': params['max_position_size']
    })
    
    # Save test config
    test_config_path = f"test_config_{name.lower()}.yml"
    with open(test_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return test_config_path

def run_quick_backtest(config_path: str) -> Dict:
    """Run backtest and extract key results"""
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

def main():
    """Test 3 promising parameter combinations"""
    base_config = "hypothesis/spread_compression_patterns/config_backtest_v4_enhanced.yml"
    
    # Define 3 strategic test configurations
    test_configs = [
        {
            'name': 'Conservative',
            'params': {
                'confidence_threshold': 0.12,  # Higher confidence
                'cooldown_minutes': 1.0,       # Longer cooldown
                'max_signals_per_hour': 120,   # Fewer signals
                'stop_loss_pct': 0.010,        # Tight stop
                'take_profit_pct': 0.025,      # Wider target
                'max_position_size': 0.06      # Smaller positions
            },
            'rationale': 'Quality over quantity - fewer, higher-confidence trades'
        },
        {
            'name': 'Aggressive', 
            'params': {
                'confidence_threshold': 0.06,  # Lower confidence
                'cooldown_minutes': 0.25,      # Short cooldown
                'max_signals_per_hour': 300,   # More signals
                'stop_loss_pct': 0.015,        # Wider stop
                'take_profit_pct': 0.020,      # Tighter target
                'max_position_size': 0.10      # Larger positions
            },
            'rationale': 'Volume approach - more trades, accept higher risk'
        },
        {
            'name': 'Balanced',
            'params': {
                'confidence_threshold': 0.09,  # Medium confidence  
                'cooldown_minutes': 0.5,       # Medium cooldown
                'max_signals_per_hour': 200,   # Moderate signals
                'stop_loss_pct': 0.012,        # Medium stop
                'take_profit_pct': 0.022,      # Medium target
                'max_position_size': 0.08      # Medium positions
            },
            'rationale': 'Balanced approach - moderate risk/reward profile'
        }
    ]
    
    print("🚀 QUICK PARAMETER TESTING")
    print("="*50)
    
    results = []
    
    for test_config in test_configs:
        print(f"\n⏳ Testing {test_config['name']} configuration...")
        print(f"   Rationale: {test_config['rationale']}")
        
        # Create config
        config_path = create_test_config(base_config, test_config['params'], test_config['name'])
        
        # Run backtest
        result = run_quick_backtest(config_path)
        result['name'] = test_config['name']
        result['params'] = test_config['params']
        results.append(result)
        
        if result['success']:
            print(f"   ✅ Results: {result['return_pct']:.2f}% return, {result['trades']} trades, {result['win_rate']:.1f}% win rate")
            print(f"   ⏱️  Runtime: {result['runtime']:.1f}s")
        else:
            print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
    
    # Summary
    print("\n" + "="*60)
    print("📊 QUICK TEST RESULTS SUMMARY")
    print("="*60)
    
    successful_results = [r for r in results if r['success']]
    if successful_results:
        # Sort by return
        successful_results.sort(key=lambda x: x['return_pct'], reverse=True)
        
        print(f"\n🏆 RANKING BY RETURN:")
        for i, result in enumerate(successful_results, 1):
            print(f"{i}. {result['name']}: {result['return_pct']:.2f}% (Trades: {result['trades']}, Win Rate: {result['win_rate']:.1f}%)")
        
        # Best configuration
        best = successful_results[0]
        print(f"\n🎯 BEST CONFIGURATION: {best['name']}")
        print(f"   Parameters: {best['params']}")
        print(f"   Performance: {best['return_pct']:.2f}% return, {best['win_rate']:.1f}% win rate")
        
        print(f"\n💡 NEXT STEPS:")
        print(f"1. Use {best['name']} parameters as baseline")
        print(f"2. Consider fine-tuning around these values")
        print(f"3. Test on different time periods for robustness")
    else:
        print("❌ No successful tests completed")

if __name__ == "__main__":
    main()