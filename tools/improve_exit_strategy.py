#!/usr/bin/env python3
"""
Intelligent Exit Strategy Improvements
======================================

Instead of hardcoded holding periods, implement smart exit conditions:
1. Confidence-based exit thresholds
2. Market volatility-aware exits  
3. Signal strength decay detection
4. Multi-timeframe confirmation
"""

import yaml
from pathlib import Path

def create_improved_exit_strategy_config():
    """Create enhanced exit strategy configuration."""
    
    # Load current config
    config_path = Path('/Users/danielwestgaard/progRel/ml-pred/hypothesis/spread_compression_patterns/config_backtest.yml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Enhanced signal processing with confidence-based exits
    config['backtesting']['signal_processing'] = {
        'confidence_threshold': 0.70,  # Higher entry threshold
        'signal_filters': ['session_filter'],
        
        # NEW: Intelligent exit conditions
        'exit_strategy': {
            'method': 'intelligent',  # vs 'immediate' model exit
            'confidence_exit_threshold': 0.30,  # Exit when model confidence drops below 30%
            'signal_strength_decay': 0.20,  # Exit when signal strength drops 20% from peak
            'volatility_exit_multiplier': 1.5,  # Exit when volatility exceeds 1.5x entry volatility
            'minimum_signal_life': 3,  # Minimum 3 minutes before considering exits (not forced holding)
            'confirmation_windows': [1, 3, 5],  # Multi-timeframe confirmation
        }
    }
    
    # Enhanced risk management with adaptive parameters
    config['backtesting']['risk_management'].update({
        'adaptive_stops': True,  # Enable adaptive stop loss based on volatility
        'volatility_multiplier': 2.0,  # Stop loss = 2x recent volatility
        'trailing_stop': True,  # Enable trailing stops
        'trailing_stop_distance': 0.01,  # 1% trailing distance
        'profit_target_scaling': True,  # Scale profit targets based on signal strength
    })
    
    return config

def create_smart_period_selection_strategy():
    """Create intelligent period selection strategy."""
    
    return {
        'period_selection': {
            'method': 'adaptive',  # Dynamic period selection
            'criteria': {
                'volatility_threshold': 0.008,  # Select periods with at least 0.8% volatility
                'signal_density_min': 5,  # At least 5 signals per day expected
                'market_session_focus': ['london_session'],  # Focus on London session
                'avoid_low_liquidity': True,  # Skip low liquidity periods
                'economic_calendar_filter': True,  # Avoid major news events
            },
            
            # Rolling window approach instead of fixed periods
            'rolling_windows': {
                'signal_detection_window': 7,  # 7-day rolling windows for signal analysis
                'performance_validation_window': 14,  # 14-day validation
                'adaptive_rebalancing': True,  # Adjust parameters based on performance
            }
        }
    }

def implement_signal_strength_tracking():
    """Implement signal strength tracking for intelligent exits."""
    
    signal_tracking_config = {
        'signal_tracking': {
            'track_signal_evolution': True,
            'signal_strength_features': [
                'compression_ratio_micro',
                'volume_ratio', 
                'signal_strength_v2',
                'compression_consistency'
            ],
            
            # Exit conditions based on signal deterioration
            'exit_conditions': {
                'signal_strength_drop': 0.25,  # Exit if signal strength drops 25%
                'compression_reversal': 0.15,  # Exit if compression reverses by 15%
                'volume_confirmation_loss': True,  # Exit if volume confirmation disappears
                'multi_feature_consensus': 0.6,  # Need 60% of features to agree on exit
            },
            
            # Dynamic confidence adjustment
            'confidence_adjustment': {
                'signal_age_decay': 0.02,  # Confidence decays 2% per minute
                'volatility_boost': 0.1,  # Increase confidence 10% in high volatility
                'session_bonus': 0.05,  # 5% confidence bonus during London session
            }
        }
    }
    
    return signal_tracking_config

def create_test_configs_with_improvements():
    """Create test configurations with intelligent improvements."""
    
    base_config = create_improved_exit_strategy_config()
    period_strategy = create_smart_period_selection_strategy()
    signal_tracking = implement_signal_strength_tracking()
    
    # Merge configurations
    enhanced_config = {**base_config}
    enhanced_config['backtesting'].update(period_strategy)
    enhanced_config['backtesting'].update(signal_tracking)
    
    # Create different test variants
    test_configs = []
    
    # 1. Conservative exit strategy
    conservative_config = enhanced_config.copy()
    conservative_config['backtesting']['signal_processing']['exit_strategy']['confidence_exit_threshold'] = 0.40
    conservative_config['backtesting']['signal_processing']['exit_strategy']['signal_strength_decay'] = 0.15
    test_configs.append(('conservative_exits', conservative_config))
    
    # 2. Aggressive exit strategy  
    aggressive_config = enhanced_config.copy()
    aggressive_config['backtesting']['signal_processing']['exit_strategy']['confidence_exit_threshold'] = 0.20
    aggressive_config['backtesting']['signal_processing']['exit_strategy']['signal_strength_decay'] = 0.30
    test_configs.append(('aggressive_exits', aggressive_config))
    
    # 3. Volatility-adaptive strategy
    volatility_config = enhanced_config.copy()
    volatility_config['backtesting']['signal_processing']['exit_strategy']['volatility_exit_multiplier'] = 2.0
    volatility_config['backtesting']['risk_management']['volatility_multiplier'] = 1.5
    test_configs.append(('volatility_adaptive', volatility_config))
    
    return test_configs

def save_improved_configs():
    """Save all improved configurations for testing."""
    
    configs_dir = Path('/Users/danielwestgaard/progRel/ml-pred/hypothesis/spread_compression_patterns/improved_configs')
    configs_dir.mkdir(exist_ok=True)
    
    test_configs = create_test_configs_with_improvements()
    
    saved_configs = []
    for config_name, config in test_configs:
        config_file = configs_dir / f'config_backtest_{config_name}.yml'
        
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        saved_configs.append(config_file)
        print(f"✅ Saved: {config_file}")
    
    return saved_configs

def create_period_selection_script():
    """Create script for intelligent period selection."""
    
    script_content = '''#!/usr/bin/env python3
"""
Intelligent Period Selection
============================

Select optimal testing periods based on market conditions rather than arbitrary dates.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def identify_optimal_periods(data_path, target_days_per_period=7):
    """Identify periods with optimal characteristics for spread compression testing."""
    
    # Load data
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    
    # Calculate rolling metrics for period selection
    window = target_days_per_period * 24 * 60  # Convert days to minutes
    
    df['volatility'] = df['close'].rolling(window).std()
    df['signal_density'] = df['target_spread_compression_binary'].rolling(window).sum()
    df['london_session_pct'] = df['london_session'].rolling(window).mean()
    df['volume_activity'] = df['volume'].rolling(window).mean()
    
    # Score each potential period
    df['period_score'] = (
        (df['volatility'] / df['volatility'].median()) * 0.3 +  # 30% weight on volatility
        (df['signal_density'] / 10) * 0.4 +  # 40% weight on signal density
        df['london_session_pct'] * 0.2 +  # 20% weight on London session coverage
        (df['volume_activity'] / df['volume_activity'].median()) * 0.1  # 10% weight on volume
    )
    
    # Find top periods
    optimal_periods = []
    
    # Use non-overlapping windows
    current_date = df.index.min()
    end_date = df.index.max() - timedelta(days=target_days_per_period)
    
    while current_date < end_date:
        period_end = current_date + timedelta(days=target_days_per_period)
        period_data = df[current_date:period_end]
        
        if len(period_data) > window * 0.8:  # At least 80% data coverage
            avg_score = period_data['period_score'].mean()
            signal_count = period_data['target_spread_compression_binary'].sum()
            
            optimal_periods.append({
                'start_date': current_date,
                'end_date': period_end,
                'score': avg_score,
                'signal_count': signal_count,
                'volatility': period_data['volatility'].mean(),
                'london_session_pct': period_data['london_session_pct'].mean(),
            })
        
        current_date += timedelta(days=target_days_per_period // 2)  # 50% overlap for sliding window
    
    # Sort by score and return top periods
    optimal_df = pd.DataFrame(optimal_periods)
    optimal_df = optimal_df.sort_values('score', ascending=False)
    
    return optimal_df.head(10)  # Return top 10 periods

if __name__ == "__main__":
    data_path = '/Users/danielwestgaard/progRel/ml-pred/storage/data/capital_com/GBPUSD/processed/1m/2024-06-01_2025-05-23.csv'
    optimal_periods = identify_optimal_periods(data_path)
    
    print("🎯 TOP 10 OPTIMAL PERIODS FOR TESTING:")
    print("=" * 60)
    for i, row in optimal_periods.iterrows():
        print(f"{i+1:2d}. {row['start_date'].strftime('%Y-%m-%d')} to {row['end_date'].strftime('%Y-%m-%d')}")
        print(f"    Score: {row['score']:.3f} | Signals: {row['signal_count']:3.0f} | "
              f"Volatility: {row['volatility']:.6f} | London %: {row['london_session_pct']:.1%}")
        print()
'''
    
    script_path = Path('/Users/danielwestgaard/progRel/ml-pred/intelligent_period_selection.py')
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"✅ Created intelligent period selection script: {script_path}")
    return script_path

def main():
    """Main execution function."""
    print("🚀 Creating Intelligent Exit Strategy and Period Selection Solutions")
    print("=" * 70)
    
    # Save improved configurations
    print("\n📁 Creating improved configurations...")
    saved_configs = save_improved_configs()
    
    # Create period selection script
    print("\n🎯 Creating intelligent period selection...")
    period_script = create_period_selection_script()
    
    print(f"\n✅ SOLUTIONS CREATED:")
    print(f"📊 Exit Strategy Improvements:")
    print(f"   • Confidence-based exits (vs immediate model exits)")
    print(f"   • Signal strength decay detection")
    print(f"   • Volatility-adaptive exits")
    print(f"   • Multi-timeframe confirmation")
    print(f"   • Adaptive stop losses and trailing stops")
    
    print(f"\n🎯 Period Selection Improvements:")
    print(f"   • Market condition-based selection")
    print(f"   • Signal density optimization") 
    print(f"   • Volatility and session-aware selection")
    print(f"   • Rolling windows instead of fixed periods")
    
    print(f"\n📂 Files Created:")
    for config_file in saved_configs:
        print(f"   • {config_file}")
    print(f"   • {period_script}")
    
    print(f"\n🎮 NEXT STEPS:")
    print(f"   1. Run intelligent period selection: python intelligent_period_selection.py")
    print(f"   2. Test improved configs: python main.py -hyp improved_configs/config_backtest_conservative_exits.yml")
    print(f"   3. Compare results with original approach")

if __name__ == "__main__":
    main()