#!/usr/bin/env python3
"""
Dynamic Parameter Configuration for Backtesting

This script creates adaptive parameters that adjust based on market conditions
and model performance during backtesting.
"""

import yaml
import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DynamicParameterGenerator:
    """Generate dynamic parameters based on market conditions and model performance"""
    
    def __init__(self, base_config_path: str):
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
    
    def calculate_market_volatility(self, data_path: str, lookback_days: int = 5) -> float:
        """Calculate recent market volatility to adjust parameters"""
        try:
            df = pd.read_csv(data_path)
            df['date'] = pd.to_datetime(df['date'])
            
            # Get recent data
            recent_data = df.tail(lookback_days * 1440)  # 1440 minutes per day
            price_changes = recent_data['close'].pct_change().dropna()
            
            # Calculate volatility (standard deviation of returns)
            volatility = price_changes.std() * np.sqrt(1440)  # Annualized
            return volatility
            
        except Exception as e:
            logger.warning(f"Could not calculate volatility: {e}")
            return 0.01  # Default volatility
    
    def calculate_signal_quality_score(self, recent_performance: Dict[str, float]) -> float:
        """Calculate signal quality based on recent performance"""
        if not recent_performance:
            return 0.5  # Neutral score
        
        win_rate = recent_performance.get('win_rate', 33.0) / 100.0
        return_pct = recent_performance.get('return_pct', 0.0) / 100.0
        
        # Combine win rate and returns for quality score
        quality_score = (win_rate * 0.6) + (min(max(return_pct, -0.1), 0.1) + 0.1) * 0.4 / 0.2
        return max(0.1, min(1.0, quality_score))
    
    def generate_adaptive_config(self, 
                                data_path: str,
                                recent_performance: Dict[str, float] = None,
                                market_regime: str = 'normal') -> str:
        """Generate configuration with adaptive parameters"""
        
        # Calculate market conditions
        volatility = self.calculate_market_volatility(data_path)
        signal_quality = self.calculate_signal_quality_score(recent_performance or {})
        
        logger.info(f"Market volatility: {volatility:.4f}, Signal quality: {signal_quality:.2f}")
        
        # Adaptive parameter calculation
        config = self.base_config.copy()
        
        # 1. VOLATILITY-BASED ADJUSTMENTS
        vol_multiplier = min(max(volatility / 0.015, 0.5), 2.0)  # Scale factor based on volatility
        
        # 2. SIGNAL PROCESSING ADAPTATIONS
        # Higher volatility = need higher confidence
        base_confidence = 0.08
        adaptive_confidence = base_confidence * (1 + (vol_multiplier - 1) * 0.5)
        adaptive_confidence = min(max(adaptive_confidence, 0.06), 0.15)
        
        # Signal quality affects signal frequency
        signal_quality_multiplier = 0.5 + signal_quality * 1.0  # 0.5 to 1.5 range
        cooldown_base = 0.5
        adaptive_cooldown = cooldown_base / signal_quality_multiplier
        adaptive_cooldown = min(max(adaptive_cooldown, 0.25), 2.0)
        
        # Max signals per hour based on volatility and quality
        base_signals_per_hour = 180
        adaptive_signals_per_hour = int(base_signals_per_hour * vol_multiplier * signal_quality_multiplier)
        adaptive_signals_per_hour = min(max(adaptive_signals_per_hour, 60), 400)
        
        # 3. RISK MANAGEMENT ADAPTATIONS
        # Higher volatility = tighter stops and smaller positions
        base_stop_loss = 0.012
        adaptive_stop_loss = base_stop_loss * (0.8 + vol_multiplier * 0.4)
        adaptive_stop_loss = min(max(adaptive_stop_loss, 0.008), 0.020)
        
        base_take_profit = 0.020
        adaptive_take_profit = base_take_profit * (0.9 + vol_multiplier * 0.6)
        adaptive_take_profit = min(max(adaptive_take_profit, 0.015), 0.035)
        
        # Position size inversely related to volatility
        base_position_size = 0.08
        adaptive_position_size = base_position_size * (1.5 - vol_multiplier * 0.5)
        adaptive_position_size = min(max(adaptive_position_size, 0.04), 0.12)
        
        # 4. POSITION MANAGEMENT ADAPTATIONS
        # Better signal quality = allow more concurrent positions
        base_max_positions = 3
        adaptive_max_positions = int(base_max_positions * signal_quality_multiplier)
        adaptive_max_positions = min(max(adaptive_max_positions, 2), 5)
        
        # Apply adaptive parameters
        config['backtesting']['signal_processing'].update({
            'confidence_threshold': round(adaptive_confidence, 3),
            'cooldown_minutes': round(adaptive_cooldown, 2),
            'max_signals_per_hour': adaptive_signals_per_hour,
            'min_signal_strength': round(adaptive_confidence, 3)
        })
        
        config['backtesting']['risk_management'].update({
            'stop_loss_pct': round(adaptive_stop_loss, 4),
            'take_profit_pct': round(adaptive_take_profit, 4),
            'max_position_size': round(adaptive_position_size, 3),
            'max_concurrent_positions': adaptive_max_positions
        })
        
        config['backtesting']['position_management']['max_concurrent_positions'] = adaptive_max_positions
        
        # Add metadata about adaptations
        config['adaptive_metadata'] = {
            'volatility': round(volatility, 4),
            'signal_quality_score': round(signal_quality, 3),
            'vol_multiplier': round(vol_multiplier, 2),
            'signal_quality_multiplier': round(signal_quality_multiplier, 2),
            'market_regime': market_regime,
            'generation_timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Save adaptive configuration
        output_path = f"adaptive_config_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.yml"
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Adaptive configuration saved to: {output_path}")
        logger.info(f"Key adaptations:")
        logger.info(f"  - Confidence threshold: {adaptive_confidence:.3f}")
        logger.info(f"  - Cooldown: {adaptive_cooldown:.2f} min")
        logger.info(f"  - Stop loss: {adaptive_stop_loss:.3f}%")
        logger.info(f"  - Take profit: {adaptive_take_profit:.3f}%")
        logger.info(f"  - Position size: {adaptive_position_size:.3f}")
        logger.info(f"  - Max positions: {adaptive_max_positions}")
        
        return output_path

def create_regime_specific_configs():
    """Create configurations for different market regimes"""
    
    base_config = "hypothesis/spread_compression_patterns/config_backtest_v4_enhanced.yml"
    data_path = "storage/data/capital_com/GBPUSD/processed/1m/2025-07-02_2025-07-31.csv"
    
    generator = DynamicParameterGenerator(base_config)
    
    # Generate configs for different scenarios
    scenarios = [
        {
            'name': 'Conservative',
            'recent_performance': {'win_rate': 25.0, 'return_pct': -2.0},
            'regime': 'low_confidence'
        },
        {
            'name': 'Balanced', 
            'recent_performance': {'win_rate': 40.0, 'return_pct': 1.0},
            'regime': 'normal'
        },
        {
            'name': 'Aggressive',
            'recent_performance': {'win_rate': 55.0, 'return_pct': 3.0}, 
            'regime': 'high_confidence'
        }
    ]
    
    configs = []
    for scenario in scenarios:
        print(f"\n=== Generating {scenario['name']} Configuration ===")
        config_path = generator.generate_adaptive_config(
            data_path=data_path,
            recent_performance=scenario['recent_performance'],
            market_regime=scenario['regime']
        )
        configs.append({
            'name': scenario['name'],
            'path': config_path,
            'scenario': scenario
        })
    
    return configs

def main():
    """Generate multiple adaptive configurations"""
    configs = create_regime_specific_configs()
    
    print("\n" + "="*60)
    print("ADAPTIVE CONFIGURATIONS GENERATED")
    print("="*60)
    
    for config in configs:
        print(f"\n{config['name']} Configuration:")
        print(f"  File: {config['path']}")
        print(f"  Regime: {config['scenario']['regime']}")
        print(f"  Performance assumption: {config['scenario']['recent_performance']}")
    
    print("\n💡 Next steps:")
    print("1. Test each configuration with: python main.py --hyp [config_file]")
    print("2. Compare results to identify best approach")
    print("3. Consider implementing real-time adaptation in backtesting engine")

if __name__ == "__main__":
    main()