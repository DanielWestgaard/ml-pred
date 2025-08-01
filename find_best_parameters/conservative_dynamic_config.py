#!/usr/bin/env python3
"""
Option A: Conservative Configuration with Dynamic Volatility Adjustments

Takes the best-performing Conservative parameters and adds market volatility-based
adjustments for improved adaptability.
"""

import yaml
import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConservativeDynamicConfig:
    """Conservative baseline with dynamic volatility adjustments"""
    
    def __init__(self, base_config_path: str):
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        
        # Conservative baseline parameters (from quick_parameter_test results)
        self.conservative_baseline = {
            'confidence_threshold': 0.12,
            'cooldown_minutes': 1.0,
            'max_signals_per_hour': 120,
            'stop_loss_pct': 0.010,
            'take_profit_pct': 0.025,
            'max_position_size': 0.06,
            'max_concurrent_positions': 2
        }
    
    def calculate_market_volatility(self, data_path: str, lookback_hours: int = 24) -> float:
        """Calculate recent market volatility"""
        try:
            df = pd.read_csv(data_path)
            df['date'] = pd.to_datetime(df['date'])
            
            # Get recent data (last 24 hours = 1440 minutes)
            recent_data = df.tail(lookback_hours * 60)
            price_changes = recent_data['close'].pct_change().dropna()
            
            # Calculate hourly volatility
            volatility = price_changes.std() * np.sqrt(60)  # Scale to hourly
            return volatility
            
        except Exception as e:
            logger.warning(f"Could not calculate volatility: {e}")
            return 0.015  # Default reasonable volatility for GBPUSD
    
    def generate_dynamic_conservative_config(self, data_path: str) -> str:
        """Generate Conservative config with volatility adjustments"""
        
        # Calculate current market volatility
        current_volatility = self.calculate_market_volatility(data_path)
        
        # Define volatility ranges and multipliers
        low_vol_threshold = 0.010   # Low volatility
        high_vol_threshold = 0.025  # High volatility
        
        # Calculate volatility adjustment factor
        if current_volatility <= low_vol_threshold:
            vol_regime = "LOW"
            vol_multiplier = 0.85  # Slightly more aggressive in low vol
        elif current_volatility >= high_vol_threshold:
            vol_regime = "HIGH" 
            vol_multiplier = 1.20  # More conservative in high vol
        else:
            vol_regime = "NORMAL"
            vol_multiplier = 1.0   # Use baseline
        
        logger.info(f"Market volatility: {current_volatility:.4f} ({vol_regime} regime)")
        logger.info(f"Volatility multiplier: {vol_multiplier:.2f}")
        
        # Apply dynamic adjustments to Conservative baseline
        config = self.base_config.copy()
        
        # Adjust confidence threshold (higher in high vol)
        adjusted_confidence = self.conservative_baseline['confidence_threshold'] * vol_multiplier
        adjusted_confidence = min(max(adjusted_confidence, 0.08), 0.18)
        
        # Adjust cooldown (longer in high vol for safety)
        adjusted_cooldown = self.conservative_baseline['cooldown_minutes'] * vol_multiplier
        adjusted_cooldown = min(max(adjusted_cooldown, 0.5), 3.0)
        
        # Adjust signals per hour (fewer in high vol)
        adjusted_signals = int(self.conservative_baseline['max_signals_per_hour'] / vol_multiplier)
        adjusted_signals = min(max(adjusted_signals, 60), 200)
        
        # Adjust stop loss (tighter in high vol)
        adjusted_stop = self.conservative_baseline['stop_loss_pct'] * vol_multiplier
        adjusted_stop = min(max(adjusted_stop, 0.008), 0.020)
        
        # Take profit stays relatively stable (Conservative approach)
        adjusted_tp = self.conservative_baseline['take_profit_pct'] * (0.9 + vol_multiplier * 0.1)
        adjusted_tp = min(max(adjusted_tp, 0.020), 0.035)
        
        # Position size (smaller in high vol)
        adjusted_position = self.conservative_baseline['max_position_size'] / vol_multiplier
        adjusted_position = min(max(adjusted_position, 0.04), 0.08)
        
        # Update configuration
        config['backtesting']['signal_processing'].update({
            'confidence_threshold': round(adjusted_confidence, 3),
            'cooldown_minutes': round(adjusted_cooldown, 2),
            'max_signals_per_hour': adjusted_signals,
            'min_signal_strength': round(adjusted_confidence, 3)
        })
        
        config['backtesting']['risk_management'].update({
            'stop_loss_pct': round(adjusted_stop, 4),
            'take_profit_pct': round(adjusted_tp, 4),
            'max_position_size': round(adjusted_position, 3),
            'max_concurrent_positions': self.conservative_baseline['max_concurrent_positions']
        })
        
        config['backtesting']['position_management']['max_concurrent_positions'] = self.conservative_baseline['max_concurrent_positions']
        
        # Add metadata (handle NaN values properly)
        volatility_value = current_volatility if not np.isnan(current_volatility) else 0.015
        config['dynamic_metadata'] = {
            'base_strategy': 'Conservative',
            'volatility_regime': vol_regime,
            'market_volatility': round(float(volatility_value), 4),
            'volatility_multiplier': round(float(vol_multiplier), 2),
            'adjustments_applied': {
                'confidence_threshold': f"{self.conservative_baseline['confidence_threshold']:.3f} → {adjusted_confidence:.3f}",
                'cooldown_minutes': f"{self.conservative_baseline['cooldown_minutes']:.2f} → {adjusted_cooldown:.2f}",
                'stop_loss_pct': f"{self.conservative_baseline['stop_loss_pct']:.3f} → {adjusted_stop:.3f}",
                'position_size': f"{self.conservative_baseline['max_position_size']:.3f} → {adjusted_position:.3f}"
            },
            'generation_timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Save configuration
        output_path = f"conservative_dynamic_{vol_regime.lower()}_vol.yml"
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Conservative Dynamic config saved to: {output_path}")
        logger.info(f"Key adjustments for {vol_regime} volatility:")
        logger.info(f"  - Confidence: {adjusted_confidence:.3f}")
        logger.info(f"  - Cooldown: {adjusted_cooldown:.2f} min")
        logger.info(f"  - Stop loss: {adjusted_stop:.3f}%")
        logger.info(f"  - Position size: {adjusted_position:.3f}")
        logger.info(f"  - Max signals/hour: {adjusted_signals}")
        
        return output_path

def main():
    """Generate Conservative Dynamic configuration"""
    base_config = "hypothesis/spread_compression_patterns/config_backtest_v4_enhanced.yml"
    data_path = "storage/data/capital_com/GBPUSD/processed/1m/2025-07-02_2025-07-31.csv"
    
    print("🎯 CONSERVATIVE DYNAMIC CONFIGURATION GENERATOR")
    print("=" * 55)
    
    generator = ConservativeDynamicConfig(base_config)
    config_path = generator.generate_dynamic_conservative_config(data_path)
    
    print(f"\n✅ Configuration generated: {config_path}")
    print(f"\n💡 Next steps:")
    print(f"1. Test with: python main.py --hyp {config_path}")
    print(f"2. Compare results with static Conservative config")
    print(f"3. Monitor how it adapts to different volatility regimes")

if __name__ == "__main__":
    main()