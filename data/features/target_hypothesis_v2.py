"""
Modular Target Hypothesis Generator - Version 2
===============================================

This is a refactored version of target_hypothesis.py that uses the new BaseTargetGenerator
and centralized feature dependency system. This version is more modular, maintainable,
and easier to extend with new hypotheses.

Key improvements:
- Inherits from BaseTargetGenerator for consistent patterns
- Uses centralized feature dependency resolution
- Eliminates code duplication across hypothesis methods
- Easier to add new hypotheses
- Better error handling and cleanup
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from data.features.base_target_generator import BaseTargetGenerator
from utils.logging_utils import get_logger

logger = get_logger()


class ModularTargetHypothesis(BaseTargetGenerator):
    """
    Modular target hypothesis generator using the new base class architecture.
    
    This class demonstrates how new target generators should be implemented:
    - Define required features
    - Optionally define custom dependencies
    - Implement hypothesis-specific target logic
    - Let the base class handle feature generation and cleanup
    """
    
    def get_required_features(self) -> List[str]:
        """
        Get required features based on the hypothesis being tested.
        
        This method can be made more sophisticated to load requirements
        from configuration files or hypothesis metadata.
        """
        if self.hypothesis == 'spread_compression_patterns_v4':
            return [
                'depth_stability',
                'institutional_footprint', 
                'liquidity_stress',
                'microstructure_health',
                'volume_ratio',
                'london_session'
            ]
        elif self.hypothesis == 'session_based_tod_anomalies':
            return [
                'atr',
                'primary_session',
                'expected_atr'
            ]
        else:
            # Default fallback - can be extended based on hypothesis name patterns
            logger.warning(f"No specific requirements defined for hypothesis: {self.hypothesis}")
            return ['volume_ratio', 'depth_stability']  # Safe defaults
    
    def get_feature_dependencies(self) -> Optional[Dict[str, Tuple[str, List[str]]]]:
        """
        Get custom feature dependencies if needed.
        
        Returns None to use the global dependency registry by default.
        Only override this if the hypothesis needs special dependency logic.
        """
        return None  # Use global registry
    
    def spread_compression_patterns_v4(self):
        """
        MODULAR VERSION: Forward-looking target with proper temporal logic.
        
        This demonstrates the new modular approach:
        1. Define required features
        2. Use base class for feature generation 
        3. Focus only on hypothesis-specific target logic
        4. Let base class handle cleanup
        """
        try:
            # Step 1: Get required features for this hypothesis
            required_features = self.get_required_features()
            
            # Step 2: Use base class to generate missing features
            successfully_generated, all_added_features = self.generate_features_with_dependencies(
                required_features
            )
            
            # Step 3: Validate that critical features are available
            critical_features = ['depth_stability', 'institutional_footprint', 'volume_ratio']
            missing_critical = [f for f in critical_features if f not in self.df.columns]
            
            if missing_critical:
                logger.error(f"Cannot proceed - missing critical features: {missing_critical}")
                return False
            
            logger.info("=== MODULAR V4: FORWARD-LOOKING PREDICTION ===")
            
            # Step 4: Forward-looking target generation (hypothesis-specific logic)
            logger.info("Calculating forward-looking targets...")
            
            # Multi-horizon future returns (no lookahead bias)
            future_return_5min = self.df['close'].shift(-5) / self.df['close'] - 1
            future_return_10min = self.df['close'].shift(-10) / self.df['close'] - 1 
            future_return_15min = self.df['close'].shift(-15) / self.df['close'] - 1
            
            # Calculate realistic thresholds for GBPUSD trading
            base_price = self.df['close'].median()
            min_pips = 4.0
            pip_value = 0.0001
            
            base_threshold = (min_pips * pip_value) / base_price
            
            price_volatility = self.df['close'].pct_change().rolling(100).std()
            volatility_factor = 0.5
            adaptive_component = price_volatility * volatility_factor
            
            dynamic_threshold_up = base_threshold + adaptive_component
            dynamic_threshold_down = -(base_threshold + adaptive_component)
            
            mean_threshold_pips = (dynamic_threshold_up.mean() * base_price) / pip_value
            
            logger.info(f"Dynamic thresholds - Mean up: {dynamic_threshold_up.mean():.6f} ({mean_threshold_pips:.1f} pips)")
            
            # Step 5: Temporal momentum features (no lookahead)
            logger.info("Generating temporal momentum features...")
            
            price_momentum_1min = self.df['close'].pct_change(1)
            price_momentum_3min = self.df['close'].pct_change(3)  
            price_momentum_5min = self.df['close'].pct_change(5)
            
            price_acceleration = price_momentum_1min.diff()
            volume_momentum = self.df['volume_ratio'].pct_change(3)
            volume_acceleration = volume_momentum.diff()
            
            current_volatility = self.df['high'].subtract(self.df['low'])
            volatility_trend = current_volatility.rolling(5).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0
            )
            
            # Step 6: Signal quality indicators
            structure_quality = (
                (self.df['depth_stability'] > self.df['depth_stability'].quantile(0.60)) &
                (self.df['microstructure_health'] > self.df['microstructure_health'].quantile(0.50)) &
                (self.df['institutional_footprint'] > self.df['institutional_footprint'].quantile(0.70))
            )
            
            momentum_bullish = (
                (price_momentum_3min > price_momentum_3min.quantile(0.70)) |
                (price_acceleration > price_acceleration.quantile(0.75))
            ) & (volume_momentum > 0)
            
            momentum_bearish = (
                (price_momentum_3min < price_momentum_3min.quantile(0.30)) |
                (price_acceleration < price_acceleration.quantile(0.25))
            ) & (volume_momentum > 0)
            
            # Step 7: Multi-horizon target generation
            logger.info("Creating multi-horizon targets...")
            
            # Primary target: 10-minute forward returns
            primary_target = np.where(
                future_return_10min > dynamic_threshold_up, 1,
                np.where(
                    future_return_10min < dynamic_threshold_down, 2,
                    0
                )
            )
            
            # For now, use basic price movement signals
            quality_filtered_target = primary_target
            
            # Multi-horizon targets
            short_term_target = np.where(
                future_return_5min > dynamic_threshold_up * 0.7, 1,
                np.where(
                    future_return_5min < dynamic_threshold_down * 0.7, 2,
                    0
                )
            )
            
            medium_term_target = np.where(
                future_return_15min > dynamic_threshold_up * 1.3, 1,
                np.where(
                    future_return_15min < dynamic_threshold_down * 1.3, 2,
                    0
                )
            )
            
            # Step 8: Save all targets
            self.df['target_spread_compression_v4_directional'] = quality_filtered_target
            self.df['target_spread_compression_v4_binary'] = (quality_filtered_target > 0).astype(int)
            self.df['target_spread_compression_v4_short'] = short_term_target
            self.df['target_spread_compression_v4_medium'] = medium_term_target
            
            # Save temporal features for model input (no lookahead)
            self.df['price_momentum_1min'] = price_momentum_1min
            self.df['price_momentum_3min'] = price_momentum_3min
            self.df['price_momentum_5min'] = price_momentum_5min
            self.df['price_acceleration'] = price_acceleration
            self.df['volume_momentum'] = volume_momentum
            self.df['volume_acceleration'] = volume_acceleration
            self.df['volatility_trend'] = volatility_trend
            
            # Step 9: Results analysis
            target_dist = pd.Series(quality_filtered_target).value_counts().sort_index()
            total_samples = len(self.df)
            
            hold_pct = target_dist.get(0, 0) / total_samples * 100
            buy_pct = target_dist.get(1, 0) / total_samples * 100  
            sell_pct = target_dist.get(2, 0) / total_samples * 100
            signal_rate = buy_pct + sell_pct
            
            logger.info(f"=== MODULAR V4 TARGET RESULTS ===")
            logger.info(f"Target distribution: {target_dist.to_dict()}")
            logger.info(f"  Class 0 (HOLD): {target_dist.get(0, 0)} ({hold_pct:.1f}%)")
            logger.info(f"  Class 1 (BUY):  {target_dist.get(1, 0)} ({buy_pct:.1f}%)")
            logger.info(f"  Class 2 (SELL): {target_dist.get(2, 0)} ({sell_pct:.1f}%)")
            logger.info(f"Signal rate: {signal_rate:.1f}%")
            
            # Validation
            future_samples_available = (~future_return_10min.isna()).sum()
            logger.info(f"Future return samples: {future_samples_available}/{total_samples}")
            
            if signal_rate < 0.1:
                logger.error(f"🚨 Signal rate {signal_rate:.1f}% too low!")
                raise ValueError(f"Target generation failed: signal rate too low")
            elif signal_rate < 5:
                logger.warning(f"⚠️  Signal rate {signal_rate:.1f}% below optimal (5-25%)")
            elif signal_rate > 25:
                logger.warning(f"⚠️  Signal rate {signal_rate:.1f}% above optimal (5-25%)")
            else:
                logger.info(f"✅ Signal rate {signal_rate:.1f}% in target range (5-25%)")
            
            # Balance check
            if buy_pct > 0 and sell_pct > 0:
                balance_ratio = min(buy_pct, sell_pct) / max(buy_pct, sell_pct)
                logger.info(f"Directional balance: {balance_ratio:.3f}")
                
                if balance_ratio > 0.6:
                    logger.info("✅ Good directional balance")
                else:
                    logger.warning(f"⚠️ Imbalanced: {('BUY' if buy_pct > sell_pct else 'SELL')} dominates")
            
            logger.info("✅ Modular V4 target generation completed!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate modular V4 target: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def session_based_tod_anomalies(self):
        """
        MODULAR VERSION: Session-based time-of-day anomaly detection.
        
        Demonstrates how the new modular approach works for a different hypothesis.
        """
        try:
            # Step 1: Get required features using base class
            required_features = self.get_required_features()
            
            # Step 2: Generate missing features
            successfully_generated, all_added_features = self.generate_features_with_dependencies(
                required_features
            )
            
            # Step 3: Validate required features
            if 'atr' not in self.df.columns or 'primary_session' not in self.df.columns:
                logger.error("Missing required features: 'atr' and 'primary_session'")
                return False
            
            # Step 4: Session-specific anomaly detection logic
            self.df['session_atr_mean'] = self.df.groupby('primary_session')['atr'].transform(
                lambda x: x.rolling(window=100, min_periods=20).mean()
            )
            self.df['session_atr_std'] = self.df.groupby('primary_session')['atr'].transform(
                lambda x: x.rolling(window=100, min_periods=20).std()
            )
            
            # Target: Future volatility > session mean + 1.5 * session std
            if 'expected_atr' in self.df.columns:
                self.df['target_session_vol_anomaly'] = (
                    (self.df['expected_atr'] > self.df['session_atr_mean'] + 1.5 * self.df['session_atr_std'])
                ).astype(int)
                
                logger.info("✅ Session-based anomaly target created successfully")
                return True
            else:
                logger.error("Missing 'expected_atr' feature for target calculation")
                return False
                
        except Exception as e:
            logger.error(f"Failed to generate session anomaly target: {e}")
            return False
    
    # Example of how to easily add a new hypothesis
    def new_hypothesis_example(self):
        """
        Example showing how easy it is to add a new hypothesis with the modular approach.
        
        Steps to add a new hypothesis:
        1. Add required features to get_required_features()
        2. Optionally add custom dependencies to get_feature_dependencies()
        3. Implement the hypothesis-specific logic here
        4. Base class handles the rest automatically
        """
        try:
            # The base class will automatically:
            # - Generate missing features based on get_required_features()
            # - Handle all dependencies
            # - Clean up temporary features
            # - Provide consistent error handling
            
            required_features = ['volume_ratio', 'depth_stability']  # Example
            successfully_generated, _ = self.generate_features_with_dependencies(required_features)
            
            # Your hypothesis-specific target logic here
            self.df['target_new_hypothesis'] = (self.df['volume_ratio'] > 1.5).astype(int)
            
            logger.info("✅ New hypothesis target created")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate new hypothesis target: {e}")
            return False


# Backward compatibility - create an alias for the old class name
TargetHypothesis = ModularTargetHypothesis