"""
Clean Target Hypothesis Generators
==================================

This is the streamlined version without backward compatibility complexity.
All hypothesis-specific target generation logic is organized here cleanly.
"""

import numpy as np
import pandas as pd
from typing import List

from .base_target_generator import BaseTargetGenerator
from utils.logging_utils import get_logger

logger = get_logger()


class TargetHypothesis(BaseTargetGenerator):
    """
    Clean, streamlined target hypothesis generator.
    
    No backward compatibility cruft - just clean, modular hypothesis implementations.
    """
    
    def get_required_features(self) -> List[str]:
        """Get required features based on the hypothesis being tested."""
        
        # Spread compression hypothesis variants
        if self.hypothesis in ['spread_compression_patterns_v4', 'spread_compression_patterns']:
            return [
                'depth_stability',
                'institutional_footprint', 
                'liquidity_stress',
                'microstructure_health',
                'volume_ratio',
                'london_session'
            ]
        
        elif self.hypothesis == 'spread_compression_patterns_v3':
            return [
                'depth_stability',
                'institutional_footprint', 
                'liquidity_stress',
                'microstructure_health',
                'volume_ratio',
                'london_session',
                'signal_conviction'
            ]
        
        # Session-based hypothesis
        elif self.hypothesis == 'session_based_tod_anomalies':
            return [
                'atr',
                'primary_session',
                'expected_atr'
            ]
        
        else:
            logger.error(f"Unknown hypothesis: {self.hypothesis}")
            logger.info("Available hypotheses: spread_compression_patterns_v4, spread_compression_patterns_v3, session_based_tod_anomalies")
            raise ValueError(f"Unsupported hypothesis: {self.hypothesis}")
    
    def get_feature_dependencies(self) -> dict | None:
        """Use global registry by default."""
        return None  # Use global registry from FeatureRegistry
    
    # Hypothesis implementations
    def spread_compression_patterns_v4(self):
        """
        Clean V4: Forward-looking target with proper temporal logic.
        
        This is the cleaned-up version without any backward compatibility code.
        """
        try:
            # Step 1: Automatic feature generation using base class
            required_features = self.get_required_features()
            self.generate_features_with_dependencies(required_features)
            
            # Validate critical features
            critical_features = ['depth_stability', 'institutional_footprint', 'volume_ratio']
            missing_critical = [f for f in critical_features if f not in self.df.columns]
            
            if missing_critical:
                logger.error(f"Cannot proceed - missing critical features: {missing_critical}")
                return False
            
            logger.info("=== CLEAN V4: FORWARD-LOOKING PREDICTION ===")
            
            # Forward-looking target generation
            logger.info("Calculating forward-looking targets...")
            
            # Multi-horizon future returns (no lookahead bias)
            future_return_5min = self.df['close'].shift(-5) / self.df['close'] - 1
            future_return_10min = self.df['close'].shift(-10) / self.df['close'] - 1 
            future_return_15min = self.df['close'].shift(-15) / self.df['close'] - 1
            
            # Realistic thresholds for GBPUSD
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
            
            # Temporal momentum features (no lookahead)
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
            
            # Multi-horizon target generation
            logger.info("Creating multi-horizon targets...")
            
            # Primary target: 10-minute forward returns
            primary_target = np.where(
                future_return_10min > dynamic_threshold_up, 1,
                np.where(future_return_10min < dynamic_threshold_down, 2, 0)
            )
            
            # Multi-horizon targets
            short_term_target = np.where(
                future_return_5min > dynamic_threshold_up * 0.7, 1,
                np.where(future_return_5min < dynamic_threshold_down * 0.7, 2, 0)
            )
            
            medium_term_target = np.where(
                future_return_15min > dynamic_threshold_up * 1.3, 1,
                np.where(future_return_15min < dynamic_threshold_down * 1.3, 2, 0)
            )
            
            # Save targets and features
            self.df['target_spread_compression_v4_directional'] = primary_target
            self.df['target_spread_compression_v4_binary'] = (primary_target > 0).astype(int)
            self.df['target_spread_compression_v4_short'] = short_term_target
            self.df['target_spread_compression_v4_medium'] = medium_term_target
            
            # Temporal features for model input
            self.df['price_momentum_1min'] = price_momentum_1min
            self.df['price_momentum_3min'] = price_momentum_3min
            self.df['price_momentum_5min'] = price_momentum_5min
            self.df['price_acceleration'] = price_acceleration
            self.df['volume_momentum'] = volume_momentum
            self.df['volume_acceleration'] = volume_acceleration
            self.df['volatility_trend'] = volatility_trend
            
            # Results analysis
            target_dist = pd.Series(primary_target).value_counts().sort_index()
            total_samples = len(self.df)
            
            hold_pct = target_dist.get(0, 0) / total_samples * 100
            buy_pct = target_dist.get(1, 0) / total_samples * 100  
            sell_pct = target_dist.get(2, 0) / total_samples * 100
            signal_rate = buy_pct + sell_pct
            
            logger.info(f"=== CLEAN V4 TARGET RESULTS ===")
            logger.info(f"Target distribution: {target_dist.to_dict()}")
            logger.info(f"  Class 0 (HOLD): {target_dist.get(0, 0)} ({hold_pct:.1f}%)")
            logger.info(f"  Class 1 (BUY):  {target_dist.get(1, 0)} ({buy_pct:.1f}%)")
            logger.info(f"  Class 2 (SELL): {target_dist.get(2, 0)} ({sell_pct:.1f}%)")
            logger.info(f"Signal rate: {signal_rate:.1f}%")
            
            # Validation
            if signal_rate < 5:
                logger.warning(f"⚠️  Signal rate {signal_rate:.1f}% below optimal (5-25%)")
            elif signal_rate > 25:
                logger.warning(f"⚠️  Signal rate {signal_rate:.1f}% above optimal (5-25%)")
            else:
                logger.info(f"✅ Signal rate {signal_rate:.1f}% in optimal range (5-25%)")
            
            if buy_pct > 0 and sell_pct > 0:
                balance_ratio = min(buy_pct, sell_pct) / max(buy_pct, sell_pct)
                logger.info(f"Directional balance: {balance_ratio:.3f}")
                
                if balance_ratio > 0.6:
                    logger.info("✅ Good directional balance")
                else:
                    logger.warning(f"⚠️ Imbalanced: {('BUY' if buy_pct > sell_pct else 'SELL')} dominates")
            
            logger.info("✅ Clean V4 target generation completed!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate clean V4 target: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def session_based_tod_anomalies(self):
        """
        Clean session-based anomaly detection.
        """
        try:
            # Automatic feature generation
            required_features = self.get_required_features()
            self.generate_features_with_dependencies(required_features)
            
            # Validate required features
            if 'atr' not in self.df.columns or 'primary_session' not in self.df.columns:
                logger.error("Missing required features: 'atr' and 'primary_session'")
                return False
            
            # Session-specific anomaly detection logic
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
                
                logger.info("✅ Clean session-based anomaly target created!")
                return True
            else:
                logger.error("Missing 'expected_atr' feature for target calculation")
                return False
                
        except Exception as e:
            logger.error(f"Failed to generate session anomaly target: {e}")
            return False
    
    def spread_compression_patterns_v3(self):
        """
        Clean V3 implementation for reference.
        This can be implemented if needed, but V4 is preferred.
        """
        logger.warning("V3 implementation not provided in clean version. Use V4 instead.")
        return self.spread_compression_patterns_v4()


# Example of how to easily add a new hypothesis
class NewHypothesisExample(BaseTargetGenerator):
    """
    Example showing how clean and simple it is to add a new hypothesis.
    """
    
    def get_required_features(self) -> List[str]:
        return ['volume_ratio', 'depth_stability']
    
    def get_feature_dependencies(self) -> dict | None:
        return None  # Use global registry
    
    def my_new_hypothesis(self):
        """
        Example new hypothesis - super clean implementation.
        """
        try:
            # Step 1: Auto feature generation
            required_features = self.get_required_features()
            self.generate_features_with_dependencies(required_features)
            
            # Step 2: Your hypothesis logic here
            self.df['target_my_new_hypothesis'] = (
                self.df['volume_ratio'] > 1.5
            ).astype(int)
            
            logger.info("✅ New hypothesis target created!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate new hypothesis target: {e}")
            return False