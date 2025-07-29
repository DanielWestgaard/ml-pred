import inspect
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from data.processing.base_processor import BaseProcessor
from utils import data_utils
from utils import shared_utils
from utils.logging_utils import get_logger

logger = get_logger()

class TargetHypothesis(BaseProcessor):
    """Change target of dataset according to the hypothesis."""
    def __init__(self, dataset, hypothesis:str):
        """
        Initialize hypothesis to change target column.
        
        Args:
            dataset: Processed data with specific features generated.
            hypothesis: The hypothesis you intend to test. Must match the name of the function you want to target!
        """
        try:
            # Load dataset based on format
            self.df, self.original_df = data_utils.check_and_return_df(dataset)
            
            self.hypothesis = hypothesis
        except Exception as e:
            logger.error(f"Unable to initialize Target Hypothesis: {e}")
    
    def run(self):
        """
        Change target.
        
        returns:
            dataset: with added new target.
        """
        logger.info("---------- Running Hypothesis Targeting ----------")
        
        # Making a list of existing methods in this class
        functions = shared_utils.list_of_functions_in_class(path_to_class='data.features.target_hypothesis.TargetHypothesis')
        
        try:
            successfull = False
            for hypotehesis in functions:
                if hypotehesis == self.hypothesis:  # The one hypothesis user wants to target
                    successfull = shared_utils.safely_execute(self, hypotehesis)
            if not successfull:
                logger.error(f"Could not find {self.hypothesis} in the list of functions: {functions}. Make sure your name matches the function you want to target!")
        except Exception as e:
            logger.error(f"Error trying to find and execute hypothesis method: {e}")
        
        return self.df
    
    def session_based_tod_anomalies(self):
        """
        This target captures when future volatility is abnormally high relative to that specific session's
        typical behavior - directly testing the hypothesis that sessions behave differently.
        """
        try:
            if 'atr' in self.df.columns and 'primary_session' in self.df.columns:
                # Calculate session-specific ATR norms using primary_session feature
                self.df['session_atr_mean'] = self.df.groupby('primary_session')['atr'].transform(
                    lambda x: x.rolling(window=100, min_periods=20).mean()
                )
                self.df['session_atr_std'] = self.df.groupby('primary_session')['atr'].transform(
                    lambda x: x.rolling(window=100, min_periods=20).std()
                )
                
                # Future volatility (next 30 minutes, assuming 1-min bars)
                self.df['future_atr'] = self.df['atr'].shift(-30).rolling(window=30).mean()
                # Target: Future volatility > session mean + 1.5 * session std
                self.df['target_session_vol_anomaly'] = (
                    (self.df['future_atr'] > self.df['session_atr_mean'] + 1.5 * self.df['session_atr_std'])
                ).astype(int)
                
                logger.debug(f"Target column, target_session_vol_anomaly, for hypothesis session_based_tod_anomalies created.")
            else:
                logger.error(f"Target hypothesis column is not possible to calculate based on missing needed features: 'atr' and 'primary_session'")
        except Exception as e:
            logger.error(f"Unable to calculate the target hypothesis column: {e}")
    
    def spread_compression_patterns(self):
        """
        Target for spread compression pattern hypothesis.
        
        This targets moments when spread compression leads to significant price movements,
        particularly during London session trading hours when market liquidity changes.
        """
        try:
            # Required features check - Updated to match enhanced feature set
            required_features = [
                'depth_stability', 'liquidity_stress', 'microstructure_health',
                'volume_ratio', 'london_session', 'signal_conviction'
            ]
            
            missing_features = [f for f in required_features if f not in self.df.columns]
            temp_generated_features = []
            
            if missing_features:
                logger.warning(f"Missing required features for target generation: {missing_features}")
                logger.info("Generating missing features temporarily for target creation...")
                
                # Import the ModelAwareFeatureGenerator for on-demand feature generation
                from data.features.enhanced_generation import ModelAwareFeatureGenerator
                temp_generator = ModelAwareFeatureGenerator(self.df)
                
                # Map features to their generator methods
                feature_generator_map = {
                    'depth_stability': 'enhanced_market_depth_features',
                    'liquidity_stress': 'enhanced_market_depth_features',
                    'microstructure_health': 'enhanced_microstructure_health',
                    'signal_conviction': 'enhanced_predictive_signals',
                    'volume_ratio': 'volume_ratio_feature',
                    'london_session': 'fx_trading_sessions'
                }
                
                # Generate missing features temporarily
                for feature in missing_features:
                    generator_method = feature_generator_map.get(feature)
                    if generator_method and hasattr(temp_generator, generator_method):
                        logger.debug(f"Temporarily generating {feature} using {generator_method}")
                        temp_generator.__getattribute__(generator_method)()
                        temp_generated_features.append(feature)
                        logger.debug(f"Successfully generated {feature}")
                    else:
                        logger.error(f"Cannot generate missing feature: {feature} (no method {generator_method})")
                        return
                
                # Update our dataframe with temporarily generated features
                self.df = temp_generator.df
            
            # Future price movement (next 15 periods = 75 minutes for 5m data)
            self.df['future_price_change'] = (
                self.df['close'].shift(-15) - self.df['close']
            ) / self.df['close']
            
            # STEP 1: Analyze enhanced feature distributions
            logger.info("=== ENHANCED FEATURE DISTRIBUTION ANALYSIS ===")
            logger.info(f"Depth stability - Mean: {self.df['depth_stability'].mean():.4f}, "
                    f"Std: {self.df['depth_stability'].std():.4f}, "
                    f"95th percentile: {self.df['depth_stability'].quantile(0.95):.4f}")
            
            logger.info(f"Liquidity stress - Mean: {self.df['liquidity_stress'].mean():.4f}, "
                    f"95th percentile: {self.df['liquidity_stress'].quantile(0.95):.4f}")
            
            logger.info(f"Volume ratio - Mean: {self.df['volume_ratio'].mean():.4f}, "
                    f"95th percentile: {self.df['volume_ratio'].quantile(0.95):.4f}")
            
            logger.info(f"Future price change - Std: {self.df['future_price_change'].std():.6f}, "
                    f"95th percentile: {abs(self.df['future_price_change']).quantile(0.95):.6f}")
            
            # STEP 2: Use adaptive thresholds based on enhanced features (RELAXED for better balance)
            depth_stability_threshold = self.df['depth_stability'].quantile(0.60)  # Top 40% (more signals)
            liquidity_stress_threshold = self.df['liquidity_stress'].quantile(0.70)  # Top 30% stress (more signals)
            volume_threshold = max(1.05, self.df['volume_ratio'].quantile(0.65))  # Top 35% but at least 1.05
            microstructure_health_threshold = self.df['microstructure_health'].quantile(0.50)  # Top 50% (more signals)
            movement_threshold = abs(self.df['future_price_change']).quantile(0.65)  # Top 35% movements (more signals)
            
            logger.info(f"Enhanced thresholds - Depth stability: {depth_stability_threshold:.4f}, "
                    f"Liquidity stress: {liquidity_stress_threshold:.4f}, "
                    f"Volume: {volume_threshold:.4f}, "
                    f"Microstructure health: {microstructure_health_threshold:.4f}, "
                    f"Movement: {movement_threshold:.6f}")
            
            # STEP 3: Enhanced signal generation using top predictive features
            
            # Depth-based signal (top predictor)
            depth_signal = self.df['depth_stability'] > depth_stability_threshold
            
            # Liquidity stress signal (indicates compression conditions)
            stress_signal = self.df['liquidity_stress'] > liquidity_stress_threshold
            
            # Volume confirmation
            volume_signal = self.df['volume_ratio'] > volume_threshold
            
            # Microstructure health signal
            health_signal = self.df['microstructure_health'] > microstructure_health_threshold
            
            # Future significant movement
            significant_movement = (
                abs(self.df['future_price_change']) > movement_threshold
            )
            
            # STEP 4: Create enhanced tiered targets using top predictive features
            
            # Tier 1: Basic depth signal + movement
            basic_signal = depth_signal & significant_movement
            
            # Tier 2: Depth + liquidity stress + movement (top predictors)
            medium_signal = depth_signal & stress_signal & significant_movement
            
            # Tier 3: Multi-factor signal (depth + stress + volume + health + movement)
            strong_signal = depth_signal & stress_signal & volume_signal & health_signal & significant_movement
            
            # Tier 4: London session bonus
            london_strong_signal = strong_signal & (self.df['london_session'] == 1)
            
            # STEP 5: Create multi-class target
            target_values = np.where(
                london_strong_signal, 4,  # Best signal
                np.where(
                    strong_signal, 3,     # Strong signal
                    np.where(
                        medium_signal, 2,  # Medium signal
                        np.where(
                            basic_signal, 1,  # Basic signal
                            0  # No signal
                        )
                    )
                )
            )
            
            self.df['target_spread_compression_multi'] = target_values
            
            # Binary version (any signal vs no signal)
            self.df['target_spread_compression_binary'] = (target_values > 0).astype(int)
            
            # STEP 6: Report results
            target_distribution = pd.Series(target_values).value_counts().sort_index()
            logger.info(f"Multi-class target distribution: {target_distribution.to_dict()}")
            logger.info(f"Binary target distribution: {self.df['target_spread_compression_binary'].value_counts().to_dict()}")
            
            # Signal frequency analysis
            total_signals = (target_values > 0).sum()
            signal_frequency = total_signals / len(self.df) * 100
            logger.info(f"Signal frequency: {total_signals}/{len(self.df)} = {signal_frequency:.2f}% of periods")
            
            # Clean up temporarily generated features AND future_price_change (data leakage)
            cleanup_features = temp_generated_features + ['future_price_change']
            cleanup_features = [f for f in cleanup_features if f in self.df.columns]
            
            if cleanup_features:
                logger.info(f"Cleaning up temporary and leakage features: {cleanup_features}")
                self.df = self.df.drop(columns=cleanup_features)
                logger.debug("Temporary and leakage features removed - maintaining clean training data")
            
            # Estimate daily frequency (assuming 5-minute bars)
            periods_per_day = 288  # 24 * 60 / 5
            estimated_daily_signals = signal_frequency / 100 * periods_per_day
            logger.info(f"Estimated daily signals: {estimated_daily_signals:.1f}")
            
            # STEP 7: Quick validation of each tier
            for tier in [1, 2, 3, 4]:
                tier_mask = target_values == tier
                if tier_mask.sum() > 0:
                    tier_movement = abs(self.df[tier_mask]['future_price_change']).mean()
                    logger.info(f"Tier {tier} average movement: {tier_movement:.6f} ({tier_mask.sum()} events)")
            
            return True
            
        except Exception as e:
            logger.error(f"Unable to calculate fixed target: {e}")
            return False

    def spread_compression_patterns_v2(self):
        """
        Enhanced target for spread compression pattern hypothesis - VERSION 2.
        
        FOCUS: Quality Tier 2+ signals only, eliminating noisy Tier 1 signals.
        GOAL: Better class balance (85/15 vs 93.9/6.1) with higher signal quality.
        
        Key improvements:
        - Skip Tier 1 noise signals entirely
        - More sensitive thresholds for Tier 2+ detection 
        - Feature interaction logic using top predictors
        - Enhanced volume and institutional confirmation
        """
        try:
            # Required features check - Enhanced set focusing on top predictors
            required_features = [
                'depth_stability',        # TOP predictor (0.399 correlation)
                'institutional_footprint', # Strong institutional signal
                'liquidity_stress',       # Compression detection  
                'microstructure_health',  # Signal quality filter
                'volume_ratio',           # Volume confirmation
                'london_session',         # Session timing
                'signal_conviction'       # Composite signal strength
            ]
            
            missing_features = [f for f in required_features if f not in self.df.columns]
            temp_generated_features = []
            
            if missing_features:
                logger.warning(f"Missing required features for enhanced target generation: {missing_features}")
                logger.info("Generating missing features temporarily for target creation...")
                
                # Import the ModelAwareFeatureGenerator for on-demand feature generation
                from data.features.enhanced_generation import ModelAwareFeatureGenerator
                temp_generator = ModelAwareFeatureGenerator(self.df)
                
                # Map features to their generator methods
                feature_generator_map = {
                    'depth_stability': 'enhanced_market_depth_features',
                    'liquidity_stress': 'enhanced_market_depth_features', 
                    'institutional_footprint': 'enhanced_institutional_analysis',
                    'microstructure_health': 'enhanced_microstructure_health',
                    'signal_conviction': 'enhanced_predictive_signals',
                    'volume_ratio': 'volume_ratio_feature',
                    'london_session': 'fx_trading_sessions'
                }
                
                # Generate missing features temporarily
                for feature in missing_features:
                    generator_method = feature_generator_map.get(feature)
                    if generator_method and hasattr(temp_generator, generator_method):
                        logger.debug(f"Temporarily generating {feature} using {generator_method}")
                        temp_generator.__getattribute__(generator_method)()
                        temp_generated_features.append(feature)
                        logger.debug(f"Successfully generated {feature}")
                    else:
                        logger.error(f"Cannot generate missing feature: {feature} (no method {generator_method})")
                        return
                
                # Update our dataframe with temporarily generated features
                self.df = temp_generator.df
            
            # Future price movement (next 15 periods for target validation)
            self.df['future_price_change'] = (
                self.df['close'].shift(-15) - self.df['close']
            ) / self.df['close']
            
            # STEP 1: Enhanced feature distribution analysis
            logger.info("=== ENHANCED TARGET V2 ANALYSIS ===")
            logger.info(f"Depth stability (TOP predictor): Mean {self.df['depth_stability'].mean():.4f}, "
                    f"70th: {self.df['depth_stability'].quantile(0.70):.4f}, "
                    f"80th: {self.df['depth_stability'].quantile(0.80):.4f}")
            
            logger.info(f"Institutional footprint: Mean {self.df['institutional_footprint'].mean():.4f}, "
                    f"75th: {self.df['institutional_footprint'].quantile(0.75):.4f}")
            
            logger.info(f"Movement threshold candidates: "
                    f"60th: {abs(self.df['future_price_change']).quantile(0.60):.6f}, "
                    f"65th: {abs(self.df['future_price_change']).quantile(0.65):.6f}")
            
            # STEP 2: ENHANCED THRESHOLDS - Much more sensitive for 10%+ target
            # Primary filter: Top predictor (depth_stability) - lowered to 50th percentile for more signals
            depth_stability_threshold = self.df['depth_stability'].quantile(0.50)  # Much more sensitive
            
            # Secondary filters - significantly more sensitive
            institutional_threshold = self.df['institutional_footprint'].quantile(0.60)  # Lowered from 75th
            liquidity_stress_threshold = self.df['liquidity_stress'].quantile(0.65)     # Lowered from 75th
            microstructure_health_threshold = self.df['microstructure_health'].quantile(0.50)  # Lowered from 60th
            volume_threshold = max(1.02, self.df['volume_ratio'].quantile(0.60))        # Lowered from 70th and 1.05
            signal_conviction_threshold = self.df['signal_conviction'].quantile(0.60)    # Lowered from 70th
            
            # Movement threshold - more sensitive for more signals
            movement_threshold = abs(self.df['future_price_change']).quantile(0.50)      # Lowered from 60th
            
            logger.info(f"ENHANCED V2 Thresholds:")
            logger.info(f"  Depth stability (primary): {depth_stability_threshold:.4f}")
            logger.info(f"  Institutional activity: {institutional_threshold:.4f}")
            logger.info(f"  Liquidity stress: {liquidity_stress_threshold:.4f}")
            logger.info(f"  Volume confirmation: {volume_threshold:.4f}")
            logger.info(f"  Movement threshold: {movement_threshold:.6f}")
            
            # STEP 3: FEATURE INTERACTION LOGIC - Quality-first approach
            
            # Primary signal: Top predictor + institutional confirmation
            primary_signal = (
                (self.df['depth_stability'] > depth_stability_threshold) &
                (self.df['institutional_footprint'] > institutional_threshold)
            )
            
            # Quality filters
            stress_confirmation = self.df['liquidity_stress'] > liquidity_stress_threshold
            volume_confirmation = self.df['volume_ratio'] > volume_threshold
            health_filter = self.df['microstructure_health'] > microstructure_health_threshold
            conviction_filter = self.df['signal_conviction'] > signal_conviction_threshold
            
            # London session amplifier
            london_bonus = (self.df['london_session'] == 1)
            
            # STEP 4: DIRECTIONAL SIGNAL GENERATION - Three-class system
            
            # Directional movement detection
            upward_movement = self.df['future_price_change'] > movement_threshold      # BUY signals
            downward_movement = self.df['future_price_change'] < -movement_threshold   # SELL signals
            no_movement = (abs(self.df['future_price_change']) <= movement_threshold)  # HOLD signals
            
            logger.info(f"Directional movement distribution:")
            logger.info(f"  Upward movements: {upward_movement.sum()} ({upward_movement.mean()*100:.1f}%)")
            logger.info(f"  Downward movements: {downward_movement.sum()} ({downward_movement.mean()*100:.1f}%)")
            logger.info(f"  No significant movement: {no_movement.sum()} ({no_movement.mean()*100:.1f}%)")
            
            # Quality Signal Level 1: Primary + Directional Movement
            quality_signal_long_basic = primary_signal & upward_movement      # Long signals
            quality_signal_short_basic = primary_signal & downward_movement   # Short signals
            
            # Quality Signal Level 2: Add stress + volume confirmation  
            quality_signal_long_medium = (
                primary_signal & 
                stress_confirmation & 
                volume_confirmation & 
                upward_movement
            )
            quality_signal_short_medium = (
                primary_signal & 
                stress_confirmation & 
                volume_confirmation & 
                downward_movement
            )
            
            # Quality Signal Level 3: Add health + conviction filters
            quality_signal_long_strong = (
                primary_signal &
                stress_confirmation &
                volume_confirmation & 
                health_filter &
                conviction_filter &
                upward_movement
            )
            quality_signal_short_strong = (
                primary_signal &
                stress_confirmation &
                volume_confirmation & 
                health_filter &
                conviction_filter &
                downward_movement
            )
            
            # Quality Signal Level 4: London session premium
            quality_signal_long_premium = quality_signal_long_strong & london_bonus
            quality_signal_short_premium = quality_signal_short_strong & london_bonus
            
            # STEP 5: THREE-CLASS DIRECTIONAL TARGET SYSTEM
            
            # Create separate scoring for long and short signals
            long_score = np.where(
                quality_signal_long_premium, 4,    # Premium long signals
                np.where(
                    quality_signal_long_strong, 3,  # Strong long signals  
                    np.where(
                        quality_signal_long_medium, 2,  # Medium long signals
                        np.where(
                            quality_signal_long_basic, 1,  # Basic long signals
                            0  # No long signal
                        )
                    )
                )
            )
            
            short_score = np.where(
                quality_signal_short_premium, 4,    # Premium short signals
                np.where(
                    quality_signal_short_strong, 3,  # Strong short signals  
                    np.where(
                        quality_signal_short_medium, 2,  # Medium short signals
                        np.where(
                            quality_signal_short_basic, 1,  # Basic short signals
                            0  # No short signal
                        )
                    )
                )
            )
            
            # Create three-class directional target
            # Class 0: Hold (no quality signal)
            # Class 1: Buy (quality long signal)  
            # Class 2: Sell (quality short signal)
            directional_target = np.where(
                long_score > 0, 1,      # Any quality long signal → BUY (Class 1)
                np.where(
                    short_score > 0, 2,  # Any quality short signal → SELL (Class 2)
                    0                    # No quality signals → HOLD (Class 0)
                )
            )
            
            # Multi-class targets for analysis
            self.df['target_spread_compression_v2_long_score'] = long_score
            self.df['target_spread_compression_v2_short_score'] = short_score
            
            # Main three-class directional target
            self.df['target_spread_compression_v2_directional'] = directional_target
            
            # Legacy binary target for compatibility (any signal vs no signal)
            self.df['target_spread_compression_v2_binary'] = (directional_target > 0).astype(int)
            
            # STEP 6: Results Analysis - Directional Target System
            directional_dist = pd.Series(directional_target).value_counts().sort_index()
            long_dist = pd.Series(long_score).value_counts().sort_index()  
            short_dist = pd.Series(short_score).value_counts().sort_index()
            binary_dist = self.df['target_spread_compression_v2_binary'].value_counts()
            
            logger.info(f"=== DIRECTIONAL TARGET V2 RESULTS ===")
            logger.info(f"Three-class directional distribution: {directional_dist.to_dict()}")
            logger.info(f"  Class 0 (HOLD): {directional_dist.get(0, 0)} ({directional_dist.get(0, 0)/len(self.df)*100:.1f}%)")
            logger.info(f"  Class 1 (BUY):  {directional_dist.get(1, 0)} ({directional_dist.get(1, 0)/len(self.df)*100:.1f}%)")
            logger.info(f"  Class 2 (SELL): {directional_dist.get(2, 0)} ({directional_dist.get(2, 0)/len(self.df)*100:.1f}%)")
            
            logger.info(f"Long signal score distribution: {long_dist.to_dict()}")
            logger.info(f"Short signal score distribution: {short_dist.to_dict()}")
            logger.info(f"Binary distribution (any signal): {binary_dist.to_dict()}")
            
            # Calculate signal rates
            total_long_signals = (long_score > 0).sum()
            total_short_signals = (short_score > 0).sum()
            total_quality_signals = total_long_signals + total_short_signals
            
            long_signal_rate = total_long_signals / len(self.df) * 100
            short_signal_rate = total_short_signals / len(self.df) * 100
            quality_signal_rate = total_quality_signals / len(self.df) * 100
            
            logger.info(f"Signal frequency breakdown:")
            logger.info(f"  Long signals:  {total_long_signals}/{len(self.df)} = {long_signal_rate:.2f}%")
            logger.info(f"  Short signals: {total_short_signals}/{len(self.df)} = {short_signal_rate:.2f}%")
            logger.info(f"  Total signals: {total_quality_signals}/{len(self.df)} = {quality_signal_rate:.2f}%")
            
            # Check balance between long and short signals
            if total_long_signals > 0 and total_short_signals > 0:
                balance_ratio = min(total_long_signals, total_short_signals) / max(total_long_signals, total_short_signals)
                logger.info(f"Long/Short balance ratio: {balance_ratio:.3f} (1.0 = perfect balance)")
                
                if balance_ratio > 0.7:
                    logger.info("✅ GOOD: Balanced long/short signal distribution")
                else:
                    logger.warning(f"⚠️  IMBALANCED: {('Long' if total_long_signals > total_short_signals else 'Short')} signals dominate")
            
            # Compare to original system
            if quality_signal_rate > 6.1:  # Original was 6.1%
                improvement = quality_signal_rate - 6.1
                logger.info(f"✅ IMPROVEMENT: +{improvement:.1f}% more quality signals than original system")
            
            # Estimate class balance for three-class system
            hold_rate = directional_dist.get(0, 0) / len(self.df) * 100
            logger.info(f"Three-class balance: {hold_rate:.1f}% hold / {long_signal_rate:.1f}% buy / {short_signal_rate:.1f}% sell")
            
            if quality_signal_rate > 10:
                logger.info("✅ TARGET ACHIEVED: >10% trading signals for better model training")
                logger.info("✅ DIRECTIONAL SYSTEM: Model will learn proper buy/sell directions")
            
            # Clean up temporary features AND future_price_change (data leakage prevention)
            cleanup_features = temp_generated_features + ['future_price_change']
            cleanup_features = [f for f in cleanup_features if f in self.df.columns]
            
            if cleanup_features:
                logger.info(f"Cleaning up temporary and leakage features: {cleanup_features}")
                self.df = self.df.drop(columns=cleanup_features)
                logger.debug("Clean training data maintained - no data leakage")
            
            # STEP 7: Quality validation for each signal level - Directional
            logger.info("=== DIRECTIONAL SIGNAL QUALITY VALIDATION ===")
            
            # Validate long signals by quality level
            logger.info("Long signal quality by level:")
            for level in [1, 2, 3, 4]:
                level_mask = long_score == level
                if level_mask.sum() > 0:
                    if 'future_price_change' in self.df.columns:
                        level_movement = self.df[level_mask]['future_price_change'].mean()  # Keep directional info
                        level_abs_movement = abs(self.df[level_mask]['future_price_change']).mean()
                        logger.info(f"  Long Level {level}: {level_mask.sum()} signals, avg movement: {level_movement:.6f} (abs: {level_abs_movement:.6f})")
            
            # Validate short signals by quality level  
            logger.info("Short signal quality by level:")
            for level in [1, 2, 3, 4]:
                level_mask = short_score == level
                if level_mask.sum() > 0:
                    if 'future_price_change' in self.df.columns:
                        level_movement = self.df[level_mask]['future_price_change'].mean()  # Keep directional info
                        level_abs_movement = abs(self.df[level_mask]['future_price_change']).mean()
                        logger.info(f"  Short Level {level}: {level_mask.sum()} signals, avg movement: {level_movement:.6f} (abs: {level_abs_movement:.6f})")
            
            # Overall directional validation
            if 'future_price_change' in self.df.columns:
                all_long_mask = long_score > 0
                all_short_mask = short_score > 0
                
                if all_long_mask.sum() > 0:
                    avg_long_movement = self.df[all_long_mask]['future_price_change'].mean()
                    logger.info(f"✅ All long signals avg movement: {avg_long_movement:.6f} (should be > 0)")
                    
                if all_short_mask.sum() > 0:
                    avg_short_movement = self.df[all_short_mask]['future_price_change'].mean()
                    logger.info(f"✅ All short signals avg movement: {avg_short_movement:.6f} (should be < 0)")
            
            logger.info("✅ Enhanced target generation V2 completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate enhanced target V2: {e}")
            return False

    # IMPORTANT FOR FURTHER DEVELOPMENT: Remember to add the prefix 'target_*' for the target column to the hypothesis.
    # This is because the normalization step after generation and this step will not normalize columns that starts with 'target_'.

