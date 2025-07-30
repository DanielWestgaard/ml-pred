import numpy as np
import pandas as pd
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
    
    def _generate_features_with_dependencies(self, required_features, feature_dependency_map):
        """
        Reusable method to generate required features and their dependencies.
        
        Args:
            required_features: List of feature names that are required
            feature_dependency_map: Dict mapping feature_name -> (generator_method, [dependencies])
                                   e.g., {'volume_ratio': ('volume_ratio_feature', ['volume_ma_20'])}
        
        Returns:
            tuple: (successfully_generated_features, all_added_features, temp_generator_df)
        """
        from data.features.enhanced_generation import ModelAwareFeatureGenerator
        
        missing_features = [f for f in required_features if f not in self.df.columns]
        if not missing_features:
            return [], [], self.df
            
        logger.warning(f"Missing required features for enhanced target generation: {missing_features}")
        logger.info("Generating missing features temporarily for target creation...")
        
        temp_generator = ModelAwareFeatureGenerator(self.df)
        
        # Track original columns to identify all added features
        original_columns = set(temp_generator.df.columns)
        
        # Ensure datetime index for session features
        if 'date' in temp_generator.df.columns:
            temp_generator.df.set_index('date', inplace=True)
            temp_generator.df.index = pd.to_datetime(temp_generator.df.index)
        
        # Build complete dependency graph including transitive dependencies
        all_needed_features = set(missing_features)
        
        def add_dependencies_recursively(feature_name):
            if feature_name in feature_dependency_map:
                _, dependencies = feature_dependency_map[feature_name]
                if dependencies:
                    for dep in dependencies:
                        if dep not in temp_generator.df.columns:
                            all_needed_features.add(dep)
                            add_dependencies_recursively(dep)
        
        for feature in missing_features:
            add_dependencies_recursively(feature)
        
        # Topologically sort features based on dependencies
        def topological_sort():
            visited = set()
            temp_visited = set()
            result = []
            
            def visit(feature):
                if feature in temp_visited:
                    logger.warning(f"Circular dependency detected involving {feature}")
                    return
                if feature in visited:
                    return
                    
                temp_visited.add(feature)
                
                if feature in feature_dependency_map:
                    _, dependencies = feature_dependency_map[feature]
                    if dependencies:
                        for dep in dependencies:
                            if dep in all_needed_features:
                                visit(dep)
                
                temp_visited.remove(feature)
                visited.add(feature)
                result.append(feature)
            
            for feature in all_needed_features:
                if feature not in visited:
                    visit(feature)
            
            return result
        
        generation_order = topological_sort()
        successfully_generated = []
        
        # Generate features in dependency order
        for feature_name in generation_order:
            if feature_name not in temp_generator.df.columns:
                if feature_name in feature_dependency_map:
                    generator_method, _ = feature_dependency_map[feature_name]
                    if generator_method and hasattr(temp_generator, generator_method):
                        try:
                            logger.debug(f"Temporarily generating {feature_name} using {generator_method}")
                            temp_generator.__getattribute__(generator_method)()
                            if feature_name in temp_generator.df.columns:
                                successfully_generated.append(feature_name)
                                logger.debug(f"Successfully generated {feature_name}")
                            else:
                                logger.warning(f"Feature {feature_name} not found after generation")
                        except Exception as e:
                            logger.warning(f"Failed to generate {feature_name}: {e}")
                    else:
                        logger.warning(f"Cannot generate missing feature: {feature_name} (no method {generator_method})")
        
        # Identify all features that were added during the process
        final_columns = set(temp_generator.df.columns)
        all_added_features = list(final_columns - original_columns)
        
        logger.info(f"Successfully generated {len(successfully_generated)} features: {successfully_generated}")
        logger.debug(f"Total features added during process: {len(all_added_features)} - {all_added_features}")
        
        return successfully_generated, all_added_features, temp_generator.df
    
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
            
            # REMOVED: Future price movement causes data leakage in target generation
            
            # STEP 1: Analyze enhanced feature distributions
            logger.info("=== ENHANCED FEATURE DISTRIBUTION ANALYSIS ===")
            logger.info(f"Depth stability - Mean: {self.df['depth_stability'].mean():.4f}, "
                    f"Std: {self.df['depth_stability'].std():.4f}, "
                    f"95th percentile: {self.df['depth_stability'].quantile(0.95):.4f}")
            
            logger.info(f"Liquidity stress - Mean: {self.df['liquidity_stress'].mean():.4f}, "
                    f"95th percentile: {self.df['liquidity_stress'].quantile(0.95):.4f}")
            
            logger.info(f"Volume ratio - Mean: {self.df['volume_ratio'].mean():.4f}, "
                    f"95th percentile: {self.df['volume_ratio'].quantile(0.95):.4f}")
            
            # STEP 2: Use adaptive thresholds based on enhanced features (RELAXED for better balance)
            depth_stability_threshold = self.df['depth_stability'].quantile(0.60)  # Top 40% (more signals)
            liquidity_stress_threshold = self.df['liquidity_stress'].quantile(0.70)  # Top 30% stress (more signals)
            volume_threshold = max(1.05, self.df['volume_ratio'].quantile(0.65))  # Top 35% but at least 1.05
            microstructure_health_threshold = self.df['microstructure_health'].quantile(0.50)  # Top 50% (more signals)
            
            # Use price volatility for movement threshold instead of future data
            price_volatility = self.df['close'].pct_change().rolling(window=100).std()
            movement_threshold = price_volatility.quantile(0.75)  # Top 25% volatility periods
            
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
            
            # Use current volatility for movement detection (no future data)
            current_volatility = self.df['close'].pct_change().rolling(window=5).std()
            significant_movement = current_volatility > movement_threshold
            
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
            
            # Clean up temporarily generated features (no future data was created)
            cleanup_features = temp_generated_features
            cleanup_features = [f for f in cleanup_features if f in self.df.columns]
            
            if cleanup_features:
                logger.info(f"Cleaning up temporary features: {cleanup_features}")
                self.df = self.df.drop(columns=cleanup_features)
                logger.debug("Temporary features cleaned up")
            
            # Estimate daily frequency (assuming 5-minute bars)
            periods_per_day = 288  # 24 * 60 / 5
            estimated_daily_signals = signal_frequency / 100 * periods_per_day
            logger.info(f"Estimated daily signals: {estimated_daily_signals:.1f}")
            
            # STEP 7: Signal tier distribution (no future validation to prevent data leakage)
            for tier in [1, 2, 3, 4]:
                tier_count = (target_values == tier).sum()
                if tier_count > 0:
                    tier_pct = tier_count / len(self.df) * 100
                    logger.info(f"Tier {tier}: {tier_count} events ({tier_pct:.3f}%)")
            
            return True
            
        except Exception as e:
            logger.error(f"Unable to calculate fixed target: {e}")
            return False

    def spread_compression_patterns_v3(self):
        """
        Enhanced target for spread compression - VERSION 3 with BETTER CLASS BALANCE.
        
        FOCUS: Achieve 15-20% signal rate (80-85% HOLD, 10-15% BUY/SELL) for better model training.
        STRATEGY: More aggressive thresholds and feature interactions to generate more signals.
        
        Key improvements over V2:
        - Ultra-aggressive thresholds for better balance
        - Multiple signal generation pathways  
        - Advanced feature combinations
        - Balanced directional signals
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
            
            # Define feature dependency mapping for this hypothesis
            feature_dependency_map = {
                'volume_ma_20': ('volume_ma_20_feature', []),
                'volume_ratio': ('volume_ratio_feature', ['volume_ma_20']),
                'depth_stability': ('enhanced_market_depth_features', []),
                'liquidity_stress': ('enhanced_market_depth_features', []),
                'institutional_footprint': ('institutional_footprint_detection', ['volume_ratio']),
                'microstructure_health': ('enhanced_microstructure_health', []),
                'signal_conviction': ('enhanced_predictive_signals', []),
                'london_session': ('fx_trading_sessions', [])
            }
            
            # Use reusable dependency resolution method from V2
            successfully_generated, all_added_features, updated_df = self._generate_features_with_dependencies(
                required_features, feature_dependency_map
            )
            
            # Update dataframe and track ALL features added for proper cleanup
            self.df = updated_df
            temp_generated_features = all_added_features
            
            logger.debug(f"V3: Successfully generated {len(successfully_generated)} features: {successfully_generated}")
            logger.debug(f"V3: Total features added for cleanup: {len(all_added_features)}")
            
            # Check if we have the minimum required features for target generation
            critical_features = ['depth_stability', 'institutional_footprint', 'volume_ratio', 'microstructure_health']
            missing_critical = [f for f in critical_features if f not in self.df.columns]
            
            if missing_critical:
                logger.error(f"Cannot proceed with target generation - missing critical features: {missing_critical}")
                return False
            
            # STEP 1: ULTRA-AGGRESSIVE THRESHOLDS for 15-20% signal rate
            logger.info("=== ENHANCED TARGET V3: BALANCED CLASS DISTRIBUTION ===")
            
            # Primary thresholds - VERY aggressive for more signals
            depth_stability_threshold = self.df['depth_stability'].quantile(0.20)      # Top 80% (very inclusive)
            institutional_threshold = self.df['institutional_footprint'].quantile(0.30)  # Top 70%
            liquidity_stress_threshold = self.df['liquidity_stress'].quantile(0.35)     # Top 65%
            microstructure_health_threshold = self.df['microstructure_health'].quantile(0.25)  # Top 75%
            volume_threshold = max(1.01, self.df['volume_ratio'].quantile(0.35))        # Top 65%, minimum 1.01
            signal_conviction_threshold = self.df['signal_conviction'].quantile(0.35)    # Top 65%
            
            logger.info(f"V3 ULTRA-AGGRESSIVE Thresholds:")
            logger.info(f"  Depth stability: {depth_stability_threshold:.4f} (80th percentile)")
            logger.info(f"  Institutional activity: {institutional_threshold:.4f} (70th percentile)")
            logger.info(f"  Liquidity stress: {liquidity_stress_threshold:.4f} (65th percentile)")
            logger.info(f"  Volume confirmation: {volume_threshold:.4f} (65th percentile)")
            
            # STEP 2: MULTIPLE SIGNAL PATHWAYS for better balance
            
            # Pathway 1: Core compression signal
            core_signal = (
                (self.df['depth_stability'] > depth_stability_threshold) |  # OR condition for more signals
                (self.df['liquidity_stress'] > liquidity_stress_threshold)
            )
            
            # Pathway 2: Volume-driven signal
            volume_signal = (
                (self.df['volume_ratio'] > volume_threshold) &
                (self.df['institutional_footprint'] > institutional_threshold)
            )
            
            # Pathway 3: Health-based signal
            health_signal = (
                (self.df['microstructure_health'] > microstructure_health_threshold) &
                (self.df['signal_conviction'] > signal_conviction_threshold)
            )
            
            # Pathway 4: Session timing bonus
            london_boost = (self.df['london_session'] == 1)
            
            # Combined primary signal (OR logic for inclusivity)
            primary_signal = core_signal | volume_signal | health_signal
            
            # STEP 3: DIRECTIONAL BIAS with more balanced approach
            
            # More sensitive directional detection
            price_momentum = self.df['close'].pct_change(periods=3)  # Shorter period for more signals
            volume_momentum = self.df['volume_ratio'] - 1.0
            
            # Lower thresholds for directional detection
            bullish_bias = (
                (price_momentum > -0.0001) &  # Almost any positive or flat momentum
                (volume_momentum > -0.1)     # Volume not significantly below average
            )
            
            bearish_bias = (
                (price_momentum < 0.0001) &   # Almost any negative or flat momentum  
                (volume_momentum > -0.1)     # Volume not significantly below average
            )
            
            # STEP 4: BALANCED SIGNAL GENERATION
            
            # Quality Level 1: Basic signals with directional bias
            basic_long_signal = primary_signal & bullish_bias
            basic_short_signal = primary_signal & bearish_bias
            
            # Quality Level 2: Enhanced signals
            enhanced_long_signal = (
                (core_signal | volume_signal) & 
                bullish_bias &
                (self.df['depth_stability'] > depth_stability_threshold)
            )
            enhanced_short_signal = (
                (core_signal | volume_signal) & 
                bearish_bias &
                (self.df['depth_stability'] > depth_stability_threshold)
            )
            
            # Quality Level 3: Premium signals (with session bonus)
            premium_long_signal = enhanced_long_signal & london_boost
            premium_short_signal = enhanced_short_signal & london_boost
            
            # STEP 5: CREATE BALANCED THREE-CLASS TARGET
            
            # Long signal strength
            long_score = np.where(
                premium_long_signal, 3,      # Premium long signals
                np.where(
                    enhanced_long_signal, 2,  # Enhanced long signals
                    np.where(
                        basic_long_signal, 1,  # Basic long signals
                        0  # No long signal
                    )
                )
            )
            
            # Short signal strength
            short_score = np.where(
                premium_short_signal, 3,      # Premium short signals
                np.where(
                    enhanced_short_signal, 2,  # Enhanced short signals
                    np.where(
                        basic_short_signal, 1,  # Basic short signals
                        0  # No short signal
                    )
                )
            )
            
            # Create balanced three-class directional target
            directional_target = np.where(
                long_score > 0, 1,      # Any long signal → BUY (Class 1)
                np.where(
                    short_score > 0, 2,  # Any short signal → SELL (Class 2)
                    0                    # No signals → HOLD (Class 0)
                )
            )
            
            # Multi-class targets for analysis
            self.df['target_spread_compression_v3_long_score'] = long_score
            self.df['target_spread_compression_v3_short_score'] = short_score
            self.df['target_spread_compression_v3_directional'] = directional_target
            self.df['target_spread_compression_v3_binary'] = (directional_target > 0).astype(int)
            
            # STEP 6: RESULTS ANALYSIS
            directional_dist = pd.Series(directional_target).value_counts().sort_index()
            
            logger.info(f"=== V3 BALANCED TARGET RESULTS ===")
            logger.info(f"Three-class distribution: {directional_dist.to_dict()}")
            
            total_samples = len(self.df)
            hold_pct = directional_dist.get(0, 0) / total_samples * 100
            buy_pct = directional_dist.get(1, 0) / total_samples * 100
            sell_pct = directional_dist.get(2, 0) / total_samples * 100
            
            logger.info(f"  Class 0 (HOLD): {directional_dist.get(0, 0)} ({hold_pct:.1f}%)")
            logger.info(f"  Class 1 (BUY):  {directional_dist.get(1, 0)} ({buy_pct:.1f}%)")
            logger.info(f"  Class 2 (SELL): {directional_dist.get(2, 0)} ({sell_pct:.1f}%)")
            
            # Check if we achieved better balance
            signal_pct = buy_pct + sell_pct
            if signal_pct > 15:
                logger.info(f"✅ SUCCESS: {signal_pct:.1f}% signal rate achieved (target: >15%)")
            else:
                logger.warning(f"⚠️  Signal rate {signal_pct:.1f}% still below 15% target")
            
            # Check directional balance
            if buy_pct > 0 and sell_pct > 0:
                balance_ratio = min(buy_pct, sell_pct) / max(buy_pct, sell_pct)
                logger.info(f"Directional balance ratio: {balance_ratio:.3f} (1.0 = perfect)")
                
                if balance_ratio > 0.5:
                    logger.info("✅ GOOD: Reasonable directional balance")
                else:
                    logger.warning(f"⚠️  IMBALANCED: {('Long' if buy_pct > sell_pct else 'Short')} signals dominate")
            
            # Clean up temporary features
            cleanup_features = temp_generated_features
            cleanup_features = [f for f in cleanup_features if f in self.df.columns]
            
            if cleanup_features:
                logger.info(f"Cleaning up temporary features: {cleanup_features}")
                self.df = self.df.drop(columns=cleanup_features)
                logger.debug("Temporary features cleaned up")
            
            logger.info("✅ Enhanced target generation V3 completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate enhanced target V3: {e}")
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
            
            # Define feature dependency mapping for this hypothesis
            feature_dependency_map = {
                'volume_ma_20': ('volume_ma_20_feature', []),
                'volume_ratio': ('volume_ratio_feature', ['volume_ma_20']),
                'depth_stability': ('enhanced_market_depth_features', []),
                'liquidity_stress': ('enhanced_market_depth_features', []),
                'institutional_footprint': ('institutional_footprint_detection', ['volume_ratio']),
                'microstructure_health': ('enhanced_microstructure_health', []),
                'signal_conviction': ('enhanced_predictive_signals', []),
                'london_session': ('fx_trading_sessions', [])
            }
            
            # Use reusable dependency resolution method
            _, all_added_features, updated_df = self._generate_features_with_dependencies(
                required_features, feature_dependency_map
            )
            
            # Update dataframe and track ALL features added (not just successfully generated)
            self.df = updated_df
            temp_generated_features = all_added_features  # Track ALL added features for cleanup
            
            # Check if we have the minimum required features for target generation
            critical_features = ['depth_stability', 'institutional_footprint', 'volume_ratio', 'microstructure_health']
            missing_critical = [f for f in critical_features if f not in self.df.columns]
            
            if missing_critical:
                logger.error(f"Cannot proceed with target generation - missing critical features: {missing_critical}")
                return False
            
            # REMOVED: Future price movement causes data leakage
            # Target generation should be based only on current market conditions
            
            # STEP 1: Enhanced feature distribution analysis
            logger.info("=== ENHANCED TARGET V2 ANALYSIS ===")
            logger.info(f"Depth stability (TOP predictor): Mean {self.df['depth_stability'].mean():.4f}, "
                    f"70th: {self.df['depth_stability'].quantile(0.70):.4f}, "
                    f"80th: {self.df['depth_stability'].quantile(0.80):.4f}")
            
            logger.info(f"Institutional footprint: Mean {self.df['institutional_footprint'].mean():.4f}, "
                    f"75th: {self.df['institutional_footprint'].quantile(0.75):.4f}")
            
            # Use price volatility statistics instead of future data
            price_returns = self.df['close'].pct_change()
            logger.info(f"Price volatility thresholds: "
                    f"75th percentile: {abs(price_returns).quantile(0.75):.6f}, "
                    f"90th percentile: {abs(price_returns).quantile(0.90):.6f}")
            
            # STEP 2: ULTRA-AGGRESSIVE THRESHOLDS - Target 15-20% signals for better balance
            # Primary filter: Top predictor (depth_stability) - VERY sensitive
            depth_stability_threshold = self.df['depth_stability'].quantile(0.30)  # Ultra-sensitive: top 70%
            
            # Secondary filters - ULTRA sensitive  
            institutional_threshold = self.df['institutional_footprint'].quantile(0.40)  # Top 60%
            liquidity_stress_threshold = self.df['liquidity_stress'].quantile(0.45)     # Top 55%
            microstructure_health_threshold = self.df['microstructure_health'].quantile(0.35)  # Top 65%
            volume_threshold = max(1.01, self.df['volume_ratio'].quantile(0.45))        # Top 55%, minimum 1.01
            signal_conviction_threshold = self.df['signal_conviction'].quantile(0.45)    # Top 55%
            
            # Movement threshold - use statistical thresholds instead of future data
            # Base movement threshold on historical price volatility
            price_volatility = self.df['close'].pct_change().rolling(window=100).std()
            movement_threshold = price_volatility.quantile(0.75)  # Top 25% volatility periods
            
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
            
            # STEP 4: DIRECTIONAL SIGNAL GENERATION - Based on current market conditions only
            
            # Use current price momentum and volume patterns for direction detection
            # This removes data leakage while maintaining directional logic
            price_momentum = self.df['close'].pct_change(periods=5)  # 5-period momentum
            volume_momentum = self.df['volume_ratio'] - 1.0  # Volume above baseline
            
            # Directional bias detection using current market structure
            bullish_bias = (
                (price_momentum > 0) & 
                (volume_momentum > 0) &
                (self.df['depth_stability'] > depth_stability_threshold)
            )
            
            bearish_bias = (
                (price_momentum < 0) & 
                (volume_momentum > 0) &
                (self.df['depth_stability'] > depth_stability_threshold)
            )
            
            neutral_bias = ~(bullish_bias | bearish_bias)
            
            logger.info(f"Market bias distribution (no future data):")
            logger.info(f"  Bullish bias: {bullish_bias.sum()} ({bullish_bias.mean()*100:.1f}%)")
            logger.info(f"  Bearish bias: {bearish_bias.sum()} ({bearish_bias.mean()*100:.1f}%)")
            logger.info(f"  Neutral bias: {neutral_bias.sum()} ({neutral_bias.mean()*100:.1f}%)")
            
            # Quality Signal Level 1: Primary + Market Bias
            quality_signal_long_basic = primary_signal & bullish_bias      # Long signals
            quality_signal_short_basic = primary_signal & bearish_bias     # Short signals
            
            # Quality Signal Level 2: Add stress + volume confirmation  
            quality_signal_long_medium = (
                primary_signal & 
                stress_confirmation & 
                volume_confirmation & 
                bullish_bias
            )
            quality_signal_short_medium = (
                primary_signal & 
                stress_confirmation & 
                volume_confirmation & 
                bearish_bias
            )
            
            # Quality Signal Level 3: Add health + conviction filters
            quality_signal_long_strong = (
                primary_signal &
                stress_confirmation &
                volume_confirmation & 
                health_filter &
                conviction_filter &
                bullish_bias
            )
            quality_signal_short_strong = (
                primary_signal &
                stress_confirmation &
                volume_confirmation & 
                health_filter &
                conviction_filter &
                bearish_bias
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
            
            # Clean up temporary features only - no future data was generated
            cleanup_features = temp_generated_features
            cleanup_features = [f for f in cleanup_features if f in self.df.columns]
            
            if cleanup_features:
                logger.info(f"Cleaning up temporary features: {cleanup_features}")
                self.df = self.df.drop(columns=cleanup_features)
                logger.debug("Temporary features cleaned up")
            
            # STEP 7: Signal level distribution analysis (no future data validation)
            logger.info("=== SIGNAL DISTRIBUTION ANALYSIS ===")
            
            # Count signals by quality level
            logger.info("Long signal distribution by quality level:")
            for level in [1, 2, 3, 4]:
                level_count = (long_score == level).sum()
                if level_count > 0:
                    level_pct = level_count / len(self.df) * 100
                    logger.info(f"  Long Level {level}: {level_count} signals ({level_pct:.3f}%)")
            
            logger.info("Short signal distribution by quality level:")
            for level in [1, 2, 3, 4]:
                level_count = (short_score == level).sum()
                if level_count > 0:
                    level_pct = level_count / len(self.df) * 100
                    logger.info(f"  Short Level {level}: {level_count} signals ({level_pct:.3f}%)")
            
            # Signal quality distribution
            total_long_signals = (long_score > 0).sum()
            total_short_signals = (short_score > 0).sum()
            
            if total_long_signals > 0:
                logger.info(f"✅ Total long signals: {total_long_signals} ({total_long_signals/len(self.df)*100:.2f}%)")
                    
            if total_short_signals > 0:
                logger.info(f"✅ Total short signals: {total_short_signals} ({total_short_signals/len(self.df)*100:.2f}%)")
            
            logger.info("✅ Enhanced target generation V2 completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate enhanced target V2: {e}")
            return False

    # IMPORTANT FOR FURTHER DEVELOPMENT: Remember to add the prefix 'target_*' for the target column to the hypothesis.
    # This is because the normalization step after generation and this step will not normalize columns that starts with 'target_'.
