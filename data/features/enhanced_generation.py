"""
Enhanced Model-Aware Feature Generator
=====================================

This module provides a robust feature generation system that:
- Automatically generates features based on model metadata
- Handles feature dependencies automatically
- Provides clear feedback about missing features
- Is easily extensible with new features
"""

import pandas as pd
import numpy as np
import yaml
from typing import Dict, List, Optional, Any, Set
from pathlib import Path

from .feature_registry import FeatureRegistry, FeatureDefinition, feature_generator, AutoRegisterMeta
from utils import data_utils
from utils.logging_utils import get_logger

logger = get_logger()

class ModelAwareFeatureGenerator(metaclass=AutoRegisterMeta):
    """
    Enhanced feature generator that can automatically generate features
    based on model requirements with dependency resolution
    """
    
    # Class-level registry
    _registry = FeatureRegistry()
    
    def __init__(self, dataset, model_metadata_path: str = None, selected_features: List[str] = None):
        """
        Initialize the enhanced feature generator
        
        Args:
            dataset: Input dataset (DataFrame or path)
            model_metadata_path: Optional path to model metadata for automatic feature selection
            selected_features: Optional list of specific features to generate
        """
        try:
            # Load and validate dataset
            self.df, self.original_df = data_utils.check_and_return_df(dataset)
            
            # Validate required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Check for sufficient data
            if len(self.df) < 200:
                logger.warning(f"Dataset contains only {len(self.df)} rows. Some features may not generate properly.")
            
            # Store configuration
            self.model_metadata_path = model_metadata_path
            self.selected_features = selected_features
            self._generated_features = set()
            
            logger.info(f"Enhanced FeatureGenerator initialized with {len(self.df)} rows")
            
        except Exception as e:
            logger.error(f"Error initializing Enhanced FeatureGenerator: {e}")
            raise
    
    @classmethod
    def get_available_features(cls) -> List[str]:
        """Get all features that can be generated"""
        return cls._registry.get_available_features()
    
    @classmethod
    def get_available_generators(cls) -> List[str]:
        """Get all available generator functions"""
        return cls._registry.get_available_generators()
    
    def generate_for_model(self, model_metadata_path: str = None) -> pd.DataFrame:
        """
        Generate features automatically based on model metadata
        
        Args:
            model_metadata_path: Path to model metadata file
            
        Returns:
            DataFrame with generated features
        """
        metadata_path = model_metadata_path or self.model_metadata_path
        
        if not metadata_path:
            raise ValueError("No model metadata path provided")
        
        # Load model metadata
        try:
            with open(metadata_path, 'r') as f:
                metadata = yaml.safe_load(f)
            
            required_features = metadata['dataset']['features']
            logger.info(f"Model requires {len(required_features)} features")
            
        except Exception as e:
            logger.error(f"Error loading model metadata: {e}")
            raise
        
        return self.generate_selected(required_features)
    
    def generate_selected(self, feature_names: List[str]) -> pd.DataFrame:
        """
        Generate specific features by name with automatic dependency resolution
        
        Args:
            feature_names: List of feature names to generate
            
        Returns:
            DataFrame with generated features
        """
        logger.info(f"Generating {len(feature_names)} selected features...")
        
        # **FEATURE EXPANSION**: Handle logical feature groups in config
        # Some config entries like "bid_ask_spread_features" refer to logical groups, not individual features
        expanded_features = []
        feature_group_mappings = {
            'bid_ask_spread_features': ['bid_ask_spread_bps', 'bid_ask_spread_relative'],
            'spread_tight_features': ['spread_tight', 'spread_wide', 'spread_avg'],
            'spread_wide_features': ['spread_tight', 'spread_wide', 'spread_avg'],  # Same as tight_features
            'spread_relative_features': ['spread_relative', 'spread_baseline_true', 'spread_recent_true', 'compression_ratio_true'],
            'multi_timeframe_compression_features': ['compression_ratio_micro', 'compression_ratio_short', 'compression_ratio_medium'],
            'advanced_volume_confirmation_features': ['volume_acceleration', 'volume_breakout', 'volume_spread_correlation'],
            'enhanced_signal_strength_calculator': ['signal_strength_v2']
        }
        
        for feature_name in feature_names:
            if feature_name in feature_group_mappings:
                expanded_features.extend(feature_group_mappings[feature_name])
                logger.debug(f"Expanded feature group '{feature_name}' to: {feature_group_mappings[feature_name]}")
            else:
                expanded_features.append(feature_name)
        
        # Update feature_names with expanded list
        feature_names = list(set(expanded_features))  # Remove duplicates
        logger.debug(f"Final expanded feature list: {feature_names}")
        
        # Validate feature availability
        available, unavailable = self._registry.validate_feature_availability(feature_names)
        
        if unavailable:
            # Check if unavailable features are basic columns
            basic_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_basic = [f for f in unavailable if f in basic_columns]
            missing_features = [f for f in unavailable if f not in basic_columns]
            
            if missing_basic:
                logger.info(f"Basic columns already available: {missing_basic}")
            
            if missing_features:
                # Check if missing features already exist in the data (e.g., close_og from processed data)
                existing_features = [f for f in missing_features if f in self.df.columns]
                truly_missing = [f for f in missing_features if f not in self.df.columns]
                
                if existing_features:
                    logger.debug(f"Features already exist in data and will be preserved: {existing_features}")
                    # Keep existing features in the list
                    feature_names = available + existing_features
                else:
                    feature_names = available
                    
                if truly_missing:
                    logger.debug(f"Features scheduled for later generation in transformation step: {truly_missing}")
                    logger.info(f"Note: Some features (like transformation-based features) are generated during transformation phase, not feature generation")
        
        # Get required generators
        generators, missing = self._registry.get_generators_for_features(feature_names)
        
        if missing:
            logger.warning(f"No generators found for features: {missing}")
        
        if not generators:
            logger.warning("No generators needed - returning original dataset")
            return self.df
        
        # Resolve dependencies and get execution order
        try:
            execution_order = self._registry.resolve_dependencies(generators)
            logger.debug(f"Execution order: {execution_order}")
        except ValueError as e:
            logger.error(f"Dependency resolution failed: {e}")
            raise
        
        # Execute generators in order
        failed_generators = []
        
        for generator_name in execution_order:
            try:
                if hasattr(self, generator_name):
                    method = getattr(self, generator_name)
                    logger.debug(f"Executing generator: {generator_name}")
                    method()
                    
                    # Track generated features
                    generator_def = self._registry.get_generator_info(generator_name)
                    if generator_def:
                        self._generated_features.update(generator_def.output_features)
                else:
                    logger.warning(f"Generator method '{generator_name}' not found")
                    failed_generators.append(generator_name)
                    
            except Exception as e:
                logger.error(f"Error in generator '{generator_name}': {e}")
                failed_generators.append(generator_name)
        
        # **CRITICAL FIX**: Filter to only keep requested features
        # This prevents feature bloat when generators produce multiple features
        original_columns = ['open', 'high', 'low', 'close', 'volume']  # Always keep these
        
        # Identify which columns to keep
        columns_to_keep = set(original_columns)
        columns_to_keep.update(f for f in feature_names if f in self.df.columns)
        
        # Also keep derived columns that are part of the dataset structure
        derived_columns = ['timestamp', 'date']  # Basic derived columns to preserve
        columns_to_keep.update(f for f in derived_columns if f in self.df.columns)
        
        # **TARGET COLUMNS**: Always preserve target columns (they start with 'target_')
        target_columns = [col for col in self.df.columns if col.startswith('target_')]
        columns_to_keep.update(target_columns)
        if target_columns:
            logger.debug(f"Preserving target columns: {target_columns}")
        
        # **FUTURE/LABEL COLUMNS**: Preserve common label columns  
        label_columns = ['future_price_change', 'close_og', 'mid_price']
        for col in label_columns:
            if col in self.df.columns:
                columns_to_keep.add(col)
                logger.debug(f"Preserving label column: {col}")
        
        # **SPREAD DATA FILTERING**: Only keep spread columns if explicitly requested
        # Raw spread data columns that come from data ingestion but may not be needed
        spread_columns = [
            'bid_open', 'ask_open', 'bid_high', 'ask_high', 'bid_low', 'ask_low', 
            'bid_close', 'ask_close', 'spread_open', 'spread_high', 'spread_low', 
            'spread_close', 'spread_tight', 'spread_wide', 'spread_avg'
        ]
        
        # Only keep spread columns if they were explicitly requested
        for col in spread_columns:
            if col in self.df.columns and col in feature_names:
                columns_to_keep.add(col)
                logger.debug(f"Keeping explicitly requested spread column: {col}")
            elif col in self.df.columns:
                logger.debug(f"Filtering out unrequested spread column: {col}")
        
        # Filter the dataframe to only requested features
        columns_before = set(self.df.columns)
        columns_to_drop = columns_before - columns_to_keep
        
        if columns_to_drop:
            logger.info(f"Filtering out {len(columns_to_drop)} unrequested features: {sorted(list(columns_to_drop))}")
            self.df = self.df[list(columns_to_keep)]
        
        # Report results
        actual_features = [f for f in feature_names if f in self.df.columns]
        logger.info(f"Successfully generated {len(actual_features)} requested features")
        
        # More accurate reporting - separate actual generator failures from missing individual generators
        actual_generator_failures = []
        missing_individual_generators = []
        
        for failed_gen in failed_generators:
            # Check if this failed generator actually had features that were generated by other generators
            if any(f in self.df.columns for f in feature_names if failed_gen in feature_names):
                missing_individual_generators.append(failed_gen)
            else:
                actual_generator_failures.append(failed_gen)
        
        if actual_generator_failures:
            logger.warning(f"Failed generators: {actual_generator_failures}")
        
        if missing_individual_generators:
            logger.debug(f"Missing individual generators (features may be generated by composite generators): {missing_individual_generators}")
        
        missing_expected = [f for f in feature_names if f not in self.df.columns and f not in ['open', 'high', 'low', 'close', 'volume']]
        if missing_expected:
            logger.warning(f"Expected features not generated: {missing_expected}")
        else:
            logger.info("All expected features successfully generated")
        
        return self.df
    
    def generate_all(self) -> pd.DataFrame:
        """Generate all available features"""
        all_generators = self._registry.get_available_generators()
        execution_order = self._registry.resolve_dependencies(all_generators)
        
        for generator_name in execution_order:
            try:
                if hasattr(self, generator_name):
                    method = getattr(self, generator_name)
                    method()
            except Exception as e:
                logger.error(f"Error in generator '{generator_name}': {e}")
        
        return self.df
    
    def _handle_infinite_values(self, feature_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle infinite values in generated features with robust error handling
        
        Args:
            feature_dict: Dictionary of feature name -> values
            
        Returns:
            Dictionary with infinite values properly handled
        """
        cleaned_features = {}
        inf_columns_found = []
        
        for feature_name, values in feature_dict.items():
            try:
                # Convert to pandas Series if it's not already
                if not isinstance(values, (pd.Series, pd.DataFrame)):
                    if hasattr(values, '__iter__') and not isinstance(values, str):
                        values = pd.Series(values, index=self.df.index)
                    else:
                        # Single value, broadcast to full length
                        values = pd.Series([values] * len(self.df), index=self.df.index)
                
                # Check for infinite values
                if hasattr(values, 'replace') and values.dtype.kind in 'fc':  # float or complex
                    # Count infinite values before cleaning
                    inf_mask = np.isinf(values)
                    if inf_mask.any():
                        inf_count = inf_mask.sum()
                        inf_columns_found.append((feature_name, inf_count))
                        
                        # Replace inf with NaN
                        values = values.replace([np.inf, -np.inf], np.nan)
                        
                        # Handle NaN values with forward fill, then backward fill, then zero
                        values = values.ffill().bfill().fillna(0)
                
                cleaned_features[feature_name] = values
                
            except Exception as e:
                logger.warning(f"Error handling infinite values in feature '{feature_name}': {e}")
                # Fallback: create zeros if we can't clean the feature
                cleaned_features[feature_name] = pd.Series(0, index=self.df.index)
        
        # Log results
        if inf_columns_found:
            total_inf = sum(count for _, count in inf_columns_found)
            logger.info(f"Cleaned {total_inf} infinite values across {len(inf_columns_found)} features: "
                       f"{[(name, count) for name, count in inf_columns_found]}")
        
        return cleaned_features
    
    def _add_features_efficiently(self, new_features_dict: Dict[str, Any], drop_columns: List[str] = None):
        """Add multiple features efficiently to avoid DataFrame fragmentation"""
        if new_features_dict:
            # Clean infinite values before adding to DataFrame
            cleaned_features = self._handle_infinite_values(new_features_dict)
            
            for name, values in cleaned_features.items():
                self.df[name] = values
        
        if drop_columns:
            existing_columns = [col for col in drop_columns if col in self.df.columns]
            if existing_columns:
                self.df.drop(columns=existing_columns, inplace=True)
    
    # ==========================================================================
    # FEATURE GENERATORS (with automatic registration)
    # ==========================================================================
    
    @feature_generator(
        output_features=['atr'],
        description="Average True Range - volatility indicator",
        category="volatility"
    )
    def average_true_range(self, period_atr: int = 14):
        """Calculate Average True Range"""
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']
        
        # True Range calculation
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period_atr).mean()
        
        self.df['atr'] = atr
    
    @feature_generator(
        output_features=['hour_sin', 'hour_cos'],
        description="Cyclical time of day features",
        category="temporal"
    )
    def time_of_day(self):
        """Generate cyclical time of day features"""
        hour = self.df.index.hour
        self.df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    
    @feature_generator(
        output_features=['hour_of_week_sin', 'hour_of_week_cos', 'is_weekend'],
        description="Cyclical time of week features",
        category="temporal"
    )
    def time_of_week(self):
        """Generate time of week features"""
        hour_of_week = self.df.index.dayofweek * 24 + self.df.index.hour
        self.df['hour_of_week_sin'] = np.sin(2 * np.pi * hour_of_week / (7 * 24))
        self.df['hour_of_week_cos'] = np.cos(2 * np.pi * hour_of_week / (7 * 24))
        self.df['is_weekend'] = (self.df.index.dayofweek >= 5).astype(int)
    
    @feature_generator(
        output_features=['month_progress_sin', 'month_progress_cos', 'month_period', 
                        'month_of_year_sin', 'month_of_year_cos'],
        description="Monthly temporal features",
        category="temporal"
    )
    def time_of_month(self):
        """Generate monthly temporal features"""
        # Month progress (0-1 through the month)
        month_progress = (self.df.index.day - 1) / self.df.index.days_in_month
        self.df['month_progress_sin'] = np.sin(2 * np.pi * month_progress)
        self.df['month_progress_cos'] = np.cos(2 * np.pi * month_progress)
        
        # Month of year
        month = self.df.index.month
        self.df['month_of_year_sin'] = np.sin(2 * np.pi * month / 12)
        self.df['month_of_year_cos'] = np.cos(2 * np.pi * month / 12)
        
        # Month period (categorical)
        def get_month_period(day, days_in_month):
            if day <= 10:
                return 'start_of_month'
            elif day >= days_in_month - 5:
                return 'end_of_month'
            else:
                return 'mid_month'
        
        self.df['month_period'] = [
            get_month_period(day, dim) 
            for day, dim in zip(self.df.index.day, self.df.index.days_in_month)
        ]
    
    @feature_generator(
        output_features=['day_of_week_sin', 'day_of_week_cos', 'is_monday', 'is_friday', 
                        'is_midweek', 'friday_afternoon', 'sunday_evening'],
        description="Day of week features",
        category="temporal"
    )
    def day_of_week(self):
        """Generate day of week features"""
        dow = self.df.index.dayofweek
        hour = self.df.index.hour
        
        # Cyclical day of week
        self.df['day_of_week_sin'] = np.sin(2 * np.pi * dow / 7)
        self.df['day_of_week_cos'] = np.cos(2 * np.pi * dow / 7)
        
        # Specific day flags
        self.df['is_monday'] = (dow == 0).astype(int)
        self.df['is_friday'] = (dow == 4).astype(int)
        self.df['is_midweek'] = dow.isin([1, 2, 3]).astype(int)
        
        # Special time periods
        self.df['friday_afternoon'] = ((dow == 4) & (hour >= 15)).astype(int)
        self.df['sunday_evening'] = ((dow == 6) & (hour >= 18)).astype(int)
    
    @feature_generator(
        output_features=['activity'],
        description="FX activity profile based on tick volume",
        category="volume"
    )
    def fx_activity_profile(self, lookback: int = 20):
        """Generate FX activity profile"""
        volume_ma = self.df['volume'].rolling(window=lookback).mean()
        self.df['activity'] = self.df['volume'] / volume_ma.where(volume_ma > 0, 1)
    
    @feature_generator(
        output_features=['asian_session', 'london_session', 'ny_session', 
                        'asian_london_overlap', 'london_ny_overlap', 'primary_session', 'session_intensity'],
        description="FX trading session features",
        category="sessions"
    )
    def fx_trading_sessions(self):
        """Generate FX trading session features"""
        hour = self.df.index.hour
        
        # Session definitions (UTC)
        self.df['asian_session'] = hour.isin(range(22, 24)).astype(int) | hour.isin(range(0, 8)).astype(int)
        self.df['london_session'] = hour.isin(range(7, 16)).astype(int)
        self.df['ny_session'] = hour.isin(range(13, 22)).astype(int)
        
        # Overlaps
        self.df['asian_london_overlap'] = hour.isin(range(7, 8)).astype(int)
        self.df['london_ny_overlap'] = hour.isin(range(13, 16)).astype(int)
        
        # Primary session
        conditions = [
            hour.isin(range(13, 16)),  # London-NY overlap
            hour.isin(range(7, 13)),   # London primary
            hour.isin(range(16, 22)),  # NY primary
            hour.isin(range(22, 24)) | hour.isin(range(0, 7))  # Asian
        ]
        choices = ['london_ny_overlap', 'london_primary', 'ny_primary', 'asian_primary']
        self.df['primary_session'] = np.select(conditions, choices, default='off_hours')
        
        # Session intensity (higher during overlaps)
        intensity_map = {
            'london_ny_overlap': 3,
            'london_primary': 2,
            'ny_primary': 2,
            'asian_primary': 1,
            'off_hours': 0
        }
        self.df['session_intensity'] = self.df['primary_session'].map(intensity_map)
    
    @feature_generator(
        output_features=['minutes_since_session_start', 'session_progress', 
                        'session_progress_sin', 'session_progress_cos'],
        dependencies=['fx_trading_sessions'],
        description="Session timing features",
        category="sessions"
    )
    def session_timing_features(self):
        """Generate session timing features"""
        hour = self.df.index.hour
        minute = self.df.index.minute
        
        # Calculate minutes since session start
        def calculate_minutes_since_start(hour, minute):
            current_minutes = hour * 60 + minute
            
            # London session: 7:00-16:00
            if 7 <= hour < 16:
                return current_minutes - (7 * 60)
            # NY session: 13:00-22:00  
            elif 13 <= hour < 22:
                return current_minutes - (13 * 60)
            # Asian session: 22:00-08:00 (next day)
            elif hour >= 22:
                return current_minutes - (22 * 60)
            elif hour < 8:
                return (current_minutes + 24 * 60) - (22 * 60)
            else:
                return 0
        
        self.df['minutes_since_session_start'] = [
            calculate_minutes_since_start(h, m) for h, m in zip(hour, minute)
        ]
        
        # Session progress (0-1)
        session_lengths = {
            'london_primary': 9 * 60,
            'ny_primary': 9 * 60,
            'asian_primary': 10 * 60,
            'london_ny_overlap': 3 * 60,
            'off_hours': 60
        }
        
        progress = []
        for minutes, session in zip(self.df['minutes_since_session_start'], self.df['primary_session']):
            length = session_lengths.get(session, 60)
            progress.append(min(1.0, max(0.0, minutes / length)))
        
        self.df['session_progress'] = progress
        
        # Cyclical session progress
        self.df['session_progress_sin'] = np.sin(2 * np.pi * self.df['session_progress'])
        self.df['session_progress_cos'] = np.cos(2 * np.pi * self.df['session_progress'])
    
    @feature_generator(
        output_features=['minutes_to_next_session', 'session_transition', 
                        'pre_london', 'pre_ny', 'london_close'],
        dependencies=['session_timing_features'],
        description="Session transition features",
        category="sessions"
    )
    def session_transition_features(self):
        """Generate session transition features"""
        hour = self.df.index.hour
        minute = self.df.index.minute
        
        # Calculate minutes to next session
        def calculate_minutes_to_next(hour, minute):
            current_minutes = hour * 60 + minute
            session_starts = [7*60, 13*60, 22*60, (7+24)*60]  # Include next day
            
            for start_time in session_starts:
                if start_time > current_minutes:
                    return start_time - current_minutes
            
            return (7*60 + 24*60) - current_minutes  # Next London session
        
        self.df['minutes_to_next_session'] = [
            calculate_minutes_to_next(h, m) for h, m in zip(hour, minute)
        ]
        
        # Session transition (within 30 minutes)
        self.df['session_transition'] = (
            (self.df['minutes_to_next_session'] <= 30) |
            (self.df['minutes_since_session_start'] <= 30)
        ).astype(int)
        
        # Specific transition periods
        self.df['pre_london'] = ((hour == 6) & (minute >= 30) | (hour == 7) & (minute < 30)).astype(int)
        self.df['pre_ny'] = ((hour == 12) & (minute >= 30) | (hour == 13) & (minute < 30)).astype(int)
        self.df['london_close'] = ((hour == 15) & (minute >= 30) | (hour == 16) & (minute < 30)).astype(int)
    
    @feature_generator(
        output_features=['returns', 'volatility_short', 'volatility_long', 'volatility_ratio',
                        'volatility_regime', 'volatility_zscore', 'volatility_change',
                        'vol_regime_high', 'vol_regime_low', 'vol_regime_medium'],
        description="Volatility regime features",
        category="volatility"
    )
    def volatility_regimes(self, lookback_short: int = 20, lookback_long: int = 100, n_regimes: int = 3):
        """Generate volatility regime features"""
        # Calculate returns
        self.df['returns'] = self.df['close'].pct_change()
        
        # Calculate volatilities
        self.df['volatility_short'] = self.df['returns'].rolling(window=lookback_short).std()
        self.df['volatility_long'] = self.df['returns'].rolling(window=lookback_long).std()
        
        # Volatility ratio
        self.df['volatility_ratio'] = self.df['volatility_short'] / self.df['volatility_long'].where(
            self.df['volatility_long'] > 0, 1
        )
        
        # Volatility z-score with adaptive window to reduce NaN values
        # Use expanding window for early periods, then switch to rolling
        adaptive_window = min(lookback_long, len(self.df) // 3)  # Use smaller window if dataset is small
        
        vol_mean = self.df['volatility_short'].rolling(window=adaptive_window, min_periods=1).mean()
        vol_std = self.df['volatility_short'].rolling(window=adaptive_window, min_periods=1).std()
        
        # Calculate z-score with better handling of division by zero
        vol_zscore = pd.Series(index=self.df.index, dtype=float)
        mask = (vol_std > 0) & vol_mean.notna() & vol_std.notna()
        vol_zscore[mask] = (self.df['volatility_short'][mask] - vol_mean[mask]) / vol_std[mask]
        vol_zscore[~mask] = 0.0  # Set to 0 where std is zero or data is missing
        
        self.df['volatility_zscore'] = vol_zscore
        
        # Volatility change with safe calculation
        vol_change = self.df['volatility_short'].pct_change(fill_method=None)
        # Replace infinite values that can occur when previous volatility is 0
        vol_change = vol_change.replace([np.inf, -np.inf], 0)
        self.df['volatility_change'] = vol_change
        
        # Volatility regimes using quantiles
        vol_quantiles = self.df['volatility_short'].quantile([0.33, 0.67])
        
        conditions = [
            self.df['volatility_short'] <= vol_quantiles.iloc[0],
            self.df['volatility_short'] <= vol_quantiles.iloc[1],
            self.df['volatility_short'] > vol_quantiles.iloc[1]
        ]
        
        self.df['volatility_regime'] = np.select(conditions, ['low', 'medium', 'high'], default='medium')
        
        # One-hot encode regimes
        self.df['vol_regime_low'] = (self.df['volatility_regime'] == 'low').astype(int)
        self.df['vol_regime_medium'] = (self.df['volatility_regime'] == 'medium').astype(int) 
        self.df['vol_regime_high'] = (self.df['volatility_regime'] == 'high').astype(int)
    
    @feature_generator(
        output_features=['session_atr_mean', 'session_atr_std'],
        dependencies=['average_true_range', 'fx_trading_sessions'],
        description="Session-based ATR statistics",
        category="volatility"
    )
    def session_atr_features(self):
        """Calculate ATR statistics by trading session"""
        # Calculate rolling session-based ATR statistics
        window = 20  # 20-period window for session statistics
        
        # Group by primary session and calculate rolling statistics
        session_groups = self.df.groupby('primary_session')['atr']
        
        # Initialize with NaN
        self.df['session_atr_mean'] = np.nan
        self.df['session_atr_std'] = np.nan
        
        # Calculate rolling statistics for each session
        for session_name, group in session_groups:
            if len(group) > 0:
                # Rolling mean and std for this session
                rolling_mean = group.rolling(window=min(window, len(group)), min_periods=1).mean()
                rolling_std = group.rolling(window=min(window, len(group)), min_periods=1).std()
                
                # Assign back to the main dataframe
                self.df.loc[group.index, 'session_atr_mean'] = rolling_mean
                self.df.loc[group.index, 'session_atr_std'] = rolling_std
        
        # Fill any remaining NaN values with global ATR statistics as fallback
        global_atr_mean = self.df['atr'].rolling(window=window, min_periods=1).mean()
        global_atr_std = self.df['atr'].rolling(window=window, min_periods=1).std()
        
        self.df['session_atr_mean'] = self.df['session_atr_mean'].fillna(global_atr_mean)
        self.df['session_atr_std'] = self.df['session_atr_std'].fillna(global_atr_std)
    
    @feature_generator(
        output_features=['future_atr'],
        dependencies=['average_true_range'],
        description="Forward-looking ATR values",
        category="volatility"
    )
    def future_atr_features(self):
        """Calculate forward-looking ATR features"""
        # Calculate future ATR by shifting current ATR forward
        # This represents the ATR that will be realized in the future
        lookforward_periods = 5  # Look 5 periods ahead
        
        # Shift ATR backward to get "future" values from current perspective
        self.df['future_atr'] = self.df['atr'].shift(-lookforward_periods)
        
        # For the last few rows where we don't have future data,
        # fill with the current ATR as a reasonable approximation
        self.df['future_atr'] = self.df['future_atr'].fillna(self.df['atr'])
    
    # ==========================================================================
    # MIGRATED FEATURES FROM LEGACY GENERATION.PY
    # ==========================================================================
    
    @feature_generator(
        output_features=['sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200',
                        'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_200'],
        description="Simple and Exponential Moving Averages",
        category="price_action"
    )
    def moving_averages(self):
        """Generate simple and exponential moving averages over different time periods"""
        num_periods = [5, 10, 20, 50, 200]
        
        new_features = {}
        
        for period in num_periods:
            new_features[f"sma_{period}"] = self.df["close"].rolling(window=period).mean()
            new_features[f"ema_{period}"] = self.df["close"].ewm(span=period, adjust=False, min_periods=period).mean()
        
        self._add_features_efficiently(new_features)
    
    @feature_generator(
        output_features=['roc_1', 'roc_5', 'roc_10', 'roc_20', 'roc_60'],
        description="Rate of change over various lookback periods",
        category="momentum"
    )
    def rate_of_change(self):
        """Price momentum: Rate of change (ROC) over various lookback periods with safe division"""
        lookback_periods = [1, 5, 10, 20, 60]
        
        new_features = {}
        
        for period in lookback_periods:
            price_shift = self.df["close"].shift(period)
            # Safe division - avoid inf when shifted price is 0
            new_features[f"roc_{period}"] = np.where(
                price_shift > 0,
                (self.df["close"] / price_shift - 1) * 100,
                0
            )
        
        self._add_features_efficiently(new_features)
    
    @feature_generator(
        output_features=['upper_band', 'lower_band', 'bb_width'],
        dependencies=['moving_averages'],
        description="Bollinger Bands for volatility analysis",
        category="volatility"
    )
    def bollinger_bands(self, period_bb: int = 20):
        """Statistical chart characterizing prices and volatility over time"""
        sma_key = f'sma_{period_bb}'
        std = self.df["close"].rolling(window=period_bb).std()
        num_std_dev = 2
        
        # Check if SMA already exists, if not calculate it
        if sma_key not in self.df.columns:
            sma = self.df["close"].rolling(window=period_bb).mean()
        else:
            sma = self.df[sma_key]
        
        # Calculate bands
        upper_band = sma + (num_std_dev * std)
        lower_band = sma - (num_std_dev * std)
        bb_width = (upper_band - lower_band) / sma
        
        new_features = {
            'upper_band': upper_band,
            'lower_band': lower_band,
            'bb_width': bb_width
        }
        
        if sma_key not in self.df.columns:
            new_features[sma_key] = sma
            
        self._add_features_efficiently(new_features)
    
    @feature_generator(
        output_features=['distance_from_high', 'distance_from_low'],
        description="Distance from recent highs and lows",
        category="support_resistance"
    )
    def support_resistance(self, lookback: int = 20):
        """Distance from recent high/lows with safe division"""
        rolling_high = self.df['high'].rolling(lookback).max()
        rolling_low = self.df['low'].rolling(lookback).min()
        
        new_features = {
            'distance_from_high': np.where(
                rolling_high > 0,
                (self.df['close'] / rolling_high - 1) * 100,
                0
            ),
            'distance_from_low': np.where(
                rolling_low > 0,
                (self.df['close'] / rolling_low - 1) * 100,
                0
            )
        }
        
        self._add_features_efficiently(new_features)
    
    @feature_generator(
        output_features=['vma_7', 'vma_15', 'vma_30', 'vma_60',
                        'relative_volume_7', 'relative_volume_15', 'relative_volume_30', 'relative_volume_60'],
        description="Volume moving averages and relative volume",
        category="volume"
    )
    def volume_moving_averages(self):
        """Identify unusual volume spikes with safe division"""
        num_periods = [7, 15, 30, 60]
        
        new_features = {}
        
        for period in num_periods:
            vma = self.df['volume'].rolling(window=period).mean()
            new_features[f'vma_{period}'] = vma
            # Safe division - avoid inf when vma is 0
            new_features[f'relative_volume_{period}'] = np.where(
                vma > 0,
                self.df['volume'] / vma,
                1
            )
            
        self._add_features_efficiently(new_features)
    
    # Individual feature generators for missing dependencies
    @feature_generator(
        output_features=['volume_ma_20'],
        description="20-period volume moving average",
        category="volume"
    )
    def volume_ma_20_feature(self):
        """Generate 20-period volume moving average"""
        self.df['volume_ma_20'] = self.df['volume'].rolling(window=20).mean()
    
    @feature_generator(
        output_features=['volume_ratio'],
        dependencies=['volume_ma_20'],
        description="Current volume relative to 20-period average",
        category="volume"
    )
    def volume_ratio_feature(self):
        """Generate volume ratio relative to 20-period average"""
        self.df['volume_ratio'] = np.where(
            self.df['volume_ma_20'] > 0,
            self.df['volume'] / self.df['volume_ma_20'],
            1.0
        )
    
    @feature_generator(
        output_features=['vroc_1', 'vroc_5', 'vroc_10', 'vroc_20'],
        description="Volume rate of change",
        category="volume"
    )
    def volume_rate_of_change(self):
        """How rapidly volume is increasing/decreasing with safe division"""
        lookback_periods = [1, 5, 10, 20]
        
        new_features = {}
        
        for period in lookback_periods:
            volume_shift = self.df["volume"].shift(period)
            # Safe division - avoid inf when shifted volume is 0
            new_features[f"vroc_{period}"] = np.where(
                volume_shift > 0,
                (self.df["volume"] / volume_shift - 1) * 100,
                0
            )
        
        self._add_features_efficiently(new_features)
    
    @feature_generator(
        output_features=['obv'],
        description="On Balance Volume indicator",
        category="volume"
    )
    def on_balance_volume(self):
        """Cumulative indicator that relates volume to price changes"""
        price_direction = np.sign(self.df['close'].diff())
        price_direction[self.df['close'].diff() == 0] = 0
        volume_direction = self.df['volume'] * price_direction
        obv = volume_direction.cumsum()
        
        self._add_features_efficiently({'obv': obv})
    
    @feature_generator(
        output_features=['vwap'],
        description="Volume Weighted Average Price",
        category="volume"
    )
    def vwap(self, period: int = 1):
        """Calculate daily Volume Weighted Average Price"""
        typical_price = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        vwap = (self.df['volume'] * typical_price).rolling(period).sum() / self.df['volume'].rolling(period).sum()
        
        self._add_features_efficiently({'vwap': vwap})
    
    @feature_generator(
        output_features=['mfi'],
        description="Money Flow Index - volume-weighted RSI",
        category="oscillator"
    )
    def money_flow_index(self, period: int = 14):
        """Calculate Money Flow Index - a volume-weighted RSI"""
        typical_price = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        money_flow = typical_price * self.df['volume']
        
        positive_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
        negative_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)
        
        positive_flow_sum = pd.Series(positive_flow, index=self.df.index).rolling(window=period).sum()
        negative_flow_sum = pd.Series(negative_flow, index=self.df.index).rolling(window=period).sum()
        
        money_ratio = np.where(negative_flow_sum != 0, positive_flow_sum / negative_flow_sum, 0)
        mfi = 100 - (100 / (1 + money_ratio))
        
        self._add_features_efficiently({'mfi': mfi})
    
    @feature_generator(
        output_features=['rsi_14'],
        description="Relative Strength Index",
        category="oscillator"
    )
    def relative_strength_index(self, window: int = 14):
        """Calculate RSI using Wilder's smoothing method"""
        delta = self.df["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # First values
        avg_gain = gain[:window].mean()
        avg_loss = loss[:window].mean()
        
        # Initialize lists with first values
        avg_gains = [avg_gain]
        avg_losses = [avg_loss]
        
        # Apply Wilder's smoothing
        for i in range(window, len(gain)):
            avg_gain = ((avg_gains[-1] * (window - 1)) + gain.iloc[i]) / window
            avg_loss = ((avg_losses[-1] * (window - 1)) + loss.iloc[i]) / window
            avg_gains.append(avg_gain)
            avg_losses.append(avg_loss)
        
        # Convert to Series with proper index
        avg_gain_series = pd.Series(avg_gains, index=self.df.index[window-1:])
        avg_loss_series = pd.Series(avg_losses, index=self.df.index[window-1:])
        
        # Calculate RS and RSI
        rs = avg_gain_series / avg_loss_series
        rsi = 100 - (100 / (1 + rs))
        
        # Add to dataframe
        rsi_full = pd.Series(index=self.df.index, dtype=float)
        rsi_full.loc[rsi.index] = rsi
        
        self._add_features_efficiently({f'rsi_{window}': rsi_full})
    
    @feature_generator(
        output_features=['macd_line', 'macd_signal', 'macd_histogram'],
        description="MACD - Moving Average Convergence Divergence",
        category="momentum"
    )
    def moving_average_convergence_divergence(self):
        """Trend-following momentum indicator"""
        ema_12 = self.df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = self.df['close'].ewm(span=26, adjust=False).mean()

        macd_line = ema_12 - ema_26
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        macd_histogram = macd_line - macd_signal
        
        new_features = {
            'macd_line': macd_line,
            'macd_signal': macd_signal,
            'macd_histogram': macd_histogram
        }
        
        self._add_features_efficiently(new_features)
    
    @feature_generator(
        output_features=['+di', '-di', 'adx'],
        description="Average Directional Index - trend strength",
        category="trend"
    )
    def average_directional_index(self, lookback: int = 14):
        """Measures trend strength"""
        high_diff = self.df["high"].diff()
        low_diff = self.df["low"].diff()
        
        up_move = high_diff
        down_move = -low_diff
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        tr1 = self.df["high"] - self.df["low"]
        tr2 = abs(self.df["high"] - self.df["close"].shift(1))
        tr3 = abs(self.df["low"] - self.df["close"].shift(1))
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        
        smoothed_tr = tr.rolling(window=lookback).sum()
        smoothed_plus_dm = pd.Series(plus_dm).rolling(window=lookback).sum()
        smoothed_minus_dm = pd.Series(minus_dm).rolling(window=lookback).sum()
        
        for i in range(lookback, len(tr)):
            smoothed_tr.iloc[i] = smoothed_tr.iloc[i-1] - (smoothed_tr.iloc[i-1]/lookback) + tr.iloc[i]
            smoothed_plus_dm.iloc[i] = smoothed_plus_dm.iloc[i-1] - (smoothed_plus_dm.iloc[i-1]/lookback) + plus_dm[i]
            smoothed_minus_dm.iloc[i] = smoothed_minus_dm.iloc[i-1] - (smoothed_minus_dm.iloc[i-1]/lookback) + minus_dm[i]
        
        plus_di = 100 * (smoothed_plus_dm / smoothed_tr)
        minus_di = 100 * (smoothed_minus_dm / smoothed_tr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=lookback).mean()
        
        new_features = {
            "+di": plus_di,
            "-di": minus_di,
            "adx": adx
        }
        
        self._add_features_efficiently(new_features)
    
    @feature_generator(
        output_features=['stoch_k', 'stoch_d'],
        description="Stochastic Oscillator",
        category="oscillator"
    )
    def stochastic_oscillator(self, k_period: int = 14, d_period: int = 3, slowing: int = 3):
        """Calculate Stochastic Oscillator"""
        low_min = self.df['low'].rolling(window=k_period).min()
        high_max = self.df['high'].rolling(window=k_period).max()
        
        k_fast = 100 * (self.df['close'] - low_min) / (high_max - low_min)
        k_fast = k_fast.replace([np.inf, -np.inf], np.nan).fillna(50)
        
        k = k_fast.rolling(window=slowing).mean() if slowing > 1 else k_fast
        d = k.rolling(window=d_period).mean()
        
        new_features = {
            'stoch_k': k,
            'stoch_d': d
        }
        
        self._add_features_efficiently(new_features)
    
    @feature_generator(
        output_features=['cci'],
        description="Commodity Channel Index",
        category="oscillator"
    )
    def commodity_channel_index(self, period: int = 20):
        """Calculate Commodity Channel Index with safe division"""
        tp = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        ma_tp = tp.rolling(window=period).mean()
        mad = abs(tp - ma_tp).rolling(window=period).mean()
        
        # Safe division - avoid inf when mad is 0
        cci = np.where(mad > 1e-10, (tp - ma_tp) / (0.015 * mad), 0)
        
        self._add_features_efficiently({'cci': cci})
    
    @feature_generator(
        output_features=['log_return'],
        description="Logarithmic returns",
        category="returns"
    )
    def log_returns(self):
        """Logarithmic returns: measure percentage change in price with safe calculation"""
        price_ratio = self.df['close'] / self.df['close'].shift(1)
        
        # Handle division by zero, negative values, and inf values
        log_return = np.where(
            (price_ratio > 0) & np.isfinite(price_ratio),
            np.log(price_ratio),
            0
        )
        
        self._add_features_efficiently({'log_return': log_return})
    
    @feature_generator(
        output_features=['vol_bin_0', 'vol_bin_1', 'vol_bin_2', 'vol_bin_3', 'vol_bin_4',
                        'vol_bin_5', 'vol_bin_6', 'vol_bin_7', 'vol_bin_8', 'vol_bin_9',
                        'vol_poc', 'vol_va_high', 'vol_va_low', 'vol_va_width',
                        'close_to_poc_pct', 'in_value_area', 'vol_concentration'],
        description="Volume Profile analysis with key metrics",
        category="volume"
    )
    def volume_profile(self, num_bins: int = 10, lookback: int = 20, key_metrics: bool = True):
        """Calculate rolling volume profile with adaptive bins and key metrics"""
        vol_profiles = np.zeros((len(self.df), num_bins))
        
        rolling_min = self.df['low'].rolling(window=lookback).min()
        rolling_max = self.df['high'].rolling(window=lookback).max()
        typical_price = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        
        new_features = {}
        temp_df = self.df.copy()
        temp_df['typical_price'] = typical_price
        
        vol_poc = np.full(len(self.df), np.nan)
        vol_va_high = np.full(len(self.df), np.nan)
        vol_va_low = np.full(len(self.df), np.nan)
        vol_va_width = np.full(len(self.df), np.nan)
        
        for i in range(lookback, len(self.df)):
            window = temp_df.iloc[i-lookback:i]
            price_min = rolling_min.iloc[i]
            price_max = rolling_max.iloc[i]
            price_range = price_max - price_min
            
            if price_range > 0:
                bin_size = price_range / num_bins
                
                for j, row in window.iterrows():
                    bin_idx = min(num_bins - 1, int((row['typical_price'] - price_min) / bin_size))
                    time_factor = 1 + 0.1 * (window.index.get_loc(j) / lookback)
                    vol_profiles[i, bin_idx] += row['volume'] * time_factor
                
                total_vol = np.sum(vol_profiles[i])
                if total_vol > 0:
                    vol_profiles[i] = vol_profiles[i] / total_vol * 100
                
                for bin_num in range(num_bins):
                    if f'vol_bin_{bin_num}' not in new_features:
                        new_features[f'vol_bin_{bin_num}'] = np.full(len(self.df), np.nan)
                    new_features[f'vol_bin_{bin_num}'][i] = vol_profiles[i, bin_num]
                
                if key_metrics:
                    poc_bin = np.argmax(vol_profiles[i])
                    vol_poc[i] = price_min + (poc_bin + 0.5) * bin_size
                    
                    value_area_threshold = 0.7
                    sorted_bins = np.argsort(vol_profiles[i])[::-1]
                    cum_vol = 0
                    value_area_bins = []
                    
                    for bin_idx in sorted_bins:
                        value_area_bins.append(bin_idx)
                        cum_vol += vol_profiles[i, bin_idx] / 100
                        if cum_vol >= value_area_threshold:
                            break
                    
                    vol_va_high[i] = price_min + (max(value_area_bins) + 1) * bin_size
                    vol_va_low[i] = price_min + min(value_area_bins) * bin_size
                    vol_va_width[i] = (vol_va_high[i] - vol_va_low[i]) / price_range
        
        if key_metrics:
            new_features['vol_poc'] = pd.Series(vol_poc, index=self.df.index)
            new_features['vol_va_high'] = pd.Series(vol_va_high, index=self.df.index)
            new_features['vol_va_low'] = pd.Series(vol_va_low, index=self.df.index)
            new_features['vol_va_width'] = pd.Series(vol_va_width, index=self.df.index)
            
            close_to_poc_pct = (self.df['close'] - new_features['vol_poc']) / new_features['vol_poc'] * 100
            in_value_area = ((self.df['close'] >= new_features['vol_va_low']) & 
                            (self.df['close'] <= new_features['vol_va_high'])).astype(int)
            
            entropy = np.zeros(len(self.df))
            for i in range(lookback, len(self.df)):
                profile = vol_profiles[i] / 100
                profile = profile[profile > 0]
                if len(profile) > 0:
                    entropy[i] = -np.sum(profile * np.log(profile))
            vol_concentration = 1 - entropy / np.log(num_bins)
            
            new_features['close_to_poc_pct'] = close_to_poc_pct
            new_features['in_value_area'] = in_value_area
            new_features['vol_concentration'] = pd.Series(vol_concentration, index=self.df.index)
        
        for key, value in new_features.items():
            if isinstance(value, np.ndarray):
                new_features[key] = pd.Series(value, index=self.df.index)
        
        self._add_features_efficiently(new_features)
    
    @feature_generator(
        output_features=['seasonal_trend', 'seasonal_seasonal', 'seasonal_residual', 'seasonal_norm'],
        description="Classical seasonal decomposition",
        category="seasonality"
    )
    def seasonal_decompose(self, period: int = 160):
        """Perform seasonal decomposition on the closing price"""
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            close_series = self.df['close'].dropna()
            
            if len(close_series) < 2 * period:
                logger.warning(f"Not enough data for seasonal decomposition (need {2*period}, have {len(close_series)})")
                return
            
            result = seasonal_decompose(
                close_series, 
                model='additive', 
                period=period, 
                extrapolate_trend='freq'
            )
            
            new_features = {
                'seasonal_trend': result.trend,
                'seasonal_seasonal': result.seasonal,
                'seasonal_residual': result.resid,
                'seasonal_norm': result.seasonal / self.df['close']
            }
            
            self._add_features_efficiently(new_features)
            
        except Exception as e:
            logger.warning(f"Error calculating seasonal decomposition: {e}")
    
    @feature_generator(
        output_features=['stl_trend', 'stl_seasonal', 'stl_residual', 'stl_seasonal_strength', 'stl_trend_strength'],
        description="STL decomposition - robust to outliers",
        category="seasonality"
    )
    def stl_decomposition(self, period: int = 160):
        """Perform STL decomposition on the closing price"""
        try:
            from statsmodels.tsa.seasonal import STL
            
            close_series = self.df['close'].dropna()
            
            if len(close_series) < 2 * period:
                logger.warning(f"Not enough data for STL decomposition (need {2*period}, have {len(close_series)})")
                return
            
            stl = STL(close_series, period=period, seasonal=13)
            result = stl.fit()
            
            new_features = {
                'stl_trend': result.trend,
                'stl_seasonal': result.seasonal,
                'stl_residual': result.resid,
                'stl_seasonal_strength': 1 - (result.resid.var() / (result.seasonal + result.resid).var()),
                'stl_trend_strength': 1 - (result.resid.var() / (result.trend + result.resid).var())
            }
            
            self._add_features_efficiently(new_features)
            
        except Exception as e:
            logger.warning(f"Error calculating STL decomposition: {e}")
    
    @feature_generator(
        output_features=['mstl_trend', 'mstl_residual'],
        description="MSTL decomposition for multiple seasonality",
        category="seasonality"
    )
    def mstl_decomposition(self, periods: list = [24, 160]):
        """Perform MSTL decomposition for multiple seasonality components"""
        try:
            from statsmodels.tsa.seasonal import MSTL
            
            close_series = self.df['close'].dropna()
            
            if len(close_series) < 2 * max(periods):
                logger.warning(f"Not enough data for MSTL decomposition (need {2*max(periods)}, have {len(close_series)})")
                return
            
            mstl = MSTL(close_series, periods=periods)
            result = mstl.fit()
            
            new_features = {
                'mstl_trend': result.trend,
                'mstl_residual': result.resid
            }
            
            # Add each seasonal component
            for i, period in enumerate(periods):
                new_features[f'mstl_seasonal_{period}'] = result.seasonal_[i]
            
            self._add_features_efficiently(new_features)
            
        except Exception as e:
            logger.warning(f"Error calculating MSTL decomposition: {e}")
    
    @feature_generator(
        output_features=['is_holiday', 'days_to_next_holiday'],
        description="Holiday features for market analysis",
        category="temporal"
    )
    def holiday_features(self):
        """Add features for market holidays and special events"""
        try:
            import holidays
            
            dates = pd.Series(self.df.index.date)
            
            # Create US holidays calendar (adjust for your target market)
            us_holidays = holidays.US(years=range(dates.dt.year.min(), dates.dt.year.max()+1))
            
            # Flag for holidays
            is_holiday = dates.isin(us_holidays).astype(int)
            
            # Distance to next holiday
            next_holiday = np.zeros(len(self.df))
            for i, date in enumerate(dates):
                for days_ahead in range(1, 30):
                    future_date = date + pd.Timedelta(days=days_ahead)
                    if future_date in us_holidays:
                        next_holiday[i] = days_ahead
                        break
            
            new_features = {
                'is_holiday': is_holiday,
                'days_to_next_holiday': next_holiday
            }
            
            self._add_features_efficiently(new_features)
            
        except ImportError:
            logger.warning("holidays package not installed. Run 'pip install holidays'")
        except Exception as e:
            logger.warning(f"Error calculating holiday features: {e}")
    
    @feature_generator(
        output_features=['hl_spread_bps', 'hl_spread_ma_20', 'hl_spread_ratio'],
        description="Market microstructure features using high-low spread as proxy",
        category="microstructure"
    )
    def market_microstructure_features(self):
        """
        Market microstructure features using high-low spread as proxy for bid-ask spread.
        This provides insights into market liquidity and volatility conditions.
        """
        new_features = {}
        
        # Market microstructure features - spread proxy using high-low
        # Until bid-ask data is available in live feed, use high-low as spread proxy
        hl_spread = self.df['high'] - self.df['low']
        mid_price = (self.df['high'] + self.df['low']) / 2
        new_features['hl_spread_bps'] = np.where(
            mid_price > 0, 
            (hl_spread / mid_price) * 10000, 
            0
        )
        
        # Rolling spread statistics - convert to pandas Series for rolling operations
        spread_series = pd.Series(new_features['hl_spread_bps'], index=self.df.index)
        new_features['hl_spread_ma_20'] = spread_series.rolling(window=20).mean()
        new_features['hl_spread_ratio'] = np.where(
            new_features['hl_spread_ma_20'] > 0,
            new_features['hl_spread_bps'] / new_features['hl_spread_ma_20'],
            1.0
        )
        
        # NOTE: When bid-ask data becomes available in live feed, 
        # replace with: new_features['bid_ask_spread_bps'] = (ask - bid) / mid * 10000
        
        self._add_features_efficiently(new_features)
    
    @feature_generator(
        output_features=['trend_strength', 'trend_regime',
                        'trend_regime_strong_trend', 'trend_regime_weak_trend', 'trend_regime_ranging'],
        dependencies=['moving_averages', 'average_directional_index'],
        description="Trend strength indicators and regime classification",
        category="trend"
    )
    def trend_strength_indicators(self, lookback: int = 20, adx_threshold: int = 25):
        """Create indicators to identify trending vs ranging markets"""
        import scipy.stats
        
        new_features = {}
        
        # ADX-based trend regime if ADX exists
        if 'adx' in self.df.columns:
            adx_trend_regime = np.where(self.df['adx'] > adx_threshold, 'trending', 'ranging')
            new_features['adx_trend_regime'] = adx_trend_regime
            
            regime_dummies = pd.get_dummies(pd.Series(adx_trend_regime, index=self.df.index), prefix='adx_regime')
            for col in regime_dummies.columns:
                new_features[col] = regime_dummies[col]
        
        # Price vs Moving Average indicators
        for period in [20, 50, 200]:
            sma_col = f'sma_{period}'
            if sma_col not in self.df.columns:
                new_features[sma_col] = self.df['close'].rolling(window=period).mean()
                sma_values = new_features[sma_col]
            else:
                sma_values = self.df[sma_col]
            
            new_features[f'dist_from_sma_{period}'] = (self.df['close'] / sma_values - 1) * 100
            new_features[f'sma_{period}_slope'] = sma_values.pct_change(5, fill_method=None) * 100
        
        # Moving average crossovers
        if 'sma_50' in self.df.columns and 'sma_200' in self.df.columns:
            new_features['ma_cross_50_200'] = np.where(self.df['sma_50'] > self.df['sma_200'], 1, -1)
        elif 'sma_50' in new_features and 'sma_200' in new_features:
            new_features['ma_cross_50_200'] = np.where(new_features['sma_50'] > new_features['sma_200'], 1, -1)
        
        # Linear regression indicators
        regression_slope = np.full(len(self.df), np.nan)
        r_squared = np.full(len(self.df), np.nan)
        
        for i in range(lookback, len(self.df)):
            y = self.df['close'].iloc[i-lookback:i].values
            x = np.arange(len(y))
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
            regression_slope[i] = slope
            r_squared[i] = r_value**2
        
        new_features['regression_slope'] = pd.Series(regression_slope, index=self.df.index)
        new_features['r_squared'] = pd.Series(r_squared, index=self.df.index)
        new_features['norm_regression_slope'] = new_features['regression_slope'] / self.df['close']
        
        # Combined trend strength indicator
        components = []
        
        if 'adx' in self.df.columns:
            components.append(self.df['adx'] / 100)
        
        components.append(new_features['r_squared'])
        
        # MA alignment component
        ma_alignment = (
            np.sign(new_features['dist_from_sma_20']) * 
            np.sign(new_features['dist_from_sma_50']) * 
            np.sign(new_features['dist_from_sma_200'])
        )
        ma_aligned = (abs(ma_alignment) == 1).astype(float)
        components.append(ma_aligned)
        
        trend_strength = pd.concat(components, axis=1).mean(axis=1) * 100
        new_features['trend_strength'] = trend_strength
        
        # Trend regime classification
        trend_regime = np.where(
            trend_strength > 70, 'strong_trend',
            np.where(trend_strength > 40, 'weak_trend', 'ranging')
        )
        new_features['trend_regime'] = trend_regime
        
        # One-hot encode trend regime
        trend_dummies = pd.get_dummies(pd.Series(trend_regime, index=self.df.index), prefix='trend_regime')
        for col in trend_dummies.columns:
            new_features[col] = trend_dummies[col]
        
        self._add_features_efficiently(new_features)

    @feature_generator(
        output_features=['mid_price'],
        description="Mid price calculation for spread analysis",
        category="spread_compression"
    )
    def mid_price_feature(self):
        """Calculate mid price for spread compression analysis"""
        self.df['mid_price'] = (self.df['high'] + self.df['low']) / 2

    @feature_generator(
        output_features=['spread_baseline', 'spread_recent', 'compression_ratio', 
                        'compression_consistency', 'spread_volatility_ewm'],
        dependencies=['hl_spread_bps', 'mid_price'],
        description="Spread compression detection features using EWMA",
        category="spread_compression"
    )
    def spread_compression_features(self):
        """
        Generate spread compression detection features using EWMA (better than std).
        
        Uses high-low spread as proxy until bid-ask data is available.
        EWMA provides better responsiveness for detecting compression patterns.
        """
        lookback_window = 60  # 60 periods for baseline
        alpha_baseline = 2 / (lookback_window + 1)  # EWMA alpha for baseline
        alpha_recent = 2 / (4)  # Fast EWMA for recent spread (3-period equivalent)
        
        # Ensure hl_spread_bps exists
        if 'hl_spread_bps' not in self.df.columns:
            # Calculate if missing
            hl_spread = self.df['high'] - self.df['low']
            mid_price = self.df['mid_price'] if 'mid_price' in self.df.columns else (self.df['high'] + self.df['low']) / 2
            spread_bps = np.where(mid_price > 0, (hl_spread / mid_price) * 10000, 0)
        else:
            spread_bps = self.df['hl_spread_bps']
        
        # EWMA baseline (slower moving average)
        self.df['spread_baseline'] = pd.Series(spread_bps).ewm(alpha=alpha_baseline, adjust=False).mean()
        
        # EWMA recent (faster moving average)  
        self.df['spread_recent'] = pd.Series(spread_bps).ewm(alpha=alpha_recent, adjust=False).mean()
        
        # Compression ratio (key feature)
        self.df['compression_ratio'] = np.where(
            self.df['spread_baseline'] > 0,
            (self.df['spread_baseline'] - self.df['spread_recent']) / self.df['spread_baseline'],
            0
        )
        
        # Spread volatility using EWMA (better than rolling std)
        spread_returns = pd.Series(spread_bps).pct_change(fill_method=None)
        spread_vol_ewm = spread_returns.ewm(alpha=0.1).std()  # 10-period equivalent EWMA std
        self.df['spread_volatility_ewm'] = spread_vol_ewm
        
        # Compression consistency (inverse of volatility)
        self.df['compression_consistency'] = np.where(
            spread_vol_ewm > 0,
            1 / (1 + spread_vol_ewm * 1000),
            1.0
        )

    @feature_generator(
        output_features=['volume_recent_ratio'],
        dependencies=['volume_ma_20'],
        description="Recent volume ratio for spread compression confirmation",
        category="spread_compression"
    )
    def volume_recent_ratio_feature(self):
        """Generate recent volume ratio using EWMA for spread compression"""
        # Fast EWMA for recent volume (3-period equivalent)
        alpha_recent = 2 / (4)
        volume_recent_ewm = self.df['volume'].ewm(alpha=alpha_recent, adjust=False).mean()
        
        # Volume recent ratio
        self.df['volume_recent_ratio'] = np.where(
            self.df['volume_ma_20'] > 0,
            volume_recent_ewm / self.df['volume_ma_20'],
            1.0
        )

    @feature_generator(
        output_features=['price_momentum_5'],
        dependencies=['mid_price'],
        description="5-period price momentum for spread compression",
        category="spread_compression"
    )
    def price_momentum_5_feature(self):
        """Generate 5-period price momentum"""
        self.df['price_momentum_5'] = (
            self.df['mid_price'] - self.df['mid_price'].shift(5)
        ) / self.df['mid_price'].shift(5)

    @feature_generator(
        output_features=['hour'],
        description="Simple hour feature for session detection",
        category="spread_compression"
    )
    def hour_feature(self):
        """Generate simple hour feature"""
        self.df['hour'] = self.df.index.hour

    @feature_generator(
        output_features=['compression_ratio_micro', 'compression_ratio_short', 'compression_ratio_medium'],
        dependencies=['hl_spread_bps', 'mid_price'],
        description="Multi-timeframe compression detection features",
        category="spread_compression"
    )
    def multi_timeframe_compression_features(self):
        """
        Generate multi-timeframe compression detection features.
        
        - Micro: 30min baseline, 3min recent (most sensitive)
        - Short: 1hr baseline, 5min recent  
        - Medium: 2hr baseline, 10min recent
        """
        # Use hl_spread_bps if available, otherwise calculate
        if 'hl_spread_bps' not in self.df.columns:
            hl_spread = self.df['high'] - self.df['low']
            mid_price = self.df['mid_price'] if 'mid_price' in self.df.columns else (self.df['high'] + self.df['low']) / 2
            spread_bps = np.where(mid_price > 0, (hl_spread / mid_price) * 10000, 0)
        else:
            spread_bps = self.df['hl_spread_bps']
        
        # Micro timeframe: 30min baseline, 3min recent (most sensitive)
        alpha_baseline_micro = 2 / (31)  # 30min baseline
        alpha_recent_micro = 2 / (4)     # 3min recent
        
        baseline_micro = pd.Series(spread_bps).ewm(alpha=alpha_baseline_micro, adjust=False).mean()
        recent_micro = pd.Series(spread_bps).ewm(alpha=alpha_recent_micro, adjust=False).mean()
        
        self.df['compression_ratio_micro'] = np.where(
            baseline_micro > 0,
            (baseline_micro - recent_micro) / baseline_micro,
            0
        )
        
        # Short timeframe: 1hr baseline, 5min recent
        alpha_baseline_short = 2 / (61)  # 1hr baseline
        alpha_recent_short = 2 / (6)     # 5min recent
        
        baseline_short = pd.Series(spread_bps).ewm(alpha=alpha_baseline_short, adjust=False).mean()
        recent_short = pd.Series(spread_bps).ewm(alpha=alpha_recent_short, adjust=False).mean()
        
        self.df['compression_ratio_short'] = np.where(
            baseline_short > 0,
            (baseline_short - recent_short) / baseline_short,
            0
        )
        
        # Medium timeframe: 2hr baseline, 10min recent
        alpha_baseline_medium = 2 / (121) # 2hr baseline
        alpha_recent_medium = 2 / (11)    # 10min recent
        
        baseline_medium = pd.Series(spread_bps).ewm(alpha=alpha_baseline_medium, adjust=False).mean()
        recent_medium = pd.Series(spread_bps).ewm(alpha=alpha_recent_medium, adjust=False).mean()
        
        self.df['compression_ratio_medium'] = np.where(
            baseline_medium > 0,
            (baseline_medium - recent_medium) / baseline_medium,
            0
        )

    @feature_generator(
        output_features=['volume_acceleration', 'volume_breakout', 'volume_spread_correlation'],
        dependencies=['volume_ratio', 'volume_ma_20'],
        description="Advanced volume confirmation features for spread compression",
        category="spread_compression"
    )
    def advanced_volume_confirmation_features(self):
        """
        Generate advanced volume confirmation features.
        
        - volume_acceleration: Momentum detection in volume
        - volume_breakout: Z-score > 2 detection
        - volume_spread_correlation: Rolling correlation
        """
        volume = self.df['volume']
        
        # Volume acceleration (momentum detection)
        # Rate of change in volume_ratio over 5 periods
        volume_ratio = self.df.get('volume_ratio', volume / self.df.get('volume_ma_20', volume.rolling(20).mean()))
        self.df['volume_acceleration'] = volume_ratio.pct_change(5, fill_method=None)
        
        # Volume breakout detection (z-score > 2)
        volume_mean = volume.rolling(60).mean()
        volume_std = volume.rolling(60).std()
        volume_zscore = np.where(volume_std > 0, (volume - volume_mean) / volume_std, 0)
        self.df['volume_breakout'] = np.where(volume_zscore > 2, 1, 0)
        
        # Rolling volume-spread correlation (20-period)
        if 'hl_spread_bps' in self.df.columns:
            spread_data = self.df['hl_spread_bps']
        else:
            # Calculate spread if not available
            hl_spread = self.df['high'] - self.df['low']
            mid_price = (self.df['high'] + self.df['low']) / 2
            spread_data = np.where(mid_price > 0, (hl_spread / mid_price) * 10000, 0)
        
        # Rolling correlation between volume and spread
        correlation_window = 20
        rolling_corr = volume.rolling(correlation_window).corr(pd.Series(spread_data, index=volume.index))
        self.df['volume_spread_correlation'] = rolling_corr.fillna(0)

    @feature_generator(
        output_features=['signal_strength_v2'],
        dependencies=['volume_ratio', 'compression_ratio_micro', 'volume_acceleration', 'london_session'],
        description="Enhanced signal strength calculator with optimized weights",
        category="spread_compression"
    )
    def enhanced_signal_strength_calculator(self):
        """
        Enhanced signal strength calculator optimized for validation results.
        
        Weights based on validation performance:
        - volume_ratio: 2.0x weight (strongest signal - 1.52x movement)
        - compression_ratio_micro: 1.5x weight
        - volume_acceleration: 1.2x weight
        - london_session: 0.5x weight
        """
        # Ensure dependencies exist with safe defaults
        volume_ratio = self.df.get('volume_ratio', 1)
        compression_ratio_micro = self.df.get('compression_ratio_micro', 0)
        volume_acceleration = self.df.get('volume_acceleration', 0)
        london_session = self.df.get('london_session', 0)
        
        # Normalize inputs for consistent scaling
        volume_ratio_norm = np.clip((volume_ratio - 1) * 2, -2, 2)  # Scale around 1
        compression_ratio_norm = np.clip(compression_ratio_micro * 5, -2, 2)  # Scale compression
        volume_accel_norm = np.clip(volume_acceleration * 10, -2, 2)  # Scale acceleration
        
        # Enhanced signal strength with optimized weights
        self.df['signal_strength_v2'] = (
            volume_ratio_norm * 2.0 +           # 2.0x weight (strongest predictor)
            compression_ratio_norm * 1.5 +      # 1.5x weight (micro timeframe)
            volume_accel_norm * 1.2 +           # 1.2x weight (momentum)
            london_session * 0.5                # 0.5x weight (session bonus)
        )

    @feature_generator(
        output_features=['signal_strength_compression'],
        dependencies=['compression_ratio', 'volume_ratio', 'compression_consistency', 'london_session'],
        description="Combined signal strength for spread compression patterns",
        category="spread_compression"
    )
    def signal_strength_compression_feature(self):
        """
        Generate combined signal strength for spread compression.
        
        Weighted combination of compression indicators with session bonus.
        """
        # Ensure dependencies exist with safe defaults
        compression_ratio = self.df.get('compression_ratio', 0)
        volume_ratio = self.df.get('volume_ratio', 1)
        compression_consistency = self.df.get('compression_consistency', 0)
        london_session = self.df.get('london_session', 0)
        
        # Combined signal strength (matching hypothesis weights)
        self.df['signal_strength_compression'] = (
            compression_ratio * 2.0 +           # Weight compression highest
            volume_ratio * 1.0 +                # Volume confirmation  
            compression_consistency * 1.5 +     # Consistency matters
            london_session * 0.3                # Session bonus
        )

    @feature_generator(
        output_features=['bid_ask_spread_bps', 'bid_ask_spread_relative'],
        description="True bid-ask spread features when spread data is available",
        category="spread_compression"
    )
    def bid_ask_spread_features(self):
        """
        Generate true bid-ask spread features when spread data is available.
        Falls back to high-low proxy if bid-ask data is not available.
        """
        # Check if true spread data is available
        if 'spread_close' in self.df.columns:
            # Use true bid-ask spread
            mid_price = (self.df['bid_close'] + self.df['ask_close']) / 2
            self.df['bid_ask_spread_bps'] = (self.df['spread_close'] / mid_price) * 10000
            self.df['bid_ask_spread_relative'] = self.df['spread_close'] / mid_price
        elif 'hl_spread_bps' in self.df.columns:
            # Fall back to high-low proxy
            self.df['bid_ask_spread_bps'] = self.df['hl_spread_bps']
            mid_price = (self.df['high'] + self.df['low']) / 2
            hl_spread = self.df['high'] - self.df['low']
            self.df['bid_ask_spread_relative'] = hl_spread / mid_price
        else:
            # Calculate high-low proxy if nothing exists
            hl_spread = self.df['high'] - self.df['low']
            mid_price = (self.df['high'] + self.df['low']) / 2
            self.df['bid_ask_spread_bps'] = (hl_spread / mid_price) * 10000
            self.df['bid_ask_spread_relative'] = hl_spread / mid_price

    @feature_generator(
        output_features=['spread_tight', 'spread_wide', 'spread_avg'],
        description="Spread compression hypothesis specific features from bid-ask data",
        category="spread_compression"
    )
    def spread_tight_features(self):
        """
        Generate spread tight, wide, and average features from your hypothesis.
        Uses true bid-ask data when available, falls back to high-low proxy.
        """
        # Check if true bid-ask data is available
        if all(col in self.df.columns for col in ['ask_low', 'bid_high', 'ask_high', 'bid_low', 'ask_close', 'bid_close']):
            # Use true bid-ask data (from your hypothesis)
            self.df['spread_tight'] = self.df['ask_low'] - self.df['bid_high']  # Tightest possible spread
            self.df['spread_wide'] = self.df['ask_high'] - self.df['bid_low']   # Widest spread in period
            self.df['spread_avg'] = self.df['ask_close'] - self.df['bid_close'] # Standard spread
        else:
            # Fall back to high-low approximations
            hl_spread = self.df['high'] - self.df['low']
            # These are approximations but better than nothing
            self.df['spread_tight'] = hl_spread * 0.3  # Assume tight spread is ~30% of HL range
            self.df['spread_wide'] = hl_spread         # Wide spread approximates HL range
            self.df['spread_avg'] = hl_spread * 0.5    # Average spread is ~50% of HL range

    @feature_generator(
        output_features=['spread_relative', 'spread_baseline_true', 'spread_recent_true', 'compression_ratio_true'],
        dependencies=['spread_avg', 'mid_price'],
        description="Enhanced spread compression features using true spread data",
        category="spread_compression"
    )
    def spread_relative_features(self):
        """
        Generate relative spread and compression features using true spread data.
        This enhances the compression detection with actual bid-ask information.
        """
        # Relative spread (normalized by mid-price)
        mid_price = self.df.get('mid_price', (self.df['high'] + self.df['low']) / 2)
        self.df['spread_relative'] = self.df['spread_avg'] / mid_price
        
        # Enhanced baseline and recent using true spread relative values
        lookback_window = 60  # 60 periods for baseline
        alpha_baseline = 2 / (lookback_window + 1)  # EWMA alpha for baseline
        alpha_recent = 2 / (4)  # Fast EWMA for recent spread (3-period equivalent)
        
        # EWMA baseline and recent using relative spread (better than absolute)
        self.df['spread_baseline_true'] = self.df['spread_relative'].ewm(alpha=alpha_baseline, adjust=False).mean()
        self.df['spread_recent_true'] = self.df['spread_relative'].ewm(alpha=alpha_recent, adjust=False).mean()
        
        # Enhanced compression ratio using true spreads
        self.df['compression_ratio_true'] = np.where(
            self.df['spread_baseline_true'] > 0,
            (self.df['spread_baseline_true'] - self.df['spread_recent_true']) / self.df['spread_baseline_true'],
            0
        )