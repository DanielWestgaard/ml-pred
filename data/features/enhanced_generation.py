"""
Enhanced Model-Aware Feature Generator
=====================================

This module provides a robust feature generation system that:
- Automatically generates features based on model metadata
- Handles feature dependencies automatically
- Provides clear feedback about missing features
- Is easily extensible with new features
"""

import logging
import pandas as pd
import numpy as np
import yaml
from typing import Dict, List, Optional, Any, Set
from pathlib import Path

from .feature_registry import FeatureRegistry, FeatureDefinition, feature_generator, AutoRegisterMeta
from utils import data_utils


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
                logging.warning(f"Dataset contains only {len(self.df)} rows. Some features may not generate properly.")
            
            # Store configuration
            self.model_metadata_path = model_metadata_path
            self.selected_features = selected_features
            self._generated_features = set()
            
            logging.info(f"Enhanced FeatureGenerator initialized with {len(self.df)} rows")
            
        except Exception as e:
            logging.error(f"Error initializing Enhanced FeatureGenerator: {e}")
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
            logging.info(f"Model requires {len(required_features)} features")
            
        except Exception as e:
            logging.error(f"Error loading model metadata: {e}")
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
        logging.info(f"Generating {len(feature_names)} selected features...")
        
        # Validate feature availability
        available, unavailable = self._registry.validate_feature_availability(feature_names)
        
        if unavailable:
            # Check if unavailable features are basic columns
            basic_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_basic = [f for f in unavailable if f in basic_columns]
            missing_features = [f for f in unavailable if f not in basic_columns]
            
            if missing_basic:
                logging.info(f"Basic columns already available: {missing_basic}")
            
            if missing_features:
                logging.warning(f"Cannot generate features: {missing_features}")
                # Continue with available features
                feature_names = available
        
        # Get required generators
        generators, missing = self._registry.get_generators_for_features(feature_names)
        
        if missing:
            logging.warning(f"No generators found for features: {missing}")
        
        if not generators:
            logging.warning("No generators needed - returning original dataset")
            return self.df
        
        # Resolve dependencies and get execution order
        try:
            execution_order = self._registry.resolve_dependencies(generators)
            logging.debug(f"Execution order: {execution_order}")
        except ValueError as e:
            logging.error(f"Dependency resolution failed: {e}")
            raise
        
        # Execute generators in order
        failed_generators = []
        
        for generator_name in execution_order:
            try:
                if hasattr(self, generator_name):
                    method = getattr(self, generator_name)
                    logging.debug(f"Executing generator: {generator_name}")
                    method()
                    
                    # Track generated features
                    generator_def = self._registry.get_generator_info(generator_name)
                    if generator_def:
                        self._generated_features.update(generator_def.output_features)
                else:
                    logging.warning(f"Generator method '{generator_name}' not found")
                    failed_generators.append(generator_name)
                    
            except Exception as e:
                logging.error(f"Error in generator '{generator_name}': {e}")
                failed_generators.append(generator_name)
        
        # Report results
        actual_features = [f for f in feature_names if f in self.df.columns]
        logging.info(f"Successfully generated {len(actual_features)} features")
        
        if failed_generators:
            logging.warning(f"Failed generators: {failed_generators}")
        
        missing_expected = [f for f in feature_names if f not in self.df.columns and f not in ['open', 'high', 'low', 'close', 'volume']]
        if missing_expected:
            logging.warning(f"Expected features not generated: {missing_expected}")
        
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
                logging.error(f"Error in generator '{generator_name}': {e}")
        
        return self.df
    
    def _add_features_efficiently(self, new_features_dict: Dict[str, Any], drop_columns: List[str] = None):
        """Add multiple features efficiently to avoid DataFrame fragmentation"""
        if new_features_dict:
            for name, values in new_features_dict.items():
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
        
        # Volatility change
        self.df['volatility_change'] = self.df['volatility_short'].pct_change()
        
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