from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from .base_combiner import BaseCombiner

logger = logging.getLogger(__name__)


class TimeSeriesDataCombiner(BaseCombiner):
    """
    Main combiner for creating time-indexed datasets from market data and calendar events.
    
    This combiner transforms discrete calendar events (earnings, options expiration, 
    holidays, fiscal quarters) into continuous time-series features aligned with 
    market data timestamps.
    
    Key Features:
    - Transforms JSON event data into time-indexed features
    - Maintains precise temporal alignment with market data
    - Supports multiple proximity calculations and categorical zones
    - Creates features suitable for ML training and statistical analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize TimeSeriesDataCombiner.
        
        Args:
            config: Configuration with feature engineering parameters
        """
        super().__init__(config)
        
        # Feature engineering configuration
        self.proximity_zones = self.config.get('proximity_zones', {
            'immediate': 1,      # 1 day
            'near_term': 7,      # 1 week  
            'medium_term': 30,   # 1 month
            'long_term': 90      # 3 months
        })
        
        self.holiday_impact_days = self.config.get('holiday_impact_days', {
            'pre_holiday': 3,
            'post_holiday': 2,
            'vacation_period': 1  # Flag during entire period
        })
    
    def combine_data(self, symbol: str, timeframe: str, date_range: Dict[str, str], 
                    data_sources: List[str]) -> pd.DataFrame:
        """
        Combine market data with calendar events into unified time-indexed DataFrame.
        
        Args:
            symbol: Trading symbol (e.g., 'GBPUSD', 'CLOV')
            timeframe: Time granularity (e.g., '1m', '5m', '1h', '1d')
            date_range: Dict with 'start' and 'end' date strings
            data_sources: List of data sources - supports ['calendar_events'] currently
            
        Returns:
            pd.DataFrame: Time-indexed DataFrame with market data + event features
        """
        # Validate inputs
        self.validate_inputs(symbol, timeframe, date_range, data_sources)
        
        # Load base market data
        market_df = self._load_market_data(symbol, timeframe, date_range)
        
        if len(market_df) == 0:
            raise ValueError(f"No market data found for {symbol} in specified date range")
        
        # Start with market data as base
        combined_df = market_df.copy()
        
        # Add features from each requested data source
        for source in data_sources:
            if source == 'calendar_events':
                calendar_features = self._load_and_process_calendar_events(symbol, combined_df.index)
                combined_df = combined_df.join(calendar_features, how='left')
            else:
                self.logger.warning(f"Unknown data source: {source}")
        
        # Fill NaN values with appropriate defaults
        combined_df = self._fill_missing_values(combined_df)
        
        # Save combined dataset
        output_path = self._save_combined_data(combined_df, symbol, timeframe, date_range, data_sources)
        
        self.logger.info(f"Combined dataset created: {len(combined_df)} rows, {len(combined_df.columns)} features")
        
        return combined_df
    
    def get_available_combinations(self, symbol: str) -> Dict[str, List[str]]:
        """
        Check what data source combinations are available for a symbol.
        
        Args:
            symbol: Trading symbol to check
            
        Returns:
            Dict mapping data types to available sources
        """
        available = {
            'market_data': [],
            'calendar_events': [],
            'supported_combinations': []
        }
        
        # Check for market data
        market_data_dir = Path(f"storage/data/capital_com/{symbol}/raw")
        if market_data_dir.exists():
            timeframes = [d.name for d in market_data_dir.iterdir() if d.is_dir()]
            available['market_data'] = timeframes
        
        # Check for calendar events data
        calendar_dir = Path(f"storage/multi_data_sources/calendar_events/{symbol}")
        if calendar_dir.exists():
            event_types = [d.name for d in calendar_dir.iterdir() if d.is_dir()]
            available['calendar_events'] = event_types
            
            if event_types:
                available['supported_combinations'].append('calendar_events')
        
        return available
    
    def _load_and_process_calendar_events(self, symbol: str, time_index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Load calendar events data and convert to time-indexed features.
        
        Args:
            symbol: Trading symbol
            time_index: Time index from market data to align features with
            
        Returns:
            pd.DataFrame: Time-indexed calendar event features
        """
        calendar_data = self._load_calendar_events_data(symbol)
        
        if not calendar_data:
            self.logger.warning(f"No calendar events data found for {symbol}")
            return pd.DataFrame(index=time_index)
        
        # Create features DataFrame
        features_df = pd.DataFrame(index=time_index)
        
        # Process earnings features
        if 'earnings' in calendar_data:
            earnings_features = self._create_earnings_features(calendar_data['earnings'], time_index)
            features_df = features_df.join(earnings_features)
        
        # Process options expiration features
        if 'options_expiration' in calendar_data:
            options_features = self._create_options_features(calendar_data['options_expiration'], time_index)
            features_df = features_df.join(options_features)
        
        # Process holiday/vacation features
        if 'holidays_vacations' in calendar_data:
            holiday_features = self._create_holiday_features(calendar_data['holidays_vacations'], time_index)
            features_df = features_df.join(holiday_features)
        
        # Process fiscal quarter features
        if 'fiscal_quarters' in calendar_data:
            quarter_features = self._create_quarter_features(calendar_data['fiscal_quarters'], time_index)
            features_df = features_df.join(quarter_features)
        
        # Process cross-calendar interaction features
        if len(features_df.columns) > 1:  # Only if we have multiple event types
            interaction_features = self._create_interaction_features(features_df)
            features_df = features_df.join(interaction_features)
        
        return features_df
    
    def _load_calendar_events_data(self, symbol: str) -> Dict[str, Any]:
        """Load all calendar events data for a symbol."""
        calendar_data = {}
        base_dir = Path(f"storage/multi_data_sources/calendar_events/{symbol}")
        
        if not base_dir.exists():
            return calendar_data
        
        # Load earnings data
        earnings_files = list((base_dir / "earnings" / "raw").glob("*.json"))
        if earnings_files:
            with open(earnings_files[0], 'r') as f:
                data = json.load(f)
                calendar_data['earnings'] = data
        
        # Load options expiration data
        options_files = list((base_dir / "options_expiration" / "raw").glob("*.json"))
        if options_files:
            with open(options_files[0], 'r') as f:
                data = json.load(f)
                calendar_data['options_expiration'] = data
        
        # Load cross-calendar data (includes holidays, fiscal quarters)
        cross_files = list((base_dir / "cross_calendar" / "raw").glob("*_cross_calendar_complete.json"))
        if cross_files:
            with open(cross_files[0], 'r') as f:
                data = json.load(f)
                calendar_data.update(data)  # Merge holidays_vacations, fiscal_quarters, etc.
        
        return calendar_data
    
    def _create_earnings_features(self, earnings_data: Dict[str, Any], time_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Create time-indexed features from earnings data."""
        features_df = pd.DataFrame(index=time_index)
        
        # Extract earnings dates
        earnings_dates = []
        
        # Get upcoming earnings
        if 'earnings_calendar' in earnings_data:
            for event in earnings_data['earnings_calendar']:
                if 'Earnings Date' in event:
                    try:
                        earnings_dates.append(pd.to_datetime(event['Earnings Date']))
                    except:
                        continue
        
        # Get historical earnings
        if 'historical_earnings_dates' in earnings_data:
            for event in earnings_data['historical_earnings_dates']:
                if 'Earnings Date' in event:
                    try:
                        earnings_dates.append(pd.to_datetime(event['Earnings Date']))
                    except:
                        continue
        
        if earnings_dates:
            # TEMPORAL INTEGRITY: Only use earnings dates that occurred BEFORE each timestamp
            features_df['days_since_last_earnings'] = self._calculate_days_since_last_event(time_index, earnings_dates)
            
            # Create proximity zones based on HISTORICAL patterns only
            features_df['earnings_proximity_zone'] = self._categorize_proximity_safe(features_df['days_since_last_earnings'])
            
            # Binary flags for key periods (based on days SINCE, not TO)
            features_df['earnings_immediate_zone'] = (features_df['days_since_last_earnings'] <= 7).astype(int)
            features_df['earnings_near_term_zone'] = (features_df['days_since_last_earnings'] <= 30).astype(int)
            
            # Fill NaN values with safe defaults
            features_df['days_since_last_earnings'] = features_df['days_since_last_earnings'].fillna(999)
        
        return features_df
    
    def _create_options_features(self, options_data: Dict[str, Any], time_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Create time-indexed features from options expiration data."""
        features_df = pd.DataFrame(index=time_index)
        
        # Extract expiration dates
        expiration_dates = []
        
        if 'expiration_calendar' in options_data:
            for event in options_data['expiration_calendar']:
                if 'expiration_date' in event:
                    try:
                        expiration_dates.append(pd.to_datetime(event['expiration_date']))
                    except:
                        continue
        
        if expiration_dates:
            # TEMPORAL INTEGRITY: Only use expiration dates that occurred BEFORE each timestamp
            features_df['days_since_options_expiration'] = self._calculate_days_since_last_event(time_index, expiration_dates)
            
            # Create proximity zones based on HISTORICAL patterns only
            features_df['options_proximity_zone'] = self._categorize_proximity_safe(features_df['days_since_options_expiration'])
            
            # Binary flags for key periods (based on days SINCE, not TO)
            features_df['options_expiry_week'] = (features_df['days_since_options_expiration'] <= 7).astype(int)
            features_df['options_expiry_immediate'] = (features_df['days_since_options_expiration'] <= 2).astype(int)
            
            # Fill NaN values with safe defaults  
            features_df['days_since_options_expiration'] = features_df['days_since_options_expiration'].fillna(999)
        
        return features_df
    
    def _create_holiday_features(self, holiday_data: Dict[str, Any], time_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Create time-indexed features from holiday/vacation data."""
        features_df = pd.DataFrame(index=time_index)
        
        # Initialize flags
        features_df['is_holiday_period'] = 0
        features_df['is_vacation_period'] = 0
        features_df['pre_holiday_effect'] = 0
        features_df['post_holiday_effect'] = 0
        
        # Process market holidays
        if 'market_holidays' in holiday_data:
            holidays = [pd.to_datetime(date) for date in holiday_data['market_holidays']]
            
            for holiday in holidays:
                # Pre-holiday effect (3 days before)
                pre_start = holiday - timedelta(days=self.holiday_impact_days['pre_holiday'])
                pre_mask = (time_index >= pre_start) & (time_index < holiday)
                features_df.loc[pre_mask, 'pre_holiday_effect'] = 1
                
                # Post-holiday effect (2 days after)
                post_end = holiday + timedelta(days=self.holiday_impact_days['post_holiday'])
                post_mask = (time_index > holiday) & (time_index <= post_end)
                features_df.loc[post_mask, 'post_holiday_effect'] = 1
                
                # Holiday itself
                holiday_mask = time_index.date == holiday.date()
                features_df.loc[holiday_mask, 'is_holiday_period'] = 1
        
        # Process vacation periods
        if 'vacation_periods' in holiday_data:
            for period in holiday_data['vacation_periods']:
                try:
                    start_date = pd.to_datetime(period['start'])
                    end_date = pd.to_datetime(period['end'])
                    
                    vacation_mask = (time_index >= start_date) & (time_index <= end_date)
                    features_df.loc[vacation_mask, 'is_vacation_period'] = 1
                except:
                    continue
        
        return features_df
    
    def _create_quarter_features(self, quarter_data: Dict[str, Any], time_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Create time-indexed features from fiscal quarter data."""
        features_df = pd.DataFrame(index=time_index)
        
        # Initialize features
        features_df['days_to_quarter_end'] = 999  # Large default value
        features_df['is_quarter_end_period'] = 0
        features_df['quarter_number'] = 0
        
        if 'quarterly_dates' in quarter_data:
            quarter_ends = []
            
            for quarter in quarter_data['quarterly_dates']:
                try:
                    quarter_end = pd.to_datetime(quarter['quarter_end_date'])
                    quarter_ends.append((quarter_end, quarter.get('quarter', 'Q1')))
                except:
                    continue
            
            if quarter_ends:
                # TEMPORAL INTEGRITY: Calculate days since last quarter end (no lookahead)
                quarter_end_dates = [qe[0] for qe in quarter_ends]
                features_df['days_since_quarter_end'] = self._calculate_days_since_last_event(time_index, quarter_end_dates)
                
                # Recent quarter-end period flag (within 30 days of last quarter end)
                features_df['is_quarter_end_period'] = (features_df['days_since_quarter_end'] <= 30).astype(int)
                
                # Fill NaN values with safe defaults
                features_df['days_since_quarter_end'] = features_df['days_since_quarter_end'].fillna(999)
                
                # Assign quarter numbers
                for timestamp in time_index:
                    for quarter_end, quarter_name in quarter_ends:
                        if timestamp <= quarter_end:
                            quarter_num = int(quarter_name[1]) if quarter_name.startswith('Q') else 1
                            features_df.loc[timestamp, 'quarter_number'] = quarter_num
                            break
        
        return features_df
    
    def _create_interaction_features(self, existing_features: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between different calendar events."""
        interaction_df = pd.DataFrame(index=existing_features.index)
        
        # Earnings + Options interactions (using SINCE-based features)
        if 'days_since_last_earnings' in existing_features.columns and 'days_since_options_expiration' in existing_features.columns:
            # Recent events (both within same period)
            interaction_df['earnings_options_same_week'] = (
                (existing_features['days_since_last_earnings'] <= 7) & 
                (existing_features['days_since_options_expiration'] <= 7)
            ).astype(int)
            
            # Recent overlapping events (both very recent)
            interaction_df['earnings_options_conflict'] = (
                (existing_features['days_since_last_earnings'] <= 3) & 
                (existing_features['days_since_options_expiration'] <= 3)
            ).astype(int)
        
        # Holiday + Quarter-end interactions
        if 'is_vacation_period' in existing_features.columns and 'is_quarter_end_period' in existing_features.columns:
            interaction_df['vacation_quarter_end_overlap'] = (
                existing_features['is_vacation_period'] & 
                existing_features['is_quarter_end_period']
            ).astype(int)
        
        # Complex institutional blind spot score
        blind_spot_factors = []
        
        if 'earnings_immediate_zone' in existing_features.columns:
            blind_spot_factors.append(existing_features['earnings_immediate_zone'])
        if 'options_expiry_immediate' in existing_features.columns:
            blind_spot_factors.append(existing_features['options_expiry_immediate'])
        if 'is_vacation_period' in existing_features.columns:
            blind_spot_factors.append(existing_features['is_vacation_period'])
        if 'is_quarter_end_period' in existing_features.columns:
            blind_spot_factors.append(existing_features['is_quarter_end_period'])
        
        if blind_spot_factors:
            interaction_df['institutional_blind_spot_score'] = sum(blind_spot_factors)
        
        return interaction_df
    
    # REMOVED: _calculate_days_to_next_event - caused lookahead bias
    # Use _calculate_days_since_last_event_safe instead
    
    def _calculate_days_since_last_event(self, time_index: pd.DatetimeIndex, event_dates: List[pd.Timestamp]) -> pd.Series:
        """Calculate days since last event for each timestamp."""
        days_since_event = pd.Series(index=time_index, dtype=float)
        
        # Normalize timezones - convert everything to UTC or remove timezone info
        normalized_events = []
        for event in event_dates:
            if event.tz is not None:
                # Convert to UTC then make timezone naive
                normalized_events.append(event.tz_convert('UTC').tz_localize(None))
            else:
                normalized_events.append(event)
        
        # Ensure time_index is also timezone naive
        if time_index.tz is not None:
            time_index_normalized = time_index.tz_convert('UTC').tz_localize(None)
        else:
            time_index_normalized = time_index
        
        for i, timestamp in enumerate(time_index_normalized):
            past_events = [event for event in normalized_events if event <= timestamp]
            
            if past_events:
                last_event = max(past_events)
                days_diff = (timestamp - last_event).days
                days_since_event.iloc[i] = days_diff
            else:
                days_since_event.iloc[i] = 999  # No past events
        
        return days_since_event
    
    def _categorize_proximity_safe(self, days_since_event: pd.Series) -> pd.Series:
        """Categorize proximity using days SINCE last event (no lookahead)."""
        proximity = pd.Series(index=days_since_event.index, dtype=str)
        
        # Categories based on days SINCE last event
        proximity[days_since_event <= 7] = 'recent_post'
        proximity[(days_since_event > 7) & (days_since_event <= 30)] = 'medium_post' 
        proximity[(days_since_event > 30) & (days_since_event <= 90)] = 'long_post'
        proximity[days_since_event > 90] = 'long_term'
        proximity[days_since_event >= 999] = 'unknown'
        
        return proximity
    
    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN values with appropriate defaults for different feature types."""
        filled_df = df.copy()
        
        # Days-based features - fill with large number indicating "far away"
        days_columns = [col for col in df.columns if 'days_to' in col or 'days_since' in col]
        for col in days_columns:
            filled_df[col] = filled_df[col].fillna(999)
        
        # Binary flags - fill with 0
        binary_columns = [col for col in df.columns if any(keyword in col for keyword in 
                         ['is_', '_zone', '_period', '_effect', '_conflict', '_overlap', '_score'])]
        for col in binary_columns:
            filled_df[col] = filled_df[col].fillna(0)
        
        # Categorical features - fill with 'long_term' or 'none'
        categorical_columns = [col for col in df.columns if 'proximity_zone' in col]
        for col in categorical_columns:
            filled_df[col] = filled_df[col].fillna('long_term')
        
        # Quarter numbers - forward fill then backfill
        if 'quarter_number' in filled_df.columns:
            filled_df['quarter_number'] = filled_df['quarter_number'].fillna(method='ffill').fillna(method='bfill').fillna(1)
        
        return filled_df