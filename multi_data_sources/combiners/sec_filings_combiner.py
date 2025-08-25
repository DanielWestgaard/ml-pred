from typing import Dict, Any, List, Optional
import pandas as pd
import logging
from pathlib import Path
from datetime import timedelta
from .base_combiner import BaseCombiner

logger = logging.getLogger(__name__)


class SECFilingsCombiner(BaseCombiner):
    """
    Combiner for creating time-indexed datasets from market data and SEC 8-K filings.
    
    This combiner transforms discrete 8-K filing events into continuous time-series 
    features aligned with market data timestamps, enabling analysis of how SEC filings
    impact market behavior.
    
    Key Features:
    - Transforms 8-K filing data into time-indexed features
    - Maintains precise temporal alignment with market data
    - Creates features based on filing dates, form types, and item categories
    - Supports proximity calculations for pre/post-filing analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize SECFilingsCombiner.
        
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
        
        # Important 8-K items that typically have market impact
        self.high_impact_items = self.config.get('high_impact_items', [
            '1.01',  # Entry into Material Agreement
            '2.02',  # Results of Operations and Financial Condition
            '5.02',  # Departure of Directors or Principal Officers
            '5.03',  # Amendments to Articles of Incorporation or Bylaws
            '7.01',  # Regulation FD Disclosure
            '8.01'   # Other Events
        ])
    
    def combine_data(self, symbol: str, timeframe: str, date_range: Dict[str, str], 
                    data_sources: List[str]) -> pd.DataFrame:
        """
        Combine market data with SEC filings into unified time-indexed DataFrame.
        
        Args:
            symbol: Trading symbol (e.g., 'BDTX')
            timeframe: Time granularity (e.g., '1m', '5m', '1h', '1d')
            date_range: Dict with 'start' and 'end' date strings
            data_sources: List of data sources - supports ['sec_filings'] currently
            
        Returns:
            pd.DataFrame: Time-indexed DataFrame with market data + SEC filing features
        """
        # Validate inputs
        self.validate_inputs(symbol, timeframe, date_range, data_sources)
        
        # Load base market data - handle timeframe mapping
        market_df = self._load_market_data_alpaca(symbol, timeframe, date_range)
        
        if len(market_df) == 0:
            raise ValueError(f"No market data found for {symbol} in specified date range")
        
        # Start with market data as base
        combined_df = market_df.copy()
        
        # Add features from each requested data source
        for source in data_sources:
            if source == 'sec_filings':
                filing_features = self._load_and_process_sec_filings(symbol, combined_df.index, date_range)
                combined_df = combined_df.join(filing_features, how='left')
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
            'sec_filings': [],
            'supported_combinations': []
        }
        
        # Check for market data
        market_data_dir = Path(f"storage/data/alpaca/{symbol}/raw")
        if market_data_dir.exists():
            timeframes = [d.name for d in market_data_dir.iterdir() if d.is_dir()]
            available['market_data'] = timeframes
        
        # Check for SEC filings data
        filings_path = Path(f"storage/data/sec_edgar/{symbol}/{symbol}_8K_filings.csv")
        if filings_path.exists():
            available['sec_filings'] = ['8K_filings']
            available['supported_combinations'].append('sec_filings')
        
        return available
    
    def _load_and_process_sec_filings(self, symbol: str, time_index: pd.DatetimeIndex, 
                                     date_range: Dict[str, str]) -> pd.DataFrame:
        """
        Load SEC filings data and convert to time-indexed features.
        
        Args:
            symbol: Trading symbol
            time_index: Time index from market data to align features with
            date_range: Date range for filtering filings
            
        Returns:
            pd.DataFrame: Time-indexed SEC filing features
        """
        filings_data = self._load_sec_filings_data(symbol, date_range)
        
        if filings_data.empty:
            self.logger.warning(f"No SEC filings data found for {symbol}")
            return pd.DataFrame(index=time_index)
        
        # Create features DataFrame
        features_df = pd.DataFrame(index=time_index)
        
        # Extract filing dates
        filing_dates = pd.to_datetime(filings_data['filing_date']).tolist()
        
        if filing_dates:
            # Basic temporal features
            features_df['days_since_last_filing'] = self._calculate_days_since_last_event(time_index, filing_dates)
            features_df['filing_proximity_zone'] = self._categorize_proximity_safe(features_df['days_since_last_filing'])
            
            # Filing frequency features
            features_df['filings_in_last_30_days'] = self._count_events_in_window(time_index, filing_dates, 30)
            features_df['filings_in_last_90_days'] = self._count_events_in_window(time_index, filing_dates, 90)
            
            # Form type features
            features_df = self._create_form_type_features(features_df, filings_data, time_index)
            
            # Item-based features
            features_df = self._create_item_features(features_df, filings_data, time_index)
            
            # High-impact filing indicators
            features_df = self._create_impact_features(features_df, filings_data, time_index)
            
            # Fill NaN values with safe defaults
            features_df['days_since_last_filing'] = features_df['days_since_last_filing'].fillna(999)
        
        return features_df
    
    def _load_sec_filings_data(self, symbol: str, date_range: Dict[str, str]) -> pd.DataFrame:
        """Load SEC filings data for a symbol."""
        filings_path = Path(f"storage/data/sec_edgar/{symbol}/{symbol}_8K_filings.csv")
        
        if not filings_path.exists():
            return pd.DataFrame()
        
        try:
            filings_df = pd.read_csv(filings_path)
            filings_df['filing_date'] = pd.to_datetime(filings_df['filing_date'])
            
            # Filter by date range
            start_date = pd.to_datetime(date_range['start'])
            end_date = pd.to_datetime(date_range['end'])
            mask = (filings_df['filing_date'] >= start_date) & (filings_df['filing_date'] <= end_date)
            filtered_df = filings_df.loc[mask].copy()
            
            self.logger.info(f"Loaded {len(filtered_df)} SEC filings for {symbol}")
            return filtered_df
            
        except Exception as e:
            self.logger.error(f"Error loading SEC filings data: {e}")
            return pd.DataFrame()
    
    def _create_form_type_features(self, features_df: pd.DataFrame, filings_data: pd.DataFrame, 
                                  time_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Create features based on form types (8-K, 8-K/A)."""
        # Count different form types
        form_8k_dates = filings_data[filings_data['form'] == '8-K']['filing_date'].tolist()
        form_8ka_dates = filings_data[filings_data['form'] == '8-K/A']['filing_date'].tolist()
        
        if form_8k_dates:
            features_df['days_since_8k'] = self._calculate_days_since_last_event(time_index, form_8k_dates)
            features_df['has_recent_8k'] = (features_df['days_since_8k'] <= 7).astype(int)
            features_df['days_since_8k'] = features_df['days_since_8k'].fillna(999)
        
        if form_8ka_dates:
            features_df['days_since_8ka'] = self._calculate_days_since_last_event(time_index, form_8ka_dates)
            features_df['has_recent_8ka'] = (features_df['days_since_8ka'] <= 7).astype(int)
            features_df['days_since_8ka'] = features_df['days_since_8ka'].fillna(999)
        
        return features_df
    
    def _create_item_features(self, features_df: pd.DataFrame, filings_data: pd.DataFrame, 
                             time_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Create features based on 8-K item categories."""
        # Parse items and create features for key categories
        item_categories = {}
        
        for _, row in filings_data.iterrows():
            if pd.notna(row['items']):
                items = str(row['items']).split(',')
                for item in items:
                    item = item.strip()
                    if item not in item_categories:
                        item_categories[item] = []
                    item_categories[item].append(row['filing_date'])
        
        # Create features for high-impact items
        for item in self.high_impact_items:
            if item in item_categories:
                col_name = f'days_since_item_{item.replace(".", "_")}'
                features_df[col_name] = self._calculate_days_since_last_event(
                    time_index, item_categories[item]
                )
                features_df[col_name] = features_df[col_name].fillna(999)
                
                # Binary flag for recent filings with this item
                recent_col = f'has_recent_item_{item.replace(".", "_")}'
                features_df[recent_col] = (features_df[col_name] <= 7).astype(int)
        
        return features_df
    
    def _create_impact_features(self, features_df: pd.DataFrame, filings_data: pd.DataFrame, 
                               time_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Create features indicating potential market impact."""
        # High-impact filing dates (earnings-related and material agreements)
        high_impact_dates = []
        
        for _, row in filings_data.iterrows():
            if pd.notna(row['items']):
                items = str(row['items']).split(',')
                items = [item.strip() for item in items]
                
                # Check if any high-impact items are present
                if any(item in self.high_impact_items for item in items):
                    high_impact_dates.append(row['filing_date'])
        
        if high_impact_dates:
            features_df['days_since_high_impact_filing'] = self._calculate_days_since_last_event(
                time_index, high_impact_dates
            )
            features_df['has_recent_high_impact'] = (
                features_df['days_since_high_impact_filing'] <= 3
            ).astype(int)
            features_df['days_since_high_impact_filing'] = features_df['days_since_high_impact_filing'].fillna(999)
        
        # Amendment filings (8-K/A) - could indicate corrections or additional info
        amendment_dates = filings_data[filings_data['form'] == '8-K/A']['filing_date'].tolist()
        if amendment_dates:
            features_df['has_recent_amendment'] = self._has_recent_event(time_index, amendment_dates, 7)
        
        return features_df
    
    def _calculate_days_since_last_event(self, time_index: pd.DatetimeIndex, 
                                        event_dates: List[pd.Timestamp]) -> pd.Series:
        """Calculate days since last event for each timestamp."""
        days_since_event = pd.Series(index=time_index, dtype=float)
        
        # Normalize timezones
        normalized_events = []
        for event in event_dates:
            if hasattr(event, 'tz') and event.tz is not None:
                normalized_events.append(event.tz_convert('UTC').tz_localize(None))
            else:
                normalized_events.append(pd.to_datetime(event).tz_localize(None))
        
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
    
    def _count_events_in_window(self, time_index: pd.DatetimeIndex, 
                               event_dates: List[pd.Timestamp], window_days: int) -> pd.Series:
        """Count events in a rolling window for each timestamp."""
        event_counts = pd.Series(index=time_index, dtype=int)
        
        # Normalize timezones
        normalized_events = []
        for event in event_dates:
            if hasattr(event, 'tz') and event.tz is not None:
                normalized_events.append(event.tz_convert('UTC').tz_localize(None))
            else:
                normalized_events.append(pd.to_datetime(event).tz_localize(None))
        
        # Ensure time_index is also timezone naive
        if time_index.tz is not None:
            time_index_normalized = time_index.tz_convert('UTC').tz_localize(None)
        else:
            time_index_normalized = time_index
        
        for i, timestamp in enumerate(time_index_normalized):
            window_start = timestamp - timedelta(days=window_days)
            events_in_window = [
                event for event in normalized_events 
                if window_start <= event <= timestamp
            ]
            event_counts.iloc[i] = len(events_in_window)
        
        return event_counts
    
    def _has_recent_event(self, time_index: pd.DatetimeIndex, 
                         event_dates: List[pd.Timestamp], days: int) -> pd.Series:
        """Check if there was an event within specified days."""
        days_since = self._calculate_days_since_last_event(time_index, event_dates)
        return (days_since <= days).astype(int)
    
    def _categorize_proximity_safe(self, days_since_event: pd.Series) -> pd.Series:
        """Categorize proximity using days SINCE last event (no lookahead)."""
        proximity = pd.Series(index=days_since_event.index, dtype=str)
        
        # Categories based on days SINCE last event
        proximity[days_since_event <= 1] = 'immediate_post'
        proximity[(days_since_event > 1) & (days_since_event <= 7)] = 'recent_post'
        proximity[(days_since_event > 7) & (days_since_event <= 30)] = 'medium_post' 
        proximity[(days_since_event > 30) & (days_since_event <= 90)] = 'long_post'
        proximity[days_since_event > 90] = 'long_term'
        proximity[days_since_event >= 999] = 'unknown'
        
        return proximity
    
    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN values with appropriate defaults for different feature types."""
        filled_df = df.copy()
        
        # Days-based features - fill with large number indicating "far away"
        days_columns = [col for col in df.columns if 'days_since' in col]
        for col in days_columns:
            filled_df[col] = filled_df[col].fillna(999)
        
        # Count-based features - fill with 0
        count_columns = [col for col in df.columns if 'filings_in' in col]
        for col in count_columns:
            filled_df[col] = filled_df[col].fillna(0)
        
        # Binary flags - fill with 0
        binary_columns = [col for col in df.columns if any(keyword in col for keyword in 
                         ['has_', '_zone', '_recent'])]
        for col in binary_columns:
            filled_df[col] = filled_df[col].fillna(0)
        
        # Categorical features - fill with 'long_term'
        categorical_columns = [col for col in df.columns if 'proximity_zone' in col]
        for col in categorical_columns:
            filled_df[col] = filled_df[col].fillna('long_term')
        
        return filled_df
    
    def _load_market_data_alpaca(self, symbol: str, timeframe: str, date_range: Dict[str, str]) -> pd.DataFrame:
        """
        Load market data from Alpaca CSV files with timeframe mapping.
        
        Args:
            symbol: Trading symbol
            timeframe: Time granularity (e.g., '5m' maps to '5min' directory)
            date_range: Date range to load
            
        Returns:
            pd.DataFrame: Time-indexed market data
        """
        from pathlib import Path
        import glob
        
        # Map standard timeframes to actual directory names
        # Note: Using directory names as they actually exist in storage
        timeframe_mapping = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30min',  # This one uses 'min' format
            '1h': '1h',
            '4h': '4h',
            '1d': '1d'
        }
        
        # Get the actual directory name
        actual_timeframe = timeframe_mapping.get(timeframe, timeframe)
        
        # Construct path pattern for market data
        data_dir = Path(f"storage/data/alpaca/{symbol}/raw/{actual_timeframe}")
        
        if not data_dir.exists():
            raise FileNotFoundError(f"Market data directory not found: {data_dir}")
        
        # Find CSV files in date range
        csv_files = glob.glob(str(data_dir / "*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")
        
        # Load and combine CSV files
        dfs = []
        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path)
                
                # Handle both 'Date' (raw datasets) and 'date' (processed datasets)
                date_column = None
                if 'Date' in df.columns:
                    date_column = 'Date'
                elif 'date' in df.columns:
                    date_column = 'date'
                else:
                    self.logger.warning(f"No 'Date' or 'date' column found in {file_path}")
                    continue
                
                df[date_column] = pd.to_datetime(df[date_column])
                
                # Standardize column name to 'Date' for consistency
                if date_column == 'date':
                    df.rename(columns={'date': 'Date'}, inplace=True)
                
                dfs.append(df)
            except Exception as e:
                self.logger.warning(f"Error loading {file_path}: {e}")
        
        if not dfs:
            raise ValueError("No valid market data files could be loaded")
        
        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df = combined_df.sort_values('Date').drop_duplicates(subset=['Date'])
        
        # Filter by date range
        start_date = pd.to_datetime(date_range['start'])
        end_date = pd.to_datetime(date_range['end'])
        mask = (combined_df['Date'] >= start_date) & (combined_df['Date'] <= end_date)
        filtered_df = combined_df.loc[mask].copy()
        
        # Set index
        filtered_df.set_index('Date', inplace=True)
        
        self.logger.info(f"Loaded {len(filtered_df)} market data points for {symbol}")
        return filtered_df