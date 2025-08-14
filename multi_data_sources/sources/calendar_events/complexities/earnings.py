import json
from typing import Any, Optional, Dict
import pandas as pd
import yfinance as yf
from datetime import datetime
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class Earnings:
    """
    Earnings complexity component for calendar events.
    
    Handles earnings-related calendar events including:
    - Quarterly earnings announcements and dates
    - Days until next earnings (proximity analysis)
    - Earnings surprises and guidance
    - Analyst price targets and revisions
    - Tuesday/Friday announcement patterns
    - Post-earnings drift analysis periods
    
    This component focuses on timing-based inefficiencies around earnings events
    that institutions may struggle to exploit due to organizational constraints.
    """
    
    def __init__(self, symbol: str, stock: bool = False, from_date: Optional[str] = None, 
                 to_date: Optional[str] = None, purpose: str = "historical"):
        """
        Initialize earnings complexity component.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            stock: Whether this is a stock symbol (vs other asset types)
            from_date: Start date for historical data (YYYY-MM-DD)
            to_date: End date for historical data (YYYY-MM-DD)
            purpose: 'historical' or 'live' data collection
        """
        self.symbol = symbol
        self.stock = stock
        self.from_date = from_date
        self.to_date = to_date
        self.purpose = purpose
        
        # Initialize yfinance ticker
        self.ticker = yf.Ticker(self.symbol)
        
        # Storage configuration - using centralized config
        from config.config import get_calendar_events_component_dir
        self.storage_base = Path(get_calendar_events_component_dir(self.symbol, 'earnings'))
        
        logger.info(f"Initialized Earnings component for {self.symbol} ({self.purpose} mode)")
    
    def get_historical_data(self) -> Dict[str, Any]:
        """
        Fetch comprehensive historical earnings data.
        
        Returns:
            Dictionary containing:
            - earnings_calendar: Upcoming/historical earnings dates
            - earnings_history: Historical earnings with surprises
            - analyst_targets: Price targets and revisions
            - earnings_features: Engineered timing features
            - metadata: Collection timestamp and parameters
        """
        logger.info(f"Fetching historical earnings data for {self.symbol}")
        
        try:
            result = {
                'earnings_calendar': self._get_earnings_calendar(),
                'historical_earnings_dates': self._get_historical_earnings_dates(),
                'earnings_history': self._get_earnings_history(),
                'analyst_targets': self._get_analyst_targets(),
                'earnings_features': self._engineer_earnings_features(),
                'historical_analysis': self._analyze_historical_patterns(),
                'metadata': {
                    'symbol': self.symbol,
                    'from_date': self.from_date,
                    'to_date': self.to_date,
                    'collection_timestamp': datetime.now().isoformat(),
                    'purpose': self.purpose
                }
            }
            
            # Save the data to centralized storage
            saved_paths = self._save_earnings_data(result)
            result['saved_paths'] = saved_paths
            
            logger.info(f"Successfully collected earnings data for {self.symbol}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to fetch earnings data for {self.symbol}: {e}")
            raise
    
    def subscribe_live_data(self) -> Dict[str, Any]:
        """
        Set up live monitoring for earnings-related events.
        
        Returns:
            Dictionary with subscription info and current earnings proximity
        """
        logger.info(f"Setting up live earnings monitoring for {self.symbol}")
        
        try:
            # Get current earnings proximity
            current_features = self._engineer_earnings_features()
            
            return {
                'subscription_type': 'earnings_monitor',
                'symbol': self.symbol,
                'current_earnings_proximity': current_features.get('days_to_next_earnings'),
                'next_earnings_date': current_features.get('next_earnings_date'),
                'monitoring_features': [
                    'days_to_next_earnings',
                    'earnings_announcement_day',
                    'post_earnings_drift_period',
                    'analyst_revision_activity'
                ],
                'update_frequency': 'daily',
                'subscription_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to set up earnings monitoring for {self.symbol}: {e}")
            raise
    
    def _get_earnings_calendar(self) -> pd.DataFrame:
        """
        Get earnings calendar with upcoming and historical dates.
        """
        try:
            calendar_data = self.ticker.calendar
            if calendar_data is not None:
                # Handle different yfinance return formats
                if isinstance(calendar_data, dict):
                    # Convert dict to DataFrame
                    df = pd.DataFrame([calendar_data]) if calendar_data else pd.DataFrame()
                elif hasattr(calendar_data, 'empty'):
                    # It's already a DataFrame
                    df = calendar_data if not calendar_data.empty else pd.DataFrame()
                else:
                    # Try to convert to DataFrame
                    df = pd.DataFrame(calendar_data) if calendar_data else pd.DataFrame()
                
                if not df.empty and len(df.columns) > 0:
                    # Add day-of-week analysis if we have date columns
                    date_columns = [col for col in df.columns if 'date' in col.lower() or 'earnings' in col.lower()]
                    if date_columns:
                        date_col = date_columns[0]
                        # Handle different date formats in earnings calendar
                        if date_col in df.columns:
                            # Extract dates from lists if they're stored as lists
                            if df[date_col].apply(lambda x: isinstance(x, list)).any():
                                df[date_col] = df[date_col].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)
                            
                            df['announcement_day_of_week'] = pd.to_datetime(
                                df[date_col], errors='coerce'
                            ).dt.day_name()
                            
                            # Flag Tuesday/Friday pattern (common for earnings)
                            df['is_tuesday_friday'] = df['announcement_day_of_week'].isin(['Tuesday', 'Friday'])
                    else:
                        # If no date columns found, set default values
                        df['announcement_day_of_week'] = 'Unknown'
                        df['is_tuesday_friday'] = False
                
                return df
            else:
                logger.warning(f"No earnings calendar data available for {self.symbol}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching earnings calendar: {e}")
            return pd.DataFrame()
    
    def _get_historical_earnings_dates(self) -> pd.DataFrame:
        """
        Get historical earnings announcement dates with surprises.
        This is the MOST RELIABLE method for historical earnings data in yfinance.
        """
        try:
            # This method gives us 2-3 years of historical earnings dates
            earnings_dates = self.ticker.earnings_dates
            
            if earnings_dates is not None:
                # Handle different return formats
                if isinstance(earnings_dates, dict):
                    df = pd.DataFrame([earnings_dates]) if earnings_dates else pd.DataFrame()
                elif hasattr(earnings_dates, 'empty'):
                    df = earnings_dates if not earnings_dates.empty else pd.DataFrame()
                else:
                    df = pd.DataFrame(earnings_dates) if earnings_dates else pd.DataFrame()
                
                if not df.empty:
                    # Filter by date range if specified
                    if self.from_date and self.to_date:
                        df_filtered = df[
                            (df.index >= self.from_date) & 
                            (df.index <= self.to_date)
                        ]
                    else:
                        df_filtered = df
                    
                    if not df_filtered.empty:
                        # Add engineered features for historical analysis
                        df_filtered = self._add_historical_earnings_features(df_filtered)
                        logger.info(f"Found {len(df_filtered)} historical earnings dates for {self.symbol}")
                        return df_filtered
            
            logger.warning(f"No historical earnings dates available for {self.symbol}")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching historical earnings dates: {e}")
            return pd.DataFrame()
    
    def _add_historical_earnings_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add engineered features to historical earnings data for inefficiency analysis.
        """
        try:
            # Add day of week for announcement timing patterns
            df['announcement_day_of_week'] = df.index.day_name()
            df['is_tuesday_friday'] = df['announcement_day_of_week'].isin(['Tuesday', 'Friday'])
            
            # Calculate earnings surprise if we have EPS data
            if 'EPS Estimate' in df.columns and 'Reported EPS' in df.columns:
                df['surprise_amount'] = df['Reported EPS'] - df['EPS Estimate']
                df['surprise_pct'] = (df['surprise_amount'] / df['EPS Estimate'].abs()) * 100
                df['beat_expectations'] = df['surprise_amount'] > 0
                df['large_surprise'] = abs(df['surprise_pct']) > 5  # >5% surprise
            
            # Add quarters for seasonal analysis
            df['quarter'] = df.index.quarter
            df['year'] = df.index.year
            
            # Days between earnings (for consistency analysis)
            df_sorted = df.sort_index()
            # Calculate days since last earnings, handling NaN for the first record
            date_diffs = df_sorted.index.to_series().diff().dt.days
            # Fill NaN with 0 for the first record or use a reasonable default
            df_sorted['days_since_last_earnings'] = date_diffs.fillna(0)
            
            return df_sorted
            
        except Exception as e:
            logger.error(f"Error adding historical earnings features: {e}")
            return df
    
    def _get_earnings_history(self) -> pd.DataFrame:
        """
        Get historical earnings with surprises and guidance.
        """
        try:
            # Get quarterly earnings data
            earnings = self.ticker.quarterly_earnings
            if earnings is not None and not earnings.empty:
                # Calculate earnings surprise if estimate data is available
                if 'Actual' in earnings.columns and 'Estimate' in earnings.columns:
                    earnings['surprise_amount'] = earnings['Actual'] - earnings['Estimate']
                    earnings['surprise_pct'] = (earnings['surprise_amount'] / earnings['Estimate'].abs()) * 100
                    earnings['beat_expectations'] = earnings['surprise_amount'] > 0
                
                return earnings
            else:
                logger.warning(f"No earnings history available for {self.symbol}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching earnings history: {e}")
            return pd.DataFrame()
    
    def _get_analyst_targets(self) -> pd.DataFrame:
        """
        Get analyst price targets and recent revisions.
        """
        try:
            targets = self.ticker.analyst_price_targets
            if targets is not None:
                # Handle different yfinance return formats
                if isinstance(targets, dict):
                    df = pd.DataFrame([targets]) if targets else pd.DataFrame()
                elif hasattr(targets, 'empty'):
                    df = targets if not targets.empty else pd.DataFrame()
                else:
                    df = pd.DataFrame(targets) if targets else pd.DataFrame()
                
                return df
            else:
                logger.warning(f"No analyst targets available for {self.symbol}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching analyst targets: {e}")
            return pd.DataFrame()
    
    def _engineer_earnings_features(self, as_of_date: str = None) -> Dict[str, Any]:
        """
        Engineer timing-based features around earnings events.
        
        CRITICAL: This method now uses ONLY historical information available
        at the specified as_of_date to prevent lookahead bias.
        
        Args:
            as_of_date: The reference date for feature calculation (YYYY-MM-DD)
                       If None, uses current date (for live trading only)
        
        These features capture the market inefficiencies around earnings timing
        that your strategy aims to exploit WITHOUT using future information.
        """
        features = {}
        
        try:
            # Determine reference date - critical for temporal integrity
            if as_of_date:
                reference_date = pd.to_datetime(as_of_date)
            else:
                # Only use current date for live trading, not historical analysis
                reference_date = pd.to_datetime(datetime.now())
                logger.warning("Using current date - ensure this is for live trading only!")
            
            # Get historical earnings dates that occurred BEFORE reference_date
            historical_dates = self._get_historical_earnings_dates()
            
            if not historical_dates.empty:
                # Filter to only earnings dates before reference_date (no lookahead)
                past_earnings = historical_dates[historical_dates.index < reference_date]
                
                if not past_earnings.empty:
                    # TEMPORAL INTEGRITY: Only use PAST earnings information
                    
                    # Days since LAST earnings (always available historically)
                    last_earnings_date = past_earnings.index.max()
                    days_since_last = (reference_date - last_earnings_date).days
                    features['days_since_last_earnings'] = days_since_last
                    features['last_earnings_date'] = last_earnings_date.strftime('%Y-%m-%d')
                    
                    # Historical earnings patterns (no future knowledge)
                    if len(past_earnings) >= 2:
                        # Average days between earnings (for rough estimation)
                        earning_intervals = past_earnings.index.to_series().diff().dt.days.dropna()
                        avg_interval = earning_intervals.mean()
                        features['avg_earnings_interval_days'] = avg_interval
                        
                        # Estimated next earnings window (rough approximation only)
                        if pd.notna(avg_interval) and avg_interval > 0:
                            estimated_next = last_earnings_date + pd.Timedelta(days=avg_interval)
                            days_to_estimated = (estimated_next - reference_date).days
                            
                            # Only include if reasonable (30-120 days typical)
                            if 30 <= days_to_estimated <= 120:
                                features['estimated_days_to_next_earnings'] = days_to_estimated
                                features['earnings_proximity_zone'] = self._classify_earnings_proximity_safe(days_since_last, avg_interval)
                            else:
                                features['earnings_proximity_zone'] = 'uncertain'
                    
                    # Earnings timing patterns from historical data
                    if 'announcement_day_of_week' in past_earnings.columns:
                        recent_pattern = past_earnings['announcement_day_of_week'].iloc[-3:].mode()
                        if len(recent_pattern) > 0:
                            features['typical_announcement_day'] = recent_pattern.iloc[0]
                            features['typically_tuesday_friday'] = recent_pattern.iloc[0] in ['Tuesday', 'Friday']
                    
                    # Post-earnings drift analysis (using past data only)
                    features['in_post_earnings_drift'] = self._is_post_earnings_drift_period_safe(days_since_last)
                    
                else:
                    # No past earnings data available
                    features['days_since_last_earnings'] = 999  # Unknown/very long
                    features['earnings_proximity_zone'] = 'unknown'
            
            # Earnings season indicator (based on reference date, not current date)
            features['earnings_season'] = self._is_earnings_season_at_date(reference_date)
            
            logger.debug(f"Engineered {len(features)} temporally-safe earnings features for {self.symbol}")
            return features
            
        except Exception as e:
            logger.error(f"Error engineering earnings features: {e}")
            return {}
    
    def _analyze_historical_patterns(self) -> Dict[str, Any]:
        """
        Analyze historical earnings patterns for market inefficiencies.
        This is where we identify the patterns institutions miss.
        """
        try:
            historical_dates = self._get_historical_earnings_dates()
            patterns = {}
            
            if not historical_dates.empty:
                # Day-of-week patterns (institutional trading constraints)
                day_counts = historical_dates['announcement_day_of_week'].value_counts()
                patterns['day_of_week_distribution'] = day_counts.to_dict()
                patterns['tuesday_friday_ratio'] = (
                    day_counts.get('Tuesday', 0) + day_counts.get('Friday', 0)
                ) / len(historical_dates) if len(historical_dates) > 0 else 0
                
                # Earnings surprise patterns
                if 'beat_expectations' in historical_dates.columns:
                    patterns['historical_beat_rate'] = historical_dates['beat_expectations'].mean()
                    patterns['avg_surprise_pct'] = historical_dates['surprise_pct'].mean()
                    patterns['surprise_volatility'] = historical_dates['surprise_pct'].std()
                
                # Quarterly consistency
                if 'quarter' in historical_dates.columns:
                    quarterly_counts = historical_dates['quarter'].value_counts().sort_index()
                    patterns['quarterly_distribution'] = quarterly_counts.to_dict()
                
                # Timing consistency (for drift analysis)
                if 'days_since_last_earnings' in historical_dates.columns:
                    avg_days = historical_dates['days_since_last_earnings'].mean()
                    patterns['avg_days_between_earnings'] = avg_days
                    patterns['earnings_timing_consistency'] = (
                        historical_dates['days_since_last_earnings'].std() / avg_days
                    ) if avg_days > 0 else None
                
                logger.info(f"Analyzed {len(patterns)} historical patterns for {self.symbol}")
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing historical patterns: {e}")
            return {}
    
    def get_price_data_for_analysis(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Get historical price data aligned with earnings dates for inefficiency analysis.
        This combines earnings events with OHLCV data to find patterns.
        """
        try:
            import yfinance as yf
            
            start = start_date or self.from_date or '2022-01-01'
            end = end_date or self.to_date or datetime.now().strftime('%Y-%m-%d')
            
            # Download price data
            price_data = yf.download(self.symbol, start=start, end=end)
            
            if not price_data.empty:
                # Calculate returns for drift analysis
                price_data['daily_return'] = price_data['Close'].pct_change()
                price_data['log_return'] = np.log(price_data['Close'] / price_data['Close'].shift(1))
                
                # Volatility measures
                price_data['volatility_20d'] = price_data['daily_return'].rolling(20).std()
                price_data['high_low_range'] = (price_data['High'] - price_data['Low']) / price_data['Close']
                
                logger.info(f"Retrieved {len(price_data)} days of price data for {self.symbol}")
                return price_data
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error getting price data: {e}")
            return pd.DataFrame()
    
    def _save_earnings_data(self, data: Dict[str, Any]) -> Dict[str, str]:
        """
        Save earnings data to centralized storage following the project's structure pattern.
        
        Structure: storage/multi_data_sources/calendar_events/earnings/{symbol}/
        - raw/ : Raw earnings data (JSON format)
        - processed/ : Processed features (CSV format) 
        - analysis/ : Pattern analysis results (JSON format)
        
        Returns:
            Dictionary with paths to saved files
        """
        try:
            # Create directory structure using centralized config
            from config.config import (get_calendar_events_raw_dir, get_calendar_events_processed_dir, 
                                     get_calendar_events_analysis_dir)
            
            # Create subdirectories
            raw_dir = Path(get_calendar_events_raw_dir(self.symbol, 'earnings'))
            processed_dir = Path(get_calendar_events_processed_dir(self.symbol, 'earnings'))
            analysis_dir = Path(get_calendar_events_analysis_dir(self.symbol, 'earnings'))
            
            for directory in [raw_dir, processed_dir, analysis_dir]:
                directory.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with date range
            date_suffix = f"{self.from_date}_{self.to_date}" if self.from_date and self.to_date else datetime.now().strftime("%Y-%m-%d")
            
            saved_paths = {}
            
            # Save raw data (complete earnings data as JSON)
            raw_filename = f"{date_suffix}_earnings_data.json"
            raw_path = raw_dir / raw_filename
            
            # Prepare data for JSON serialization (handle pandas timestamps)
            def convert_timestamps(df):
                """Convert pandas timestamps to strings for JSON serialization"""
                if df.empty:
                    return []
                # Reset index to convert timestamps to regular columns
                df_copy = df.copy()
                if isinstance(df_copy.index, pd.DatetimeIndex):
                    df_copy = df_copy.reset_index()
                    df_copy.columns = [str(col) for col in df_copy.columns]
                return df_copy.to_dict('records')
            
            json_data = {
                'metadata': data['metadata'],
                'earnings_calendar': convert_timestamps(data['earnings_calendar']),
                'historical_earnings_dates': convert_timestamps(data['historical_earnings_dates']),
                'earnings_history': convert_timestamps(data['earnings_history']),
                'analyst_targets': convert_timestamps(data['analyst_targets']),
                'earnings_features': {str(k): str(v) for k, v in data['earnings_features'].items()},
                'historical_analysis': data['historical_analysis']
            }
            
            with open(raw_path, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            saved_paths['raw_data'] = str(raw_path)
            
            # Save processed features (CSV for easy ML usage)
            if not data['historical_earnings_dates'].empty:
                processed_filename = f"{date_suffix}_earnings_features.csv"
                processed_path = processed_dir / processed_filename
                data['historical_earnings_dates'].to_csv(processed_path)
                saved_paths['processed_features'] = str(processed_path)
            
            # Save analysis patterns (JSON for easy consumption)
            if data['historical_analysis']:
                analysis_filename = f"{date_suffix}_patterns.json"
                analysis_path = analysis_dir / analysis_filename
                
                # Include additional analysis metadata
                analysis_data = {
                    'symbol': self.symbol,
                    'analysis_timestamp': datetime.now().isoformat(),
                    'date_range': {'from': self.from_date, 'to': self.to_date},
                    'patterns': data['historical_analysis'],
                    'summary': {
                        'total_earnings_events': len(data['historical_earnings_dates']) if not data['historical_earnings_dates'].empty else 0,
                        'data_quality': 'high' if data['historical_analysis'] else 'limited'
                    }
                }
                
                with open(analysis_path, 'w') as f:
                    json.dump(analysis_data, f, indent=2, default=str)
                saved_paths['analysis'] = str(analysis_path)
            
            logger.info(f"Earnings data saved for {self.symbol}: {len(saved_paths)} files")
            return saved_paths
            
        except Exception as e:
            logger.error(f"Error saving earnings data for {self.symbol}: {e}")
            return {}
    
    def load_saved_earnings_data(self, date_range: str = None) -> Dict[str, Any]:
        """
        Load previously saved earnings data from storage.
        
        Args:
            date_range: Specific date range to load (e.g., "2022-01-01_2025-01-01")
                       If None, loads the most recent data
        
        Returns:
            Dictionary with loaded earnings data
        """
        try:
            from config.config import get_calendar_events_raw_dir
            symbol_dir = Path(get_calendar_events_raw_dir(self.symbol, 'earnings'))
            
            if not symbol_dir.exists():
                logger.warning(f"No saved earnings data found for {self.symbol}")
                return {}
            
            # Find the appropriate file
            if date_range:
                filename = f"{date_range}_earnings_data.json"
                file_path = symbol_dir / filename
            else:
                # Get the most recent file
                json_files = list(symbol_dir.glob("*_earnings_data.json"))
                if not json_files:
                    logger.warning(f"No earnings data files found for {self.symbol}")
                    return {}
                file_path = max(json_files, key=lambda x: x.stat().st_mtime)
            
            if file_path.exists():
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Convert back to DataFrames
                result = {
                    'metadata': data['metadata'],
                    'earnings_calendar': pd.DataFrame(data['earnings_calendar']),
                    'historical_earnings_dates': pd.DataFrame.from_dict(data['historical_earnings_dates'], orient='index'),
                    'earnings_history': pd.DataFrame(data['earnings_history']),
                    'analyst_targets': pd.DataFrame(data['analyst_targets']),
                    'earnings_features': data['earnings_features'],
                    'historical_analysis': data['historical_analysis']
                }
                
                logger.info(f"Successfully loaded saved earnings data for {self.symbol}")
                return result
            else:
                logger.warning(f"Earnings data file not found: {file_path}")
                return {}
                
        except Exception as e:
            logger.error(f"Error loading saved earnings data for {self.symbol}: {e}")
            return {}
    
    def _classify_earnings_proximity_safe(self, days_since_last: int, avg_interval: float) -> str:
        """
        Classify earnings proximity using ONLY historical information.
        Uses days since last earnings and typical interval patterns.
        """
        # Use historical patterns, not future knowledge
        if days_since_last <= 7:
            return 'immediate_post_earnings'
        elif days_since_last <= 30:
            return 'recent_post_earnings'
        elif avg_interval and days_since_last >= (avg_interval * 0.8):
            return 'approaching_next_earnings'
        elif avg_interval and days_since_last >= (avg_interval * 0.6):
            return 'mid_cycle'
        else:
            return 'early_cycle'
    
    # REMOVED: _classify_earnings_proximity - caused lookahead bias
    # Use _classify_earnings_proximity_safe instead
    
    def _is_post_earnings_drift_period_safe(self, days_since_last: int) -> bool:
        """
        Check if we're in a post-earnings drift period using only historical data.
        """
        try:
            # Post-earnings drift typically lasts 1-60 days after announcement
            return 1 <= days_since_last <= 60
        except:
            return False
    
    # REMOVED: _is_post_earnings_drift_period - use safe version
    
    def _is_earnings_season_at_date(self, reference_date: pd.Timestamp) -> bool:
        """
        Determine if reference_date is in earnings season (typically Jan, Apr, Jul, Oct).
        """
        reference_month = reference_date.month
        earnings_months = [1, 4, 7, 10]  # Typical earnings season months
        return reference_month in earnings_months
    
    # REMOVED: _is_earnings_season - use _is_earnings_season_at_date instead
    
    def _count_recent_revisions(self) -> int:
        """
        Count recent analyst revisions (proxy for sentiment changes).
        """
        try:
            # This would require more detailed analyst data
            # For now, return 0 - can be enhanced with additional data sources
            return 0
        except:
            return 0