import json
from typing import Any, Optional, Dict
import pandas as pd
import yfinance as yf
from datetime import datetime
import numpy as np
import logging

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
            
            logger.info(f"Successfully collected earnings data for {self.symbol}")
            print(result)
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
                        df['announcement_day_of_week'] = pd.to_datetime(
                            df[date_col], errors='coerce'
                        ).dt.day_name()
                        
                        # Flag Tuesday/Friday pattern (common for earnings)
                        df['is_tuesday_friday'] = df['announcement_day_of_week'].isin(['Tuesday', 'Friday'])
                
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
            df_sorted['days_since_last_earnings'] = df_sorted.index.to_series().diff().dt.days
            
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
    
    def _engineer_earnings_features(self) -> Dict[str, Any]:
        """
        Engineer timing-based features around earnings events.
        
        These features capture the market inefficiencies around earnings timing
        that your strategy aims to exploit.
        """
        features = {}
        
        try:
            # Get next earnings date
            calendar = self._get_earnings_calendar()
            if not calendar.empty and 'Earnings Date' in calendar.columns:
                next_earnings = calendar['Earnings Date'].iloc[0] if len(calendar) > 0 else None
                
                if next_earnings:
                    next_date = pd.to_datetime(next_earnings)
                    current_date = datetime.now()
                    
                    # Days to next earnings (key timing feature)
                    days_to_earnings = (next_date - current_date).days
                    features['days_to_next_earnings'] = days_to_earnings
                    features['next_earnings_date'] = next_date.strftime('%Y-%m-%d')
                    
                    # Earnings proximity zones (institutions often have blackout periods)
                    features['earnings_proximity_zone'] = self._classify_earnings_proximity(days_to_earnings)
                    
                    # Day of week for next earnings (Tuesday/Friday analysis)
                    features['next_earnings_day_of_week'] = next_date.strftime('%A')
                    features['is_tuesday_friday_announcement'] = features['next_earnings_day_of_week'] in ['Tuesday', 'Friday']
                    
                    # Post-earnings drift period indicator
                    features['in_post_earnings_drift'] = self._is_post_earnings_drift_period()
            
            # Earnings season indicator
            features['earnings_season'] = self._is_earnings_season()
            
            # Recent analyst activity
            features['recent_analyst_revisions'] = self._count_recent_revisions()
            
            logger.debug(f"Engineered {len(features)} earnings features for {self.symbol}")
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
    
    def _classify_earnings_proximity(self, days_to_earnings: int) -> str:
        """
        Classify proximity to earnings for institutional blackout analysis.
        """
        if days_to_earnings < 0:
            return 'post_earnings'
        elif days_to_earnings <= 2:
            return 'immediate_pre_earnings'
        elif days_to_earnings <= 7:
            return 'near_term_pre_earnings'
        elif days_to_earnings <= 21:
            return 'mid_term_pre_earnings'
        else:
            return 'distant_from_earnings'
    
    def _is_post_earnings_drift_period(self) -> bool:
        """
        Check if we're in a post-earnings drift period (typically 1-60 days after earnings).
        """
        try:
            # This would need historical earnings dates to determine
            # For now, return False - can be enhanced with more data
            return False
        except:
            return False
    
    def _is_earnings_season(self) -> bool:
        """
        Determine if we're in earnings season (typically Jan, Apr, Jul, Oct).
        """
        current_month = datetime.now().month
        earnings_months = [1, 4, 7, 10]  # Typical earnings season months
        return current_month in earnings_months
    
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