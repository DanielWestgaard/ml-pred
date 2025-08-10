import json
from typing import Any, Optional, Dict, List
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class OptionsExpiration:
    """
    Options expiration complexity component for calendar events.
    
    Focuses on "Option expiration proximity effects on small caps" - identifying 
    market inefficiencies around options expiration dates that institutions miss 
    due to their focus on large-scale opportunities.
    
    Key Strategy:
    - Small-cap stocks often show unique patterns around options expiration
    - Institutions miss these micro-patterns due to position sizing constraints
    - Options activity can create temporary price distortions in smaller stocks
    - Post-expiration "cleanup" effects as market makers unwind positions
    
    This component analyzes stock price behavior patterns around known expiration
    dates rather than historical options pricing (which yfinance doesn't provide).
    """
    
    def __init__(self, symbol: str, stock: bool = False, from_date: Optional[str] = None, 
                 to_date: Optional[str] = None, purpose: str = "historical"):
        """
        Initialize options expiration complexity component.
        
        Args:
            symbol: Stock symbol (e.g., 'CLOV', 'TLRY')
            stock: Whether this is a stock symbol (vs other asset types)
            from_date: Start date for historical analysis (YYYY-MM-DD)
            to_date: End date for historical analysis (YYYY-MM-DD)
            purpose: 'historical' or 'live' data collection
        """
        self.symbol = symbol
        self.stock = stock
        self.from_date = from_date
        self.to_date = to_date
        self.purpose = purpose
        
        # Initialize yfinance ticker
        self.ticker = yf.Ticker(self.symbol)
        
        # Storage configuration
        self.storage_base = Path("storage/multi_data_sources/calendar_events/options_expiration")  # TODO: add hard coded path to config/config.py
        
        logger.info(f"Initialized OptionsExpiration component for {self.symbol} ({self.purpose} mode)")
    
    def get_historical_data(self) -> Dict[str, Any]:
        """
        Fetch comprehensive historical options expiration data and analyze proximity effects.
        
        Returns:
            Dictionary containing:
            - current_options_data: Current options chains and expiration dates
            - historical_proximity_analysis: Stock behavior around past expirations
            - expiration_calendar: Upcoming and historical expiration dates
            - options_activity_metrics: Volume, open interest, and activity patterns
            - proximity_features: Engineered features for expiration effects
            - metadata: Collection timestamp and parameters
        """
        logger.info(f"Analyzing options expiration effects for {self.symbol}")
        
        try:
            result = {
                'current_options_data': self._get_current_options_data(),
                'expiration_calendar': self._get_expiration_calendar(),
                'historical_proximity_analysis': self._analyze_historical_proximity_effects(),
                'options_activity_metrics': self._calculate_activity_metrics(),
                'proximity_features': self._engineer_proximity_features(),
                'small_cap_analysis': self._analyze_small_cap_effects(),
                'metadata': {
                    'symbol': self.symbol,
                    'from_date': self.from_date,
                    'to_date': self.to_date,
                    'collection_timestamp': datetime.now().isoformat(),
                    'purpose': self.purpose
                }
            }
            
            # Save the data to centralized storage
            saved_paths = self._save_options_data(result)
            result['saved_paths'] = saved_paths
            
            logger.info(f"Successfully collected options expiration data for {self.symbol}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to fetch options expiration data for {self.symbol}: {e}")
            raise
    
    def subscribe_live_data(self) -> Dict[str, Any]:
        """
        Set up live monitoring for options expiration proximity effects.
        
        Returns:
            Dictionary with subscription info and current proximity status
        """
        logger.info(f"Setting up live options expiration monitoring for {self.symbol}")
        
        try:
            # Get current proximity status
            current_features = self._engineer_proximity_features()
            
            return {
                'subscription_type': 'options_expiration_monitor',
                'symbol': self.symbol,
                'days_to_next_expiration': current_features.get('days_to_next_expiration'),
                'next_expiration_date': current_features.get('next_expiration_date'),
                'current_options_activity': current_features.get('options_volume_ratio'),
                'monitoring_features': [
                    'days_to_next_expiration',
                    'pre_expiration_drift',
                    'options_volume_spike',
                    'put_call_ratio_anomaly',
                    'small_cap_expiration_effects'
                ],
                'update_frequency': 'daily',
                'subscription_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to set up options expiration monitoring for {self.symbol}: {e}")
            raise
    
    def _get_current_options_data(self) -> Dict[str, Any]:
        """
        Get current options chains and expiration dates.
        """
        try:
            # Get all available expiration dates
            expirations = self.ticker.options
            
            if not expirations:
                logger.warning(f"No options data available for {self.symbol}")
                return {'has_options': False, 'expirations': [], 'chains': {}}
            
            options_data = {
                'has_options': True,
                'expirations': list(expirations),
                'chains': {},
                'market_cap_category': self._determine_market_cap_category()
            }
            
            # Get options chains for near-term expirations (next 3 months)
            for exp_date in expirations[:6]:  # Limit to first 6 expirations
                try:
                    chain = self.ticker.option_chain(exp_date)
                    
                    # Convert to dictionaries for JSON serialization
                    options_data['chains'][exp_date] = {
                        'calls': chain.calls.to_dict('records') if not chain.calls.empty else [],
                        'puts': chain.puts.to_dict('records') if not chain.puts.empty else []
                    }
                except Exception as e:
                    logger.warning(f"Failed to get options chain for {exp_date}: {e}")
                    continue
            
            logger.info(f"Retrieved {len(expirations)} expiration dates for {self.symbol}")
            return options_data
            
        except Exception as e:
            logger.error(f"Error fetching current options data: {e}")
            return {'has_options': False, 'error': str(e)}
    
    def _get_expiration_calendar(self) -> List[Dict[str, Any]]:
        """
        Create a calendar of options expiration dates with metadata.
        """
        try:
            expirations = self.ticker.options
            if not expirations:
                return []
            
            calendar = []
            for exp_str in expirations:
                exp_date = datetime.strptime(exp_str, '%Y-%m-%d')
                
                calendar_entry = {
                    'expiration_date': exp_str,
                    'days_from_now': (exp_date.date() - datetime.now().date()).days,
                    'day_of_week': exp_date.strftime('%A'),
                    'is_monthly': self._is_monthly_expiration(exp_date),
                    'is_quarterly': self._is_quarterly_expiration(exp_date),
                    'expiration_type': self._classify_expiration_type(exp_date)
                }
                calendar.append(calendar_entry)
            
            return calendar
            
        except Exception as e:
            logger.error(f"Error creating expiration calendar: {e}")
            return []
    
    def _analyze_historical_proximity_effects(self) -> Dict[str, Any]:
        """
        Analyze stock price behavior around historical options expiration dates.
        This is the core analysis for finding small-cap inefficiencies.
        
        Since yfinance doesn't provide historical expiration dates, we use 
        algorithmic reconstruction based on standard options expiration patterns.
        """
        try:
            # Determine date range for analysis
            if self.from_date and self.to_date:
                start_date = pd.to_datetime(self.from_date)
                end_date = pd.to_datetime(self.to_date)
            else:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365)  # Look back 1 year
            
            # Get historical price data with extra buffer for volatility calculations
            buffer_days = 30
            price_start = start_date - timedelta(days=buffer_days)
            price_data = yf.download(self.symbol, start=price_start, end=end_date + timedelta(days=1))
            
            if price_data.empty:
                logger.warning(f"No price data available for {self.symbol}")
                return {}
            
            # Calculate daily returns and volatility
            price_data['daily_return'] = price_data['Close'].pct_change()
            price_data['volume_ratio'] = price_data['Volume'] / price_data['Volume'].rolling(20).mean()
            price_data['volatility_5d'] = price_data['daily_return'].rolling(5).std()
            
            # Generate historical expiration dates using algorithmic approach
            historical_expirations = self._get_historical_monthly_expirations(start_date, end_date)
            
            logger.info(f"Generated {len(historical_expirations)} historical expiration dates for analysis")
            
            # Analyze effects around each historical expiration
            historical_effects = []
            
            for exp_info in historical_expirations:
                exp_date = pd.to_datetime(exp_info['expiration_date']).date()
                
                # Find effect around this expiration
                effect = self._analyze_single_expiration_effect(exp_date, price_data)
                if effect:
                    # Add calculated expiration metadata
                    effect.update({
                        'expiration_type': exp_info.get('type', 'monthly'),
                        'is_quarterly': exp_info.get('is_quarterly', False),
                        'calculation_method': 'algorithmic_third_friday'
                    })
                    historical_effects.append(effect)
            
            # Calculate aggregate statistics
            if historical_effects:
                aggregate_stats = self._calculate_aggregate_expiration_stats(historical_effects)
                # Add pattern strength analysis
                aggregate_stats.update(self._analyze_pattern_strength(historical_effects))
            else:
                aggregate_stats = {}
            
            return {
                'individual_expirations': historical_effects,
                'aggregate_statistics': aggregate_stats,
                'analysis_period': {
                    'start': start_date.strftime('%Y-%m-%d'),
                    'end': end_date.strftime('%Y-%m-%d'),
                    'total_expirations_analyzed': len(historical_effects),
                    'expiration_calculation_method': 'algorithmic_monthly_third_friday'
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing historical proximity effects: {e}")
            return {}
    
    def _analyze_single_expiration_effect(self, exp_date, price_data) -> Optional[Dict[str, Any]]:
        """
        Analyze stock behavior around a single expiration date.
        """
        try:
            # Find closest trading day to expiration
            closest_idx = None
            min_diff = float('inf')
            
            for idx, date in enumerate(price_data.index):
                diff = abs((date.date() - exp_date).days)
                if diff < min_diff:
                    min_diff = diff
                    closest_idx = idx
            
            if closest_idx is None or min_diff > 3:  # Skip if no close trading day
                return None
            
            # Analyze pre-expiration period (5 days before)
            pre_start = max(0, closest_idx - 5)
            pre_end = closest_idx
            
            # Analyze post-expiration period (5 days after)
            post_start = closest_idx + 1
            post_end = min(len(price_data), closest_idx + 6)
            
            if post_end <= post_start:
                return None
            
            pre_data = price_data.iloc[pre_start:pre_end]
            post_data = price_data.iloc[post_start:post_end]
            
            return {
                'expiration_date': exp_date.strftime('%Y-%m-%d'),
                'closest_trading_day': price_data.index[closest_idx].strftime('%Y-%m-%d'),
                'pre_expiration_return': pre_data['daily_return'].sum(),
                'post_expiration_return': post_data['daily_return'].sum(),
                'pre_expiration_volatility': pre_data['volatility_5d'].mean(),
                'post_expiration_volatility': post_data['volatility_5d'].mean(),
                'volume_spike_pre': pre_data['volume_ratio'].max(),
                'volume_spike_post': post_data['volume_ratio'].max(),
                'price_at_expiration': price_data['Close'].iloc[closest_idx]
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing expiration {exp_date}: {e}")
            return None
    
    def _calculate_activity_metrics(self) -> Dict[str, Any]:
        """
        Calculate options activity metrics for current expiration dates.
        """
        try:
            expirations = self.ticker.options
            if not expirations:
                return {'has_activity': False}
            
            activity_metrics = {
                'has_activity': True,
                'total_expirations': len(expirations),
                'near_term_activity': {},
                'put_call_ratios': {},
                'volume_patterns': {}
            }
            
            # Analyze first 3 expiration dates for activity
            for exp_date in expirations[:3]:
                try:
                    chain = self.ticker.option_chain(exp_date)
                    
                    if chain.calls.empty and chain.puts.empty:
                        continue
                    
                    # Calculate activity metrics
                    calls_volume = chain.calls['volume'].sum() if not chain.calls.empty else 0
                    puts_volume = chain.puts['volume'].sum() if not chain.puts.empty else 0
                    calls_oi = chain.calls['openInterest'].sum() if not chain.calls.empty else 0
                    puts_oi = chain.puts['openInterest'].sum() if not chain.puts.empty else 0
                    
                    activity_metrics['near_term_activity'][exp_date] = {
                        'calls_volume': calls_volume,
                        'puts_volume': puts_volume,
                        'total_volume': calls_volume + puts_volume,
                        'calls_open_interest': calls_oi,
                        'puts_open_interest': puts_oi,
                        'total_open_interest': calls_oi + puts_oi
                    }
                    
                    # Put/Call ratio
                    if calls_volume > 0:
                        activity_metrics['put_call_ratios'][exp_date] = puts_volume / calls_volume
                    else:
                        activity_metrics['put_call_ratios'][exp_date] = None
                        
                except Exception as e:
                    logger.warning(f"Error calculating activity for {exp_date}: {e}")
                    continue
            
            return activity_metrics
            
        except Exception as e:
            logger.error(f"Error calculating activity metrics: {e}")
            return {'has_activity': False, 'error': str(e)}
    
    def _engineer_proximity_features(self) -> Dict[str, Any]:
        """
        Engineer features related to options expiration proximity for ML models.
        """
        try:
            features = {}
            expirations = self.ticker.options
            
            if not expirations:
                return {'has_options': False}
            
            # Next expiration timing features
            next_exp = pd.to_datetime(expirations[0])
            days_to_next = (next_exp.date() - datetime.now().date()).days
            
            features.update({
                'has_options': True,
                'days_to_next_expiration': days_to_next,
                'next_expiration_date': expirations[0],
                'expiration_proximity_zone': self._classify_proximity_zone(days_to_next),
                'is_expiration_week': days_to_next <= 7,
                'is_pre_expiration_period': days_to_next <= 3,
                'weekly_vs_monthly': self._is_monthly_expiration(next_exp)
            })
            
            # Options activity features (if available)
            try:
                chain = self.ticker.option_chain(expirations[0])
                if not chain.calls.empty or not chain.puts.empty:
                    # Get current stock price
                    current_price = self.ticker.info.get('currentPrice', 0)
                    
                    if current_price > 0:
                        # Calculate moneyness features
                        atm_calls = chain.calls[
                            abs(chain.calls['strike'] - current_price) <= current_price * 0.05
                        ] if not chain.calls.empty else pd.DataFrame()
                        
                        atm_puts = chain.puts[
                            abs(chain.puts['strike'] - current_price) <= current_price * 0.05
                        ] if not chain.puts.empty else pd.DataFrame()
                        
                        features.update({
                            'atm_call_volume': atm_calls['volume'].sum() if not atm_calls.empty else 0,
                            'atm_put_volume': atm_puts['volume'].sum() if not atm_puts.empty else 0,
                            'atm_call_oi': atm_calls['openInterest'].sum() if not atm_calls.empty else 0,
                            'atm_put_oi': atm_puts['openInterest'].sum() if not atm_puts.empty else 0
                        })
                        
                        if features['atm_call_volume'] > 0:
                            features['atm_put_call_ratio'] = features['atm_put_volume'] / features['atm_call_volume']
                        else:
                            features['atm_put_call_ratio'] = None
                            
            except Exception as e:
                logger.debug(f"Could not calculate options activity features: {e}")
            
            return features
            
        except Exception as e:
            logger.error(f"Error engineering proximity features: {e}")
            return {'has_options': False, 'error': str(e)}
    
    def _analyze_small_cap_effects(self) -> Dict[str, Any]:
        """
        Specific analysis for small-cap options expiration effects.
        This targets the institutional blind spots you're looking to exploit.
        """
        try:
            # Determine if this is indeed a small cap
            market_cap_category = self._determine_market_cap_category()
            
            analysis = {
                'market_cap_category': market_cap_category,
                'is_institutional_blind_spot': market_cap_category in ['micro', 'small'],
                'options_availability': len(self.ticker.options) > 0 if self.ticker.options else False
            }
            
            if analysis['is_institutional_blind_spot'] and analysis['options_availability']:
                # This is exactly what we're looking for!
                analysis.update({
                    'inefficiency_potential': 'high',
                    'reasons': [
                        'Small market cap limits institutional position sizes',
                        'Options activity may create temporary distortions',
                        'Lower analyst coverage means delayed price discovery',
                        'Market makers may have looser risk controls for small positions'
                    ]
                })
                
                # Calculate specific small-cap metrics
                try:
                    info = self.ticker.info
                    analysis.update({
                        'float_percentage': info.get('floatShares', 0) / info.get('sharesOutstanding', 1) if info.get('sharesOutstanding', 0) > 0 else None,
                        'avg_volume': info.get('averageVolume', 0),
                        'beta': info.get('beta', None),
                        'short_ratio': info.get('shortRatio', None)
                    })
                except:
                    pass
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing small-cap effects: {e}")
            return {'error': str(e)}
    
    def _determine_market_cap_category(self) -> str:
        """
        Determine market cap category for small-cap analysis.
        """
        try:
            info = self.ticker.info
            market_cap = info.get('marketCap', 0)
            
            if market_cap == 0:
                return 'unknown'
            elif market_cap < 300_000_000:  # < $300M
                return 'micro'
            elif market_cap < 2_000_000_000:  # < $2B
                return 'small'
            elif market_cap < 10_000_000_000:  # < $10B
                return 'mid'
            else:
                return 'large'
                
        except:
            return 'unknown'
    
    def _classify_proximity_zone(self, days_to_expiration: int) -> str:
        """
        Classify proximity to options expiration for analysis.
        """
        if days_to_expiration < 0:
            return 'post_expiration'
        elif days_to_expiration <= 1:
            return 'expiration_day'
        elif days_to_expiration <= 7:
            return 'expiration_week'
        elif days_to_expiration <= 21:
            return 'near_term'
        else:
            return 'distant'
    
    def _is_monthly_expiration(self, exp_date) -> bool:
        """
        Check if expiration is a monthly (3rd Friday) expiration.
        """
        # Monthly options expire on 3rd Friday of the month
        # This is a simplified check - could be enhanced
        return exp_date.weekday() == 4 and 15 <= exp_date.day <= 21
    
    def _is_quarterly_expiration(self, exp_date) -> bool:
        """
        Check if expiration is a quarterly expiration.
        """
        return exp_date.month in [3, 6, 9, 12] and self._is_monthly_expiration(exp_date)
    
    def _classify_expiration_type(self, exp_date) -> str:
        """
        Classify type of expiration (weekly, monthly, quarterly, etc.).
        """
        if self._is_quarterly_expiration(exp_date):
            return 'quarterly'
        elif self._is_monthly_expiration(exp_date):
            return 'monthly'
        else:
            return 'weekly'
    
    def _get_historical_monthly_expirations(self, start_date, end_date) -> List[Dict[str, Any]]:
        """
        Generate historical monthly options expiration dates using algorithmic approach.
        Monthly options expire on the 3rd Friday of each month.
        """
        try:
            expirations = []
            
            # Ensure we're working with datetime objects, not dates
            if hasattr(start_date, 'date'):
                start_dt = start_date
            else:
                start_dt = pd.to_datetime(start_date)
                
            if hasattr(end_date, 'date'):
                end_dt = end_date
            else:
                end_dt = pd.to_datetime(end_date)
            
            # Start from first day of start month
            current_date = pd.to_datetime(f"{start_dt.year}-{start_dt.month}-01")
            
            while current_date <= end_dt:
                # Calculate 3rd Friday of the month
                third_friday = self._calculate_third_friday(current_date.year, current_date.month)
                
                if start_dt.date() <= third_friday <= end_dt.date():
                    expirations.append({
                        'expiration_date': third_friday.strftime('%Y-%m-%d'),
                        'year': third_friday.year,
                        'month': third_friday.month,
                        'type': 'monthly',
                        'is_quarterly': third_friday.month in [3, 6, 9, 12],
                        'day_of_week': third_friday.strftime('%A'),
                        'calculation_method': 'third_friday_algorithm'
                    })
                
                # Move to next month
                if current_date.month == 12:
                    current_date = pd.to_datetime(f"{current_date.year + 1}-01-01")
                else:
                    current_date = pd.to_datetime(f"{current_date.year}-{current_date.month + 1:02d}-01")
            
            logger.info(f"Generated {len(expirations)} historical expiration dates")
            return expirations
            
        except Exception as e:
            logger.error(f"Error generating historical expiration dates: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def _calculate_third_friday(self, year: int, month: int):
        """
        Calculate the 3rd Friday of a given month and year.
        This is the standard monthly options expiration date.
        """
        from datetime import date
        
        # Find the first day of the month
        first_day = date(year, month, 1)
        
        # Find the first Friday (weekday 4 = Friday)
        days_until_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + timedelta(days=days_until_friday)
        
        # Add 14 days to get the third Friday
        third_friday = first_friday + timedelta(days=14)
        
        # Handle edge case where third Friday might be in the next month
        if third_friday.month != month:
            # This can happen for February in some years, use second Friday instead
            third_friday = first_friday + timedelta(days=7)
        
        return third_friday
    
    def _analyze_pattern_strength(self, effects: List[Dict]) -> Dict[str, Any]:
        """
        Analyze the strength and consistency of expiration patterns.
        This is crucial for identifying reliable small-cap inefficiencies.
        """
        if not effects:
            return {}
        
        df = pd.DataFrame(effects)
        
        # Pattern consistency metrics
        pre_returns = df['pre_expiration_return']
        post_returns = df['post_expiration_return']
        
        pattern_analysis = {}
        
        # Directional consistency
        positive_pre = (pre_returns > 0).sum()
        negative_pre = (pre_returns < 0).sum()
        total_observations = len(pre_returns)
        
        pattern_analysis.update({
            'directional_consistency_pre': max(positive_pre, negative_pre) / total_observations if total_observations > 0 else 0,
            'dominant_pre_direction': 'positive' if positive_pre > negative_pre else 'negative',
            'pre_expiration_win_rate': positive_pre / total_observations if total_observations > 0 else 0
        })
        
        # Statistical significance (basic)
        from scipy import stats
        try:
            # Test if pre-expiration returns are significantly different from zero
            _, p_value_pre = stats.ttest_1samp(pre_returns.dropna(), 0)
            _, p_value_post = stats.ttest_1samp(post_returns.dropna(), 0)
            
            pattern_analysis.update({
                'pre_expiration_significance': p_value_pre,
                'post_expiration_significance': p_value_post,
                'pre_statistically_significant': p_value_pre < 0.05
            })
        except:
            # If scipy not available, skip statistical tests
            pattern_analysis.update({
                'pre_expiration_significance': None,
                'statistical_test_available': False
            })
        
        # Pattern strength score (0-100)
        consistency_score = pattern_analysis['directional_consistency_pre'] * 50  # 0-50
        magnitude_score = min(abs(pre_returns.mean()) * 1000, 25)  # 0-25 (scaled)
        sample_score = min(total_observations * 2, 25)  # 0-25 (more observations = higher score)
        
        pattern_analysis['pattern_strength_score'] = consistency_score + magnitude_score + sample_score
        pattern_analysis['strength_rating'] = self._rate_pattern_strength(pattern_analysis['pattern_strength_score'])
        
        return pattern_analysis
    
    def _rate_pattern_strength(self, score: float) -> str:
        """Rate pattern strength based on score."""
        if score >= 80:
            return 'very_strong'
        elif score >= 60:
            return 'strong'
        elif score >= 40:
            return 'moderate'
        elif score >= 20:
            return 'weak'
        else:
            return 'very_weak'
    
    def _calculate_aggregate_expiration_stats(self, effects: List[Dict]) -> Dict[str, Any]:
        """
        Calculate aggregate statistics across multiple expiration effects.
        """
        if not effects:
            return {}
        
        df = pd.DataFrame(effects)
        
        return {
            'average_pre_expiration_return': df['pre_expiration_return'].mean(),
            'average_post_expiration_return': df['post_expiration_return'].mean(),
            'pre_vs_post_return_difference': df['pre_expiration_return'].mean() - df['post_expiration_return'].mean(),
            'volatility_increase_pre_expiration': (df['pre_expiration_volatility'].mean() / df['post_expiration_volatility'].mean()) - 1 if df['post_expiration_volatility'].mean() > 0 else 0,
            'volume_spike_frequency': (df['volume_spike_pre'] > 1.5).mean(),
            'consistent_pattern_strength': abs(df['pre_expiration_return'].mean()) / df['pre_expiration_return'].std() if df['pre_expiration_return'].std() > 0 else 0,
            'median_pre_return': df['pre_expiration_return'].median(),
            'median_post_return': df['post_expiration_return'].median(),
            'max_pre_spike': df['volume_spike_pre'].max(),
            'quarterly_vs_monthly_effect': self._compare_quarterly_monthly(df) if 'is_quarterly' in df.columns else {}
        }
    
    def _compare_quarterly_monthly(self, df) -> Dict[str, Any]:
        """Compare quarterly vs monthly expiration effects."""
        if 'is_quarterly' not in df.columns:
            return {}
        
        quarterly = df[df['is_quarterly'] == True]
        monthly = df[df['is_quarterly'] == False]
        
        if quarterly.empty or monthly.empty:
            return {'comparison_available': False}
        
        return {
            'comparison_available': True,
            'quarterly_avg_pre_return': quarterly['pre_expiration_return'].mean(),
            'monthly_avg_pre_return': monthly['pre_expiration_return'].mean(),
            'quarterly_stronger': quarterly['pre_expiration_return'].mean() > monthly['pre_expiration_return'].mean(),
            'quarterly_sample_size': len(quarterly),
            'monthly_sample_size': len(monthly)
        }
    
    def _save_options_data(self, data: Dict[str, Any]) -> Dict[str, str]:
        """
        Save options expiration data to centralized storage.
        
        Structure: storage/multi_data_sources/calendar_events/options_expiration/{symbol}/
        - raw/ : Complete options data (JSON)
        - processed/ : Analysis results (CSV)
        - analysis/ : Proximity effects analysis (JSON)
        """
        try:
            # Create directory structure
            base_dir = self.storage_base
            symbol_dir = base_dir / self.symbol
            
            raw_dir = symbol_dir / "raw"
            processed_dir = symbol_dir / "processed"
            analysis_dir = symbol_dir / "analysis"
            
            for directory in [raw_dir, processed_dir, analysis_dir]:
                directory.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            date_suffix = f"{self.from_date}_{self.to_date}" if self.from_date and self.to_date else datetime.now().strftime("%Y-%m-%d")
            saved_paths = {}
            
            # Save raw data
            raw_filename = f"{date_suffix}_options_expiration_data.json"
            raw_path = raw_dir / raw_filename
            
            # Prepare JSON-serializable data
            json_data = {
                'metadata': data['metadata'],
                'current_options_data': data['current_options_data'],
                'expiration_calendar': data['expiration_calendar'],
                'historical_proximity_analysis': data['historical_proximity_analysis'],
                'options_activity_metrics': data['options_activity_metrics'],
                'proximity_features': data['proximity_features'],
                'small_cap_analysis': data['small_cap_analysis']
            }
            
            with open(raw_path, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            saved_paths['raw_data'] = str(raw_path)
            
            # Save proximity analysis
            if data['historical_proximity_analysis']:
                analysis_filename = f"{date_suffix}_proximity_analysis.json"
                analysis_path = analysis_dir / analysis_filename
                
                analysis_data = {
                    'symbol': self.symbol,
                    'analysis_timestamp': datetime.now().isoformat(),
                    'date_range': {'from': self.from_date, 'to': self.to_date},
                    'proximity_effects': data['historical_proximity_analysis'],
                    'small_cap_analysis': data['small_cap_analysis'],
                    'summary': {
                        'has_options': data['current_options_data'].get('has_options', False),
                        'total_expirations': len(data['expiration_calendar']),
                        'market_cap_category': data['small_cap_analysis'].get('market_cap_category', 'unknown')
                    }
                }
                
                with open(analysis_path, 'w') as f:
                    json.dump(analysis_data, f, indent=2, default=str)
                saved_paths['analysis'] = str(analysis_path)
            
            logger.info(f"Options expiration data saved for {self.symbol}: {len(saved_paths)} files")
            return saved_paths
            
        except Exception as e:
            logger.error(f"Error saving options expiration data for {self.symbol}: {e}")
            return {}
    
    def load_saved_options_data(self, date_range: str = None) -> Dict[str, Any]:
        """
        Load previously saved options expiration data from storage.
        
        Args:
            date_range: Specific date range to load (e.g., "2022-01-01_2025-01-01")
        
        Returns:
            Dictionary with loaded options expiration data
        """
        try:
            base_dir = self.storage_base / self.symbol / "raw"
            
            if not base_dir.exists():
                logger.warning(f"No saved options data found for {self.symbol}")
                return {}
            
            # Find appropriate file
            if date_range:
                filename = f"{date_range}_options_expiration_data.json"
                file_path = base_dir / filename
            else:
                # Get most recent file
                json_files = list(base_dir.glob("*_options_expiration_data.json"))
                if not json_files:
                    logger.warning(f"No options data files found for {self.symbol}")
                    return {}
                file_path = max(json_files, key=lambda x: x.stat().st_mtime)
            
            if file_path.exists():
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                logger.info(f"Successfully loaded saved options data for {self.symbol}")
                return data
            else:
                logger.warning(f"Options data file not found: {file_path}")
                return {}
                
        except Exception as e:
            logger.error(f"Error loading saved options data for {self.symbol}: {e}")
            return {}