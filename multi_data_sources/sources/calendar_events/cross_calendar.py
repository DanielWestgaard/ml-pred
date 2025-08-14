from typing import Optional
import logging
import pandas as pd
from datetime import datetime, timedelta
from multi_data_sources.sources.ds_base import DataSourceBase
from multi_data_sources.sources.calendar_events.complexities.earnings import Earnings
from multi_data_sources.sources.calendar_events.complexities.options_expiration import OptionsExpiration

logger = logging.getLogger(__name__)


class CalendarEventsDataSource(DataSourceBase):
    """..."""
    
    # Overview of what calendar related has "implemented"
    # Temporary: These are only suggestions as I try to find the best usecase for this
    CROSS_CALENDAR_COMPLEXITY = [
        "earnings",  # TODO: This in and of itself can be complex - tuesday/friday announcements, days until next earnings call, etc.
        "options_expiration",  # TODO: Proximity (to see effects on small caps)
        "holidays",  # TODO: Classification of holiday or no?
        "fiscal_year_ends"  # TODO: Fiscal quarter-end effects combined with company size
        # Other suggestions: ..? Perhaps only starting off simple
    ]
    
    def __init__(self, symbol: str, stock: bool = False, from_date: Optional[str] = None, 
                 to_date: Optional[str] = None, purpose: str = "historical"):
        """
        ...
        """
        # Read configuration
        
        # Set variables
        self.symbol = symbol
        self.stock = stock
        self.from_date = from_date
        self.to_date = to_date
        self.purpose = purpose
        
        # Initialize complexity components
        self.earnings = Earnings(self.symbol, self.stock, self.from_date, self.to_date, self.purpose)
        self.options_expiration = OptionsExpiration(self.symbol, self.stock, self.from_date, self.to_date, self.purpose)
    
    def get_historical_data(self):
        """
        Fetch historical data from all calendar event complexities.
        
        Returns:
            Dictionary containing data from all complexity components:
            - earnings_data: Historical earnings analysis
            - options_expiration_data: Options expiration proximity analysis
            - cross_calendar_analysis: Combined calendar effects analysis  TODO: perhaps this will be moved out later to "separate concerns"?
        """
        result = {
            'earnings_data': self.earnings.get_historical_data(),
            'options_expiration_data': self.options_expiration.get_historical_data(),
            'cross_calendar_analysis': self._analyze_cross_calendar_effects(),
            'holidays_vacations': self.holiday_vacation(),
            'fiscal_quarters': self.fiscal_quarter(),
            'metadata': {
                'symbol': self.symbol,
                'complexities_analyzed': ['earnings', 'options_expiration', 'holidays_vacations', 'fiscal_quarters'],
                'from_date': self.from_date,
                'to_date': self.to_date,
                'purpose': self.purpose,
                'phase_1_complete': True,
                'institutional_blind_spots_targeted': [
                    'small_cap_options_expiration',
                    'holiday_vacation_periods', 
                    'fiscal_quarter_window_dressing',
                    'earnings_timing_interactions'
                ]
            }
        }
        
        # Save cross-calendar data to centralized storage
        saved_paths = self._save_cross_calendar_data(result)
        result['saved_paths'] = saved_paths
        
        return result
        
    def subscribe_live_data(self):
        """
        Subscribe to live data from all calendar event complexities.
        
        Returns:
            Dictionary with live subscription information from all components
        """
        return {
            'earnings_subscription': self.earnings.subscribe_live_data(),
            'options_expiration_subscription': self.options_expiration.subscribe_live_data(),
            'cross_calendar_monitoring': {
                'combined_proximity_effects': True,
                'earnings_options_interaction_analysis': True,
                'institutional_blind_spot_detection': True
            }
        }
    
    def _analyze_cross_calendar_effects(self):
        """
        Analyze interactions between different calendar events.
        This is where we find the real inefficiencies - when multiple events align.
        """
        try:
            # Get proximity data from both components (using historical data only)
            earnings_features = self.earnings._engineer_earnings_features(self.to_date)
            options_features = self.options_expiration._engineer_proximity_features(self.to_date)
            
            cross_effects = {
                'has_both_events': (
                    earnings_features.get('days_since_last_earnings') is not None and 
                    options_features.get('days_since_last_expiration') is not None
                ),
                'event_proximity_analysis': {},
                'institutional_blind_spot_score': 0
            }
            
            if cross_effects['has_both_events']:
                days_since_earnings = earnings_features.get('days_since_last_earnings', 999)
                days_since_options = options_features.get('days_since_last_expiration', 999)
                
                # Analyze event proximity (using historical data only)
                cross_effects['event_proximity_analysis'] = {
                    'days_since_earnings': days_since_earnings,
                    'days_since_options': days_since_options,
                    'both_events_recent': (days_since_earnings <= 30 and days_since_options <= 30),
                    'earnings_more_recent': days_since_earnings < days_since_options,
                    'options_more_recent': days_since_options < days_since_earnings,
                    'conflicting_market_forces': self._detect_conflicting_forces(earnings_features, options_features)
                }
                
                # Calculate institutional blind spot score
                market_cap_category = self.options_expiration._determine_market_cap_category()
                blind_spot_factors = []
                
                if market_cap_category in ['micro', 'small']:
                    blind_spot_factors.append('small_market_cap')
                
                if cross_effects['event_proximity_analysis']['events_same_week']:
                    blind_spot_factors.append('multiple_events_same_week')
                
                if options_features.get('has_options') and earnings_features.get('has_options', True):
                    blind_spot_factors.append('options_earnings_interaction')
                
                cross_effects['institutional_blind_spot_score'] = len(blind_spot_factors)
                cross_effects['blind_spot_factors'] = blind_spot_factors
            
            return cross_effects
            
        except Exception as e:
            logger.error(f"Error analyzing cross-calendar effects: {e}")
            return {'error': str(e)}
    
    def _detect_conflicting_forces(self, earnings_features, options_features):
        """
        Detect when earnings and options expiration create conflicting market forces.
        This is a key source of inefficiencies institutions miss.
        """
        try:
            conflicting_signals = []
            
            # Check for earnings announcement near options expiration
            if (earnings_features.get('earnings_proximity_zone') in ['immediate_pre_earnings', 'near_term_pre_earnings'] and
                options_features.get('expiration_proximity_zone') in ['expiration_week', 'near_term']):
                conflicting_signals.append('earnings_announcement_during_options_week')
            
            # Check for high options activity after recent earnings
            if (options_features.get('atm_call_volume', 0) + options_features.get('atm_put_volume', 0) > 1000 and
                earnings_features.get('days_since_last_earnings', 999) <= 7):
                conflicting_signals.append('high_options_activity_post_earnings')
            
            # Check for put/call ratio extremes after recent earnings
            put_call_ratio = options_features.get('atm_put_call_ratio')
            if (put_call_ratio is not None and 
                (put_call_ratio > 2.0 or put_call_ratio < 0.5) and
                earnings_features.get('days_since_last_earnings', 999) <= 14):
                conflicting_signals.append('extreme_put_call_ratio_post_earnings')
            
            return {
                'has_conflicts': len(conflicting_signals) > 0,
                'conflict_signals': conflicting_signals,
                'conflict_severity': len(conflicting_signals)
            }
            
        except Exception as e:
            logger.warning(f"Error detecting conflicting forces: {e}")
            return {'has_conflicts': False, 'error': str(e)}
        
    def holiday_vacation(self):
        """
        Holiday/vacation period patterns in specific sectors.
        Focuses on American stocks and major market holidays that create trading anomalies.
        
        Key insight: Small-cap stocks often show unique patterns during holiday periods
        when institutional trading desks are understaffed or closed.
        """
        try:
            import exchange_calendars as xcals
            
            # Get NYSE calendar for US market holidays
            nyse = xcals.get_calendar("NYSE")
            
            # Define analysis date range
            if self.from_date and self.to_date:
                start_date = pd.to_datetime(self.from_date)
                end_date = pd.to_datetime(self.to_date)
            else:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365)
            
            # Get market holidays in the date range
            # Try different methods to get holidays from exchange_calendars
            try:
                # Method 1: Try sessions and find gaps
                sessions = nyse.sessions_in_range(start_date, end_date)
                all_weekdays = pd.bdate_range(start=start_date, end=end_date)
                holidays_mask = ~all_weekdays.isin(sessions)
                relevant_holidays = all_weekdays[holidays_mask]
                
            except Exception as e:
                logger.debug(f"Sessions method failed: {e}")
                try:
                    # Method 2: Use regular_holidays and adhoc_holidays
                    regular_holidays = nyse.regular_holidays
                    adhoc_holidays = nyse.adhoc_holidays
                    
                    # Combine and filter by date range
                    all_holidays = []
                    for year in range(start_date.year, end_date.year + 1):
                        for holiday in regular_holidays:
                            try:
                                holiday_date = holiday.dates(start_date=f'{year}-01-01', end_date=f'{year}-12-31', return_name=False)
                                if len(holiday_date) > 0:
                                    all_holidays.extend(holiday_date)
                            except:
                                continue
                    
                    # Add adhoc holidays
                    for holiday_date in adhoc_holidays.keys():
                        if start_date <= holiday_date <= end_date:
                            all_holidays.append(holiday_date)
                    
                    relevant_holidays = pd.DatetimeIndex(all_holidays).unique()
                    
                except Exception as e2:
                    logger.warning(f"Could not get NYSE holidays: {e2}")
                    # Fallback to basic holidays
                    relevant_holidays = self._get_basic_holidays(start_date, end_date)
            
            # Define vacation periods (institutional blind spots)
            vacation_periods = self._get_vacation_periods(start_date, end_date)
            
            holiday_analysis = {
                'market_holidays': [h.strftime('%Y-%m-%d') for h in relevant_holidays],
                'vacation_periods': vacation_periods,
                'pre_holiday_effects': self._analyze_pre_holiday_patterns(relevant_holidays),
                'vacation_period_effects': self._analyze_vacation_patterns(vacation_periods),
                'institutional_blind_spots': {
                    'december_effect': self._analyze_december_effect(start_date, end_date),
                    'summer_doldrums': self._analyze_summer_patterns(start_date, end_date),
                    'options_expiry_holiday_conflict': self._check_expiry_holiday_conflicts(relevant_holidays)
                },
                'total_holidays_analyzed': len(relevant_holidays),
                'total_vacation_periods': len(vacation_periods)
            }
            
            logger.info(f"Analyzed {len(relevant_holidays)} holidays and {len(vacation_periods)} vacation periods for {self.symbol}")
            return holiday_analysis
            
        except ImportError:
            logger.warning("exchange_calendars not available, using basic holiday analysis")
            return self._basic_holiday_analysis()
        except Exception as e:
            logger.error(f"Error analyzing holiday patterns: {e}")
            return {'error': str(e)}
    
    def _get_vacation_periods(self, start_date, end_date):
        """Define major vacation periods when institutional activity decreases."""
        vacation_periods = []
        
        for year in range(start_date.year, end_date.year + 1):
            # Summer vacation: July 4th week + week after
            vacation_periods.append({
                'name': f'Summer Vacation {year}',
                'start': f'{year}-06-28',
                'end': f'{year}-07-11',
                'type': 'summer',
                'institutional_impact': 'high'
            })
            
            # Year-end vacation: Week between Christmas and New Year
            vacation_periods.append({
                'name': f'Year-end Vacation {year}',
                'start': f'{year}-12-24',
                'end': f'{year+1}-01-02',
                'type': 'year_end',
                'institutional_impact': 'very_high'
            })
            
            # Thanksgiving week
            vacation_periods.append({
                'name': f'Thanksgiving Week {year}',
                'start': f'{year}-11-22',
                'end': f'{year}-11-29',
                'type': 'thanksgiving',
                'institutional_impact': 'medium'
            })
        
        # Filter to date range
        filtered_periods = []
        for period in vacation_periods:
            period_start = pd.to_datetime(period['start'])
            if start_date <= period_start <= end_date:
                filtered_periods.append(period)
        
        return filtered_periods
    
    def _analyze_pre_holiday_patterns(self, holidays):
        """Analyze price behavior before major holidays."""
        if len(holidays) == 0:
            return {}
        
        return {
            'pattern_description': 'Pre-holiday effects often show reduced volume and altered price patterns',
            'analysis_method': 'Would analyze 1-3 days before each holiday for volume/return anomalies',
            'expected_small_cap_effect': 'Small caps may show larger moves due to reduced institutional oversight',
            'holidays_count': len(holidays)
        }
    
    def _analyze_vacation_patterns(self, vacation_periods):
        """Analyze market behavior during vacation periods."""
        if len(vacation_periods) == 0:
            return {}
        
        return {
            'pattern_description': 'Vacation periods create institutional blind spots',
            'key_insights': [
                'Reduced institutional trading volume',
                'Increased retail trader influence',
                'Higher probability of gap moves in small caps',
                'Delayed reaction to news and earnings'
            ],
            'periods_analyzed': len(vacation_periods),
            'highest_impact_periods': [p['name'] for p in vacation_periods if p['institutional_impact'] in ['high', 'very_high']]
        }
    
    def _analyze_december_effect(self, start_date, end_date):
        """Analyze December small-cap outperformance patterns."""
        return {
            'effect_name': 'December Small-Cap Effect',
            'description': 'Small caps often outperform in December due to tax loss selling reversal and window dressing',
            'institutional_blind_spot': 'Large institutions avoid small caps during year-end due to liquidity concerns',
            'analysis_period': f"{start_date.strftime('%Y')} to {end_date.strftime('%Y')}"
        }
    
    def _analyze_summer_patterns(self, start_date, end_date):
        """Analyze summer doldrums effects on small caps."""
        return {
            'effect_name': 'Summer Doldrums',
            'description': 'Reduced institutional activity in June-August creates opportunities',
            'key_factors': [
                'Vacation schedules reduce institutional oversight',
                'Lower volume can amplify small-cap price moves',
                'Earnings surprises may have delayed reactions'
            ]
        }
    
    def _check_expiry_holiday_conflicts(self, holidays):
        """Check when options expiry dates conflict with holidays."""
        conflicts = []
        
        # Get options expiration dates
        if hasattr(self, 'options_expiration'):
            exp_calendar = self.options_expiration._get_expiration_calendar()
            
            for exp in exp_calendar[:12]:  # Check near-term expirations
                exp_date = pd.to_datetime(exp['expiration_date'])
                
                # Check if within 2 days of a holiday
                for holiday in holidays:
                    if abs((exp_date - holiday).days) <= 2:
                        conflicts.append({
                            'expiration_date': exp['expiration_date'],
                            'holiday_date': holiday.strftime('%Y-%m-%d'),
                            'days_apart': abs((exp_date - holiday).days),
                            'potential_impact': 'high' if abs((exp_date - holiday).days) <= 1 else 'medium'
                        })
        
        return {
            'conflicts_found': len(conflicts),
            'conflicts': conflicts,
            'analysis': 'Holiday-expiry conflicts create unique arbitrage opportunities'
        }
    
    def _get_basic_holidays(self, start_date, end_date):
        """Generate basic US market holidays for date range."""
        holidays = []
        
        for year in range(start_date.year, end_date.year + 1):
            # Major US market holidays
            year_holidays = [
                f'{year}-01-01',  # New Year's Day
                f'{year}-07-04',  # Independence Day  
                f'{year}-12-25',  # Christmas Day
                f'{year}-11-28',  # Thanksgiving (approximate)
            ]
            
            for holiday_str in year_holidays:
                holiday_date = pd.to_datetime(holiday_str)
                if start_date <= holiday_date <= end_date:
                    holidays.append(holiday_date)
        
        return pd.DatetimeIndex(holidays)
    
    def _basic_holiday_analysis(self):
        """Fallback holiday analysis without exchange_calendars."""
        return {
            'basic_holidays': [
                '2024-01-01', '2024-07-04', '2024-12-25',  # Major US holidays
                '2025-01-01', '2025-07-04', '2025-12-25'
            ],
            'analysis': 'Basic holiday analysis - install exchange_calendars for full functionality',
            'limitation': 'Limited to hardcoded major holidays'
        }
        
    
    def fiscal_quarter(self):
        """
        Fiscal quarter-end effects combined with company size.
        
        Key insight: Small-cap stocks show different patterns around fiscal quarter-ends
        due to institutional window dressing, rebalancing, and reporting requirements.
        """
        try:
            
            # Define analysis date range
            if self.from_date and self.to_date:
                start_date = pd.to_datetime(self.from_date)
                end_date = pd.to_datetime(self.to_date)
            else:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365)
            
            # Get company's fiscal year information from yfinance if available
            fiscal_info = self._get_company_fiscal_info()
            
            # Generate quarterly dates
            quarterly_dates = self._generate_fiscal_quarters(start_date, end_date, fiscal_info)
            
            # Analyze quarterly effects
            quarterly_analysis = {
                'company_fiscal_info': fiscal_info,
                'quarterly_dates': quarterly_dates,
                'institutional_effects': self._analyze_institutional_quarter_effects(),
                'small_cap_specific_effects': self._analyze_small_cap_quarter_effects(),
                'window_dressing_periods': self._identify_window_dressing_periods(quarterly_dates),
                'rebalancing_effects': self._analyze_rebalancing_effects(quarterly_dates),
                'quarter_end_conflicts': self._check_quarter_earnings_conflicts(quarterly_dates),
                'total_quarters_analyzed': len(quarterly_dates)
            }
            
            logger.info(f"Analyzed {len(quarterly_dates)} fiscal quarters for {self.symbol}")
            return quarterly_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing fiscal quarter effects: {e}")
            return {'error': str(e)}
    
    def _get_company_fiscal_info(self):
        """Get company's fiscal year information."""
        try:
            if hasattr(self, 'earnings') and hasattr(self.earnings, 'ticker'):
                info = self.earnings.ticker.info
                fiscal_year_end = info.get('fiscalYearEnds')
                
                if fiscal_year_end:
                    # Convert timestamp to readable format
                    fiscal_end_date = pd.to_datetime(fiscal_year_end, unit='s')
                    fiscal_end_month = fiscal_end_date.month
                else:
                    fiscal_end_month = 12  # Default to calendar year
                
                return {
                    'fiscal_year_end_month': fiscal_end_month,
                    'fiscal_calendar_type': 'calendar_year' if fiscal_end_month == 12 else 'fiscal_year',
                    'last_fiscal_year_end': fiscal_end_date.strftime('%Y-%m-%d') if fiscal_year_end else None
                }
        except:
            pass
        
        # Default to calendar year quarters
        return {
            'fiscal_year_end_month': 12,
            'fiscal_calendar_type': 'calendar_year_assumed',
            'note': 'Using calendar year quarters as default'
        }
    
    def _generate_fiscal_quarters(self, start_date, end_date, fiscal_info):
        """Generate fiscal quarter-end dates."""
        quarterly_dates = []
        fiscal_end_month = fiscal_info['fiscal_year_end_month']
        
        # Calculate quarter-end months based on fiscal year end
        if fiscal_end_month == 12:  # Calendar year
            quarter_months = [3, 6, 9, 12]
        else:  # Fiscal year
            base_month = fiscal_end_month
            quarter_months = [
                ((base_month + 3 - 1) % 12) + 1,   # Q1 end
                ((base_month + 6 - 1) % 12) + 1,   # Q2 end  
                ((base_month + 9 - 1) % 12) + 1,   # Q3 end
                base_month                          # Q4 end (fiscal year end)
            ]
        
        # Generate quarter-end dates for the analysis period
        current_year = start_date.year
        while current_year <= end_date.year + 1:
            for i, month in enumerate(quarter_months):
                # Determine the year for this quarter
                if fiscal_end_month != 12 and month < fiscal_end_month:
                    quarter_year = current_year + 1
                else:
                    quarter_year = current_year
                
                # Last day of the quarter month
                if month == 12:
                    quarter_end = pd.to_datetime(f'{quarter_year}-12-31')
                else:
                    next_month = pd.to_datetime(f'{quarter_year}-{month + 1}-01')
                    quarter_end = next_month - timedelta(days=1)
                
                if start_date <= quarter_end <= end_date:
                    quarterly_dates.append({
                        'quarter_end_date': quarter_end.strftime('%Y-%m-%d'),
                        'quarter': f'Q{i+1}',
                        'fiscal_year': quarter_year if fiscal_end_month == 12 else (quarter_year if month >= fiscal_end_month else quarter_year - 1),
                        'is_fiscal_year_end': (i == 3),  # Q4 is always fiscal year end
                        'month': month
                    })
            
            current_year += 1
        
        return sorted(quarterly_dates, key=lambda x: x['quarter_end_date'])
    
    def _analyze_institutional_quarter_effects(self):
        """Analyze how institutions behave around quarter-ends."""
        return {
            'window_dressing': {
                'description': 'Institutions buy/sell stocks to improve portfolio appearance',
                'timing': 'Last 2-3 trading days of quarter',
                'small_cap_impact': 'Higher volatility due to lower liquidity'
            },
            'portfolio_rebalancing': {
                'description': 'Systematic rebalancing creates predictable flows',
                'timing': 'Quarter-end and first few days of new quarter',
                'opportunity': 'Small caps may be overlooked in large rebalancing'
            },
            'reporting_requirements': {
                'description': '13F filings and portfolio disclosures affect behavior',
                'impact': 'Some institutions avoid small positions to stay off radar',
                'blind_spot': 'Creates opportunities in sub-$100M market cap stocks'
            }
        }
    
    def _analyze_small_cap_quarter_effects(self):
        """Analyze quarter-end effects specific to small-cap stocks."""
        return {
            'liquidity_constraints': {
                'issue': 'Large institutions cannot easily trade small caps at quarter-end',
                'opportunity': 'Creates temporary price dislocations',
                'timing': '3-5 days around quarter-end'
            },
            'index_rebalancing': {
                'russell_reconstitution': 'Major annual event in June affecting small caps',
                'quarterly_adjustments': 'Smaller but regular rebalancing events',
                'flow_predictability': 'Known dates create arbitrage opportunities'
            },
            'earnings_calendar_conflict': {
                'issue': 'Some companies report earnings near quarter-end',
                'complexity': 'Creates multiple overlapping effects',
                'institution_response': 'Often delayed due to competing priorities'
            }
        }
    
    def _identify_window_dressing_periods(self, quarterly_dates):
        """Identify specific window dressing periods around quarter-ends."""
        window_dressing_periods = []
        
        for quarter in quarterly_dates:
            quarter_end = pd.to_datetime(quarter['quarter_end_date'])
            
            # Window dressing typically occurs 3-5 trading days before quarter-end
            window_start = quarter_end - timedelta(days=7)  # Buffer for weekends
            
            window_dressing_periods.append({
                'quarter': quarter['quarter'],
                'fiscal_year': quarter['fiscal_year'],
                'window_start': window_start.strftime('%Y-%m-%d'),
                'quarter_end': quarter['quarter_end_date'],
                'intensity': 'very_high' if quarter['is_fiscal_year_end'] else 'high',
                'expected_effects': [
                    'Increased volume in large caps',
                    'Potential selling pressure in small caps',
                    'Portfolio "beautification" trades',
                    'Increased volatility in illiquid names'
                ]
            })
        
        return window_dressing_periods
    
    def _analyze_rebalancing_effects(self, quarterly_dates):
        """Analyze systematic rebalancing effects around quarter-ends."""
        return {
            'systematic_rebalancing': {
                'description': 'ETFs and index funds rebalance on quarter-ends',
                'small_cap_impact': 'May create temporary supply/demand imbalances',
                'timing': 'Typically after market close on quarter-end date'
            },
            'pension_fund_rebalancing': {
                'description': 'Large pension funds rebalance quarterly',
                'blind_spot': 'Often ignore small caps due to liquidity constraints',
                'opportunity_window': '1-3 days after quarter-end'
            },
            'mutual_fund_flows': {
                'description': 'Month-end and quarter-end flows from retail investors',
                'pattern': 'Often favors large caps, creating small cap opportunities'
            }
        }
    
    def _check_quarter_earnings_conflicts(self, quarterly_dates):
        """Check when quarter-ends conflict with earnings announcements."""
        conflicts = []
        
        # Get earnings dates if available
        if hasattr(self, 'earnings'):
            try:
                earnings_data = self.earnings._get_upcoming_earnings()
                earnings_dates = [e['date'] for e in earnings_data if 'date' in e]
                
                for quarter in quarterly_dates:
                    quarter_end = pd.to_datetime(quarter['quarter_end_date'])
                    
                    # Check for earnings within 5 days of quarter-end
                    for earnings_date_str in earnings_dates:
                        earnings_date = pd.to_datetime(earnings_date_str)
                        days_diff = abs((earnings_date - quarter_end).days)
                        
                        if days_diff <= 5:
                            conflicts.append({
                                'quarter_end': quarter['quarter_end_date'],
                                'earnings_date': earnings_date_str,
                                'days_apart': days_diff,
                                'quarter': quarter['quarter'],
                                'complexity_level': 'high' if days_diff <= 2 else 'medium'
                            })
            except:
                pass
        
        return {
            'conflicts_found': len(conflicts),
            'conflicts': conflicts,
            'analysis': 'Quarter-end + earnings creates complex institutional decision-making'
        }
    
    def _save_cross_calendar_data(self, data):
        """Save cross-calendar analysis data to centralized storage."""
        try:
            from pathlib import Path
            import json
            from config.config import get_calendar_events_raw_dir, get_calendar_events_analysis_dir
            
            # Create directory structure using centralized config
            raw_dir = Path(get_calendar_events_raw_dir(self.symbol, 'cross_calendar'))
            analysis_dir = Path(get_calendar_events_analysis_dir(self.symbol, 'cross_calendar'))
            
            for directory in [raw_dir, analysis_dir]:
                directory.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            date_suffix = f"{self.from_date}_{self.to_date}" if self.from_date and self.to_date else datetime.now().strftime("%Y-%m-%d")
            saved_paths = {}
            
            # Save complete cross-calendar data
            raw_filename = f"{date_suffix}_cross_calendar_complete.json"
            raw_path = raw_dir / raw_filename
            
            # Prepare JSON-serializable data
            json_data = {
                'holidays_vacations': data['holidays_vacations'],
                'fiscal_quarters': data['fiscal_quarters'],
                'cross_calendar_analysis': data['cross_calendar_analysis'],
                'metadata': data['metadata']
            }
            
            with open(raw_path, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            saved_paths['cross_calendar_complete'] = str(raw_path)
            
            # Save holiday-specific analysis
            if data['holidays_vacations']:
                holiday_filename = f"{date_suffix}_holiday_analysis.json"
                holiday_path = analysis_dir / holiday_filename
                
                holiday_data = {
                    'symbol': self.symbol,
                    'analysis_timestamp': datetime.now().isoformat(),
                    'date_range': {'from': self.from_date, 'to': self.to_date},
                    'holiday_analysis': data['holidays_vacations'],
                    'summary': {
                        'total_holidays': data['holidays_vacations'].get('total_holidays_analyzed', 0),
                        'vacation_periods': data['holidays_vacations'].get('total_vacation_periods', 0),
                        'institutional_impact': 'high' if data['holidays_vacations'].get('total_vacation_periods', 0) > 0 else 'low'
                    }
                }
                
                with open(holiday_path, 'w') as f:
                    json.dump(holiday_data, f, indent=2, default=str)
                saved_paths['holiday_analysis'] = str(holiday_path)
            
            # Save fiscal quarter analysis
            if data['fiscal_quarters']:
                quarter_filename = f"{date_suffix}_fiscal_quarter_analysis.json"
                quarter_path = analysis_dir / quarter_filename
                
                quarter_data = {
                    'symbol': self.symbol,
                    'analysis_timestamp': datetime.now().isoformat(),
                    'date_range': {'from': self.from_date, 'to': self.to_date},
                    'fiscal_quarter_analysis': data['fiscal_quarters'],
                    'summary': {
                        'total_quarters': data['fiscal_quarters'].get('total_quarters_analyzed', 0),
                        'fiscal_calendar_type': data['fiscal_quarters'].get('company_fiscal_info', {}).get('fiscal_calendar_type', 'unknown'),
                        'window_dressing_potential': 'high'
                    }
                }
                
                with open(quarter_path, 'w') as f:
                    json.dump(quarter_data, f, indent=2, default=str)
                saved_paths['fiscal_quarter_analysis'] = str(quarter_path)
            
            logger.info(f"Cross-calendar data saved for {self.symbol}: {len(saved_paths)} files")
            return saved_paths
            
        except Exception as e:
            logger.error(f"Error saving cross-calendar data for {self.symbol}: {e}")
            return {}