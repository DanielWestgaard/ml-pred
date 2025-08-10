from typing import Optional
import logging
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
            - cross_calendar_analysis: Combined calendar effects analysis
        """
        result = {
            'earnings_data': self.earnings.get_historical_data(),
            'options_expiration_data': self.options_expiration.get_historical_data(),
            'cross_calendar_analysis': self._analyze_cross_calendar_effects(),
            'metadata': {
                'symbol': self.symbol,
                'complexities_analyzed': ['earnings', 'options_expiration'],
                'from_date': self.from_date,
                'to_date': self.to_date,
                'purpose': self.purpose
            }
        }
        
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
            # Get proximity data from both components
            earnings_features = self.earnings._engineer_earnings_features()
            options_features = self.options_expiration._engineer_proximity_features()
            
            cross_effects = {
                'has_both_events': (
                    earnings_features.get('days_to_next_earnings') is not None and 
                    options_features.get('days_to_next_expiration') is not None
                ),
                'event_proximity_analysis': {},
                'institutional_blind_spot_score': 0
            }
            
            if cross_effects['has_both_events']:
                days_to_earnings = earnings_features.get('days_to_next_earnings', 999)
                days_to_options = options_features.get('days_to_next_expiration', 999)
                
                # Analyze event proximity
                cross_effects['event_proximity_analysis'] = {
                    'days_between_events': abs(days_to_earnings - days_to_options),
                    'events_same_week': abs(days_to_earnings - days_to_options) <= 7,
                    'earnings_before_options': days_to_earnings < days_to_options,
                    'options_before_earnings': days_to_options < days_to_earnings,
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
            
            # Check for high options activity before earnings
            if (options_features.get('atm_call_volume', 0) + options_features.get('atm_put_volume', 0) > 1000 and
                earnings_features.get('days_to_next_earnings', 999) <= 7):
                conflicting_signals.append('high_options_activity_pre_earnings')
            
            # Check for put/call ratio extremes near earnings
            put_call_ratio = options_features.get('atm_put_call_ratio')
            if (put_call_ratio is not None and 
                (put_call_ratio > 2.0 or put_call_ratio < 0.5) and
                earnings_features.get('days_to_next_earnings', 999) <= 14):
                conflicting_signals.append('extreme_put_call_ratio_near_earnings')
            
            return {
                'has_conflicts': len(conflicting_signals) > 0,
                'conflict_signals': conflicting_signals,
                'conflict_severity': len(conflicting_signals)
            }
            
        except Exception as e:
            logger.warning(f"Error detecting conflicting forces: {e}")
            return {'has_conflicts': False, 'error': str(e)}