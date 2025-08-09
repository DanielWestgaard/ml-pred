from multi_data_sources.sources.ds_base import DataSourceBase
from multi_data_sources.sources.calendar_events.complexities.earnings import Earnings
from multi_data_sources.sources.calendar_events.complexities.options_expiration import OptionsExpiration


class CrossCalendar(DataSourceBase):
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
    
    def __init__(self, path_to_config:str = None):
        """
        ...
        """
        # Read configuration
        
        # Set variables
    
    
    def get_historical_data(self):
        """Fetch historical to store in storage/"""
        
        
    def subscribe_live_data(self):
        """
        Subscribeto live date, using the source's approach to this.
        """
        pass