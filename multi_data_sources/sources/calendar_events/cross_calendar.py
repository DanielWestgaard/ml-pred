from typing import Optional
from multi_data_sources.sources.ds_base import DataSourceBase
from multi_data_sources.sources.calendar_events.complexities.earnings import Earnings
from multi_data_sources.sources.calendar_events.complexities.options_expiration import OptionsExpiration


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
    
    def get_historical_data(self):
        """Fetch historical to store in storage/"""
        
        
    def subscribe_live_data(self):
        """
        Subscribeto live date, using the source's approach to this.
        """
        pass