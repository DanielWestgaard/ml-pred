from abc import ABC, abstractmethod
from typing import Any, Optional
from datetime import datetime


class DataSourceBase(ABC):
    """Base class for all data source implementations."""
    
    def __init__(self, symbol: str, stock: bool = False, from_date: Optional[str] = None,
                 to_date: Optional[str] = None, purpose: str = "historical"):
        """
        Initialize base data source.
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'GBPUSD')
            from_date: Start date for historical data (YYYY-MM-DD format)
            to_date: End date for historical data (YYYY-MM-DD format)
            purpose: 'historical' or 'live' data collection
        """
        self.symbol = symbol
        self.from_date = from_date
        self.to_date = to_date
        self.purpose = purpose
        self.initialized_at = datetime.now()
    
    @abstractmethod
    def get_historical_data(self) -> Any:
        """
        Fetch historical data to store in storage/.
        
        Returns:
            Historical data in appropriate format (DataFrame, dict, etc.)
        """
        pass
        
    @abstractmethod
    def subscribe_live_data(self) -> Any:
        """
        Subscribe to live data using the source's approach.
        
        Returns:
            Live data subscription object or stream
        """
        pass
    
    def get_source_info(self) -> dict:
        """Get metadata about this data source."""
        return {
            'symbol': self.symbol,
            'from_date': self.from_date,
            'to_date': self.to_date,
            'purpose': self.purpose,
            'initialized_at': self.initialized_at.isoformat(),
            'source_type': self.__class__.__name__
        }