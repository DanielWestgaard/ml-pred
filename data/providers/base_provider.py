from abc import ABC, abstractmethod


class ProviderInterface(ABC):
    """Interface class to outline needed classes for different historical market data providers."""
    
    @abstractmethod
    def __init__(self):
        """Initialize the provider with configuration."""
        pass
    
    @abstractmethod
    def fetch_historical_data(self, symbol:str, timeframe:str, 
                            from_date:str, to_date:str):
        """Fetch historical market data."""
        pass
    
    @abstractmethod
    def convert_market_data_to_csv(self):
        """Convert response from API to .csv-file format."""
        pass