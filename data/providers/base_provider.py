from abc import ABC, abstractmethod


class BaseProvider(ABC):
    """Interface class to outline needed classes for different historical market data providers."""
    
    @abstractmethod
    def __init__(self):
        """Initialize the provider with configuration."""
        pass
    
    @abstractmethod
    def fetch_and_save_historical_data(self, symbol:str, timeframe:str, 
                            from_date:str, to_date:str):
        """Fetch historical market data."""
        pass
    
    @abstractmethod
    def convert_market_data_to_csv(self):
        """Convert response from API to .csv-file format."""
        pass
    
    @abstractmethod
    def end_session(self):
        """Assuming all providers will have their own API. Some will need to properly close the session."""
        pass