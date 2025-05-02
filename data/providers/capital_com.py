from data.providers.base_provider import BaseProvider
from broker.capital_com.capitalcom import CapitalCom

class ProviderCapitalCom(BaseProvider):
    """A Provider class that uses Capital.com's API solely to get historical market data."""
    
    def __init__(self):
        """Initialize the provider with configuration."""
        self.broker = CapitalCom()
        self.broker.start_session()
        pass
    
    def fetch_and_save_historical_data(self, symbol:str, timeframe:str,
                                        from_date:str, to_date:str,  # "2025-04-15T00:00:00"
                                        print_answer:bool = False):
        """Fetch historical market data."""
        self.broker.fetch_and_save_historical_prices(epic=symbol, resolution=timeframe,
                                                     from_date=from_date, to_date=to_date,
                                                     print_answer=print_answer)
    
    def convert_market_data_to_csv(self):
        """Convert response from API to .csv-file format."""
        pass
    
    def end_session(self):
        """End session with Capital.com API."""
        self.broker.end_session()