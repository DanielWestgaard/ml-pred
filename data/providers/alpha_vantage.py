import requests

from data.providers.base_provider import BaseProvider


class ProviderAlphaVantage(BaseProvider):
    def __init__(self):
        """Initialize the provider with configuration."""
        pass
    
    def fetch_and_save_historical_data(self, symbol:str, timeframe:str, 
                            from_date:str, to_date:str):
        """Fetch historical market data."""
        print("inside provider")
        # replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
        url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey=9R74O21JDRQ0HPRM'
        r = requests.get(url)
        data = r.json()

        print(data)
        print("after provider")
    
    def convert_market_data_to_csv(self):
        """Convert response from API to .csv-file format."""
        pass
    
    def end_session(self):
        """Assuming all providers will have their own API. Some will need to properly close the session."""
        pass