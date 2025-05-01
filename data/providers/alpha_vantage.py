import json
import logging
import requests

from data.providers.base_provider import BaseProvider
import utils.shared_utils as shared_utils


class ProviderAlphaVantage(BaseProvider):
    def __init__(self):
        """
        Initialize the provider with configuration.
        
        Note: The api key has reportedly a 25 request limit per day, but this is actually only
        related to you IP-adress and not the key, so a VPN can help with this.
        """
        logging.info("Initializing Alpha Vantage Provider")
        self.secrets = shared_utils.load_secrets({"alpha_vantage_free_api_key"})
        self.api_key = self.secrets.get("alpha_vantage_free_api_key")
    
    def fetch_and_save_historical_data(self, symbol:str, timeframe:str, 
                                        month:str, print_answer:bool = False):
        """
        Fetch historical market data.
        
        Inputs:
            symbol: The name of the equity of your choice. Eg. symbol=GBPUSD
            timeframe: Time interval between two consecutive data points in the time series: 1min, 5min, 15min, 30min, 60min
            
        """
        
        # replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
        #       https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&month=2009-01&outputsize=full&apikey=demo
        url = ("https://www.alphavantage.co/query?"
                "function=TIME_SERIES_INTRADAY"     # 
                f"&symbol={symbol}"
                # "&extended_hours=true"              # 
                f"&interval={timeframe}"
                f"&apikey={self.api_key}"    
                f"&month={month}"                   #
                "&outputsize=full"                  # full returns full-length timeseries of 20+ years historical data, compact is only the latest 100 data points
                # "&adjusted=false"
                )
        r = requests.get(url)
        data = r.json()

        if print_answer:
            print(json.dumps(data, indent=4))
            
        # Maybe use the same methods from broker/capital_com/rest_api/markets_info.py and fetch_and_save_historical_prices()?
          # Generate filename
          # Convert json response to csv file
    
    def convert_market_data_to_csv(self):
        """Convert response from API to .csv-file format."""
        pass
    
    def end_session(self):
        """Assuming all providers will have their own API. Some will need to properly close the session."""
        pass