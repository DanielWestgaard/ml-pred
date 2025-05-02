import json
import logging
import os
import requests
from datetime import date

from data.providers.base_provider import BaseProvider
import utils.shared_utils as shared_utils
import config.config as config
import utils.data_utils as data_util
import utils.alpha_vantage_utils as alpha_utils


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
    
    def fetch_and_save_historical_data(self, symbol:str, timeframe:str, month:str,
                                       print_answer:bool = False, store_answer:bool = False):
        """
        Fetch historical intraday market data.
        
        Inputs:
            symbol: The name of the equity of your choice. Eg. symbol=GBPUSD
            timeframe: Time interval between two consecutive data points in the time series: 1min, 5min, 15min, 30min, 60min
            month: String format of YYYY-MM
        """
        # Fetch the data using Alpha Vantage API
        json_data = self.fetch_historical_data(symbol=symbol, timeframe=timeframe, month=month, print_answer=print_answer, store_answer=store_answer)
        
        # Extremly simple file naming convention, but will focus on this when I chose to pay for alpha vantage
        file_name = shared_utils.get_available_filename(directory=config.ALPVAN_RAW_DATA_DIR, filename="testing.csv")
        
        # Convert the json data into appropriate csv-format
        alpha_utils.convert_market_data_to_csv(json_data=json_data, output_file_name=file_name)
        
    def fetch_historical_data(self, symbol:str, timeframe:str, month:str,
                                       print_answer:bool = False, store_answer:bool = False):
        """Only fetch an return market data."""
        url = ("https://www.alphavantage.co/query?"
                "function=TIME_SERIES_INTRADAY"     
                f"&symbol={symbol}"
                "&extended_hours=true"              
                f"&interval={timeframe}"            # Timeframe
                f"&apikey={self.api_key}"           # Your API Key
                f"&month={month}"                   # The start "time" of data. Format YYYY-MM
                "&outputsize=full"                  # full returns full-length timeseries of 20+ years historical data, compact is only the latest 100 data points
                # "&adjusted=false"
                )
        r = requests.get(url)
        data = r.json()
        
        if print_answer:
            print(json.dumps(data, indent=4))
            
        if store_answer:
            output_dir = config.ALPVAN_RESPONSE_JSON_DIR
            shared_utils.ensure_path_exists(output_dir)
            output_dir = os.path.join(output_dir, 'response.json')
            with open(output_dir, 'w') as f:
                json.dump(data, f, indent=4)
        return data
    
    def convert_market_data_to_csv():
        pass
        
    def end_session(self):
        """Assuming all providers will have their own API. Some will need to properly close the session."""
        pass