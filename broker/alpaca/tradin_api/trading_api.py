import requests
import json
from utils.logging_utils import get_logger

logger = get_logger()

class TradingAPI:
    """
    Actual implementation of the possibilities with trading API. Called from alpaca.py.
    These methods and logic will wollow the naming provided about the Trading API in
    the documentation: https://docs.alpaca.markets/reference/
    """
    
    def __init__(self, key_id, secret_key, print_responses):
        """Initialize the Alpaca Trading API."""
        
        self.endpoint = "https://paper-api.alpaca.markets/v2"

        self.key_id = key_id
        self.secret_key = secret_key
        self.print_responses = print_responses
        
        # TEMPORARY CALLING ALL METHODS MANUALLY FOR TESTING PURPOSES
        self.get_account()
    
    # ACCOUNTS
    
    def get_account(self):
        """
        Returns your account details.
        Link/documentation: https://docs.alpaca.markets/reference/getaccount-1
        """
        headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": f"{self.key_id}",
            "APCA-API-SECRET-KEY": f"{self.secret_key}"
        }
        
        response = requests.get(url = f"{self.endpoint}/account", headers = headers)

        if self.print_responses:
            parsed_data = json.loads(response.text)
            logger.debug(f"Account response: {json.dumps(parsed_data, indent=4)}")
    
    # ASSETS
    
    # COROPORATE ACTIONS
    
    # ORDERS
    
    # POSITIONS
    
    # PORTFOLIO HISTORY
    
    # WATCHLISTS
    
    # ACCOUNT CONFIGURATIONS
    
    # ACCOUNT ACTIVITIES
    
    # CALENDAR
    
    # CLOCK
    
    # CRYPTO FUNDING