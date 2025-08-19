import requests
import json
from utils.logging_utils import get_logger

logger = get_logger()

class TradingAPI:
    """
    Actual implementation of the possibilities with trading API. Called from alpaca.py.
    These methods and logic will wollow the naming provided about the Trading API in
    the documentation: https://docs.alpaca.markets/reference/
    
    TODO: In the future, if this file get overly populated with methods, it could be an idea
    to divide into files based on the "type". Eg. accounts.py, assets.py, etc.
    """
    
    def __init__(self, key_id, secret_key, print_responses):
        """Initialize the Alpaca Trading API."""
        
        self.key_id = key_id
        self.secret_key = secret_key
        self.print_responses = print_responses

        self.endpoint = "https://paper-api.alpaca.markets/v2"
        self.headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": f"{self.key_id}",
            "APCA-API-SECRET-KEY": f"{self.secret_key}"
        }
        
        # TEMPORARY CALLING ALL METHODS MANUALLY FOR TESTING PURPOSES
        self.get_all_orders()
    
    # ACCOUNTS
    
    def get_account(self) -> any:
        """
        Returns your account details.
        Documentation: https://docs.alpaca.markets/reference/getaccount-1
        
        Returns: The json response object
        """
        
        response = requests.get(url = f"{self.endpoint}/account", headers = self.headers)
        
        if self.print_responses:
            parsed_data = json.loads(response.text)
            logger.debug(f"Account response: {json.dumps(parsed_data, indent=4)}")
        
        return json.loads(response.text)
    
    # ASSETS
    
    # COROPORATE ACTIONS
    
    # ORDERS
    
    def create_an_order(self, symbol, qty, notion, side, type, time_in_force, limit_price, stop_price, trail_price, trail_):
        """
        Places a new order for the given account. An order request may be rejected if the account
        is not authorized for trading, or if the tradable balance is insufficient to fill the order.
        
        Documentation: https://docs.alpaca.markets/reference/postorder
        """
        payload = {
            "type": "market",
            "time_in_force": "day"
        }
        
        response = requests.get(url = f"{self.endpoint}/orders", json = payload, headers = self.headers)
        
        if self.print_responses:
            parsed_data = json.loads(response.text)
            logger.debug(f"Account response: {json.dumps(parsed_data, indent=4)}")
        
        return json.loads(response.text)
    
    def get_all_orders(self):
        """
        Retrieves a list of orders for the account, filtered by the supplied query parameters.
        
        Documentation: https://docs.alpaca.markets/reference/getallorders-1
        """
        response = requests.get(url = f"{self.endpoint}/orders", headers = self.headers)
        
        if self.print_responses:
            parsed_data = json.loads(response.text)
            logger.debug(f"Account response: {json.dumps(parsed_data, indent=4)}")
        
        return json.loads(response.text)
    
    # POSITIONS
    
    # PORTFOLIO HISTORY
    
    # WATCHLISTS
    
    # ACCOUNT CONFIGURATIONS
    
    # ACCOUNT ACTIVITIES
    
    # CALENDAR
    
    # CLOCK
    
    # CRYPTO FUNDING