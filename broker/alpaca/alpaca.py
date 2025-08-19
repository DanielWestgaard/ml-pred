from broker.base_interface import BaseBroker
from utils import shared_utils
import requests
from utils.logging_utils import get_logger

logger = get_logger()

class Alpaca(BaseBroker):
    """
    Used to interact with implemente API endpoints.
    
    Note: Alpaca have verious API's for different purposes: https://docs.alpaca.markets/
    I have implemented ...
    """
    
    # Global defined url variable
    url = "https://paper-api.alpaca.markets/v2/account"
    
    def __init__(self):
        """Initialize the Alpaca API's."""
        
        try: 
            # Getting secrets
            self.secrets = shared_utils.load_secrets(desired_keys={'API_KEY_CAP', 'PASSWORD_CAP', 'EMAIL'})
            print(self.secrets)
        except Exception as e:
            logger.warning(f"Could not initiate BaseBroker with secrets. Error: {e}")
