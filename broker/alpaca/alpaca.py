from broker.base_interface import BaseBroker

class Alpaca(BaseBroker):
    """
    Used to interact with implemente API endpoints.
    
    Note: Alpaca have verious API's for different purposes: https://docs.alpaca.markets/
    I have implemented ...
    """
    
    def __init__(self):
        """Initialize the Alpaca API's."""