from abc import abstractmethod
from datetime import datetime
import json
from typing import Dict, List, Optional
from broker.alpaca.tradin_api.trading_api import TradingAPI
import pandas as pd
from broker.base_interface import BaseBroker
from utils import shared_utils
from utils.logging_utils import get_logger

logger = get_logger()

class Alpaca(BaseBroker):
    """
    Used to interact with implemente API endpoints.
    
    Note: Alpaca have verious API's for different purposes: https://docs.alpaca.markets/
    I have implemented ...
    """
    
    # Global defined url variable
    
    def __init__(self, print_responses:bool = False):
        """
        Initialize the Alpaca API's.
        
        Args:
            - print_responses: True will print out the responses in the terminal. 
        """
        
        self.print_responses = print_responses
        
        try: 
            # Getting secrets
            self.secrets = shared_utils.load_secrets(desired_keys={'alpaca_paper_account_key_id', 'alpaca_paper_account_secret_key'})
            self.key_id = self.secrets.get('alpaca_paper_account_key_id')
            self.secret_key = self.secrets.get('alpaca_paper_account_secret_key')
        except Exception as e:
            logger.warning(f"Could not initiate Alpaca with secrets. Error: {e}")
        
        # Initialize the trading API
        self.trading_api = TradingAPI(
            key_id=self.key_id, 
            secret_key=self.secret_key, 
            print_responses=self.print_responses
        )

    # =================== SESSION METHODS ==================
    
    
    def start_session(email, password, api_key, use_encryption=True, print_answer=False):
        """Start a new session with the broker."""
        pass
    
    def end_session(X_SECURITY_TOKEN, CST):
        """End current session with the broker."""
        pass
    
        # ==================== DATA METHODS ====================
    
    
    def get_historical_data(self, 
                           epic: str, 
                           timeframe: str, 
                           start_date: datetime, 
                           end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Fetch historical OHLCV data.
        
        Args:
            symbol: The market symbol
            timeframe: Timeframe (e.g., "1m", "5m", "1h", "1d")
            start_date: Start date for historical data
            end_date: End date for historical data (optional)
            
        Returns:
            DataFrame with historical data
        """
        pass
    
    # ==================== ACCOUNT METHODS ====================
    
    def get_account_balance(self) -> Dict:
        """Get account balance information.
        
        Returns:
            Dictionary with balance information
        """
        return self.trading_api.get_account()    
    
    def get_positions(self) -> List[Dict]:
        """Get current open positions.
        
        Returns:
            List of position dictionaries
        """
        result = self.trading_api.get_all_positions()
        if isinstance(result, list):
            return result
        elif isinstance(result, dict) and "error" not in result:
            return result if isinstance(result, list) else []
        else:
            logger.error(f"Error getting positions: {result}")
            return []
    
    def get_orders(self) -> List[Dict]:
        """Get current pending orders.
        
        Returns:
            List of order dictionaries
        """
        result = self.trading_api.get_all_orders(status="open")
        if isinstance(result, list):
            return result
        elif isinstance(result, dict) and "error" not in result:
            return result if isinstance(result, list) else []
        else:
            logger.error(f"Error getting orders: {result}")
            return []
    
    # ==================== TRADING METHODS ====================
    
    
    def place_market_order(self, 
                          symbol: str, 
                          side: str, 
                          quantity: float,
                          take_profit: Optional[float] = None,
                          stop_loss: Optional[float] = None) -> Dict:
        """Place a market order.
        
        Args:
            symbol: The market symbol
            side: "buy" or "sell"
            quantity: Order quantity
            take_profit: Take profit price (optional)
            stop_loss: Stop loss price (optional)
            
        Returns:
            Order information dictionary
        """
        if take_profit or stop_loss:
            return self.trading_api.place_bracket_order(
                symbol=symbol,
                side=side,
                qty=quantity,
                take_profit_price=take_profit,
                stop_loss_price=stop_loss
            )
        else:
            return self.trading_api.place_market_order(
                symbol=symbol,
                side=side,
                qty=quantity
            )
    
    
    def place_limit_order(self,
                         symbol: str,
                         side: str,
                         quantity: float,
                         price: float,
                         take_profit: Optional[float] = None,
                         stop_loss: Optional[float] = None) -> Dict:
        """Place a limit order.
        
        Args:
            symbol: The market symbol
            side: "buy" or "sell"
            quantity: Order quantity
            price: Limit price
            take_profit: Take profit price (optional)
            stop_loss: Stop loss price (optional)
            
        Returns:
            Order information dictionary
        """
        if take_profit or stop_loss:
            # For bracket orders with limit entry, we need to use create_an_order directly
            take_profit_dict = {"limit_price": str(take_profit)} if take_profit else None
            stop_loss_dict = {"stop_price": str(stop_loss)} if stop_loss else None
            
            return self.trading_api.create_an_order(
                symbol=symbol,
                side=side,
                qty=quantity,
                order_type="limit",
                limit_price=price,
                order_class="bracket",
                take_profit=take_profit_dict,
                stop_loss=stop_loss_dict
            )
        else:
            return self.trading_api.place_limit_order(
                symbol=symbol,
                side=side,
                qty=quantity,
                limit_price=price
            )
    
    
    def close_order(self, order_id: str) -> bool:
        """Cancel an existing order.
        
        Args:
            order_id: The ID of the order to cancel
            
        Returns:
            True if successful, False otherwise
        """
        result = self.trading_api.cancel_order(order_id)
        return "success" in result and result["success"]
    
    
    def modify_position(self,
                       position_id: str,
                       take_profit: Optional[float] = None,
                       stop_loss: Optional[float] = None) -> bool:
        """Modify an existing position.
        
        Args:
            position_id: The ID of the position to modify
            take_profit: New take profit price (optional)
            stop_loss: New stop loss price (optional)
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    
    def sub_live_market_data(self,
                             symbol: str,
                             timeframe: str) -> bool:
        """Subscribe to live market data.
        
        Args:
            symbol: The symbol, or epic, you want to recieve data about. Eg. GOLD, GBPUSD, US100, etc.
            timeframe: The timeframe (/candles) of the symbol. Eg. 1M, 5M, 1H, 4H, etc.
        Returns:
            ...
        """
        pass