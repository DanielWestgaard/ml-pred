from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
from datetime import datetime


class BrokerInterface(ABC):
    """Abstract interface class for broker implementations."""
    
    @abstractmethod
    def __init__(self):
        """Initialize the broker with configuration."""
        pass
    
    # =================== SESSION METHODS ==================
    
    @abstractmethod
    def start_session(email, password, api_key, use_encryption=True, print_answer=False):
        """Start a new session with the broker."""
        pass
    
    @abstractmethod
    def end_session(X_SECURITY_TOKEN, CST):
        """End current session with the broker."""
        pass
    
        # ==================== DATA METHODS ====================
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Dict]:
        """Get current open positions.
        
        Returns:
            List of position dictionaries
        """
        pass
    
    @abstractmethod
    def get_orders(self) -> List[Dict]:
        """Get current pending orders.
        
        Returns:
            List of order dictionaries
        """
        pass
    
    # ==================== TRADING METHODS ====================
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def close_order(self, order_id: str) -> bool:
        """Cancel an existing order.
        
        Args:
            order_id: The ID of the order to cancel
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
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
    
    @abstractmethod
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
