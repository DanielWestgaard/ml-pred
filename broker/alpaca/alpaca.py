from abc import abstractmethod
from datetime import datetime
import json
from typing import Dict, List, Optional, Union
from broker.alpaca.tradin_api.trading_api import TradingAPI
from broker.alpaca.market_data_api.historical_data import HistoricalDataAPI
from broker.alpaca.market_data_api.live_data import LiveDataAPI
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
        
        # Initialize the APIs
        self.trading_api = TradingAPI(
            key_id=self.key_id, 
            secret_key=self.secret_key, 
            print_responses=self.print_responses
        )
        
        self.historical_data_api = HistoricalDataAPI(
            key_id=self.key_id,
            secret_key=self.secret_key,
            print_responses=self.print_responses
        )
        
        self.live_data_api = LiveDataAPI(
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
                           start_date: Union[datetime, str], 
                           end_date: Optional[Union[datetime, str]] = None) -> pd.DataFrame:
        """Fetch historical OHLCV data.
        
        Args:
            epic: The market symbol (e.g., "AAPL", "SPY")
            timeframe: Timeframe (e.g., "1Min", "5Min", "1Hour", "1Day")
            start_date: Start date for historical data (datetime object or ISO string)
            end_date: End date for historical data (datetime object or ISO string, optional)
            
        Returns:
            DataFrame with historical data
        """
        # Convert timeframe to Alpaca format if needed
        timeframe_mapping = {
            "1m": "1Min", "5m": "5Min", "15m": "15Min", "30m": "30Min",
            "1h": "1Hour", "1d": "1Day", "1w": "1Week", "1M": "1Month"
        }
        alpaca_timeframe = timeframe_mapping.get(timeframe, timeframe)
        
        # Handle both datetime objects and string inputs
        if isinstance(start_date, str):
            # If string doesn't end with 'Z', add it for UTC timezone
            start_str = start_date if start_date.endswith('Z') else f"{start_date}Z"
        else:
            start_str = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        if end_date:
            if isinstance(end_date, str):
                end_str = end_date if end_date.endswith('Z') else f"{end_date}Z"
            else:
                end_str = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            end_str = None
        
        # Get bars data
        bars_response = self.historical_data_api.get_stock_bars(
            symbols=epic,
            timeframe=alpaca_timeframe,
            start=start_str,
            end=end_str
        )
        
        # Convert to DataFrame
        df = self.historical_data_api.bars_to_dataframe(bars_response, symbol=epic)
        return df if df is not None else pd.DataFrame()
    
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
    
    def cancel_all_orders(self) -> Dict:
        """Cancel all open orders.
        
        Returns:
            Dictionary containing cancellation results with counts
        """
        return self.trading_api.cancel_all_orders()
    
    def close_all_positions(self) -> Dict:
        """Close all open positions.
        
        Returns:
            Dictionary containing closure results with counts
        """
        return self.trading_api.close_all_positions()
    
    
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
                             timeframe: str = None) -> bool:
        """Subscribe to live market data.
        
        Args:
            symbol: The symbol you want to receive data about (e.g., "AAPL", "SPY")
            timeframe: The timeframe (optional, Alpaca provides real-time bars automatically)
            
        Returns:
            True if subscription successful, False otherwise
        """
        try:
            if not self.live_data_api.is_connected:
                self.live_data_api.connect()
            
            # Subscribe to all data types for comprehensive coverage
            self.live_data_api.subscribe_all_data([symbol])
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to live data for {symbol}: {e}")
            return False
    
    def start_live_stream(self, 
                         symbols: List[str], 
                         data_types: List[str] = None,
                         trade_handler=None,
                         quote_handler=None,
                         bar_handler=None):
        """
        Start live data streaming for multiple symbols.
        
        Args:
            symbols: List of symbols to stream
            data_types: Types of data to stream (["trades", "quotes", "bars"])
            trade_handler: Custom trade message handler
            quote_handler: Custom quote message handler  
            bar_handler: Custom bar message handler
        """
        self.live_data_api.start_simple_stream(
            symbols=symbols,
            data_types=data_types,
            trade_handler=trade_handler,
            quote_handler=quote_handler,
            bar_handler=bar_handler
        )
    
    def stop_live_stream(self):
        """Stop live data streaming and disconnect."""
        if self.live_data_api.is_connected:
            self.live_data_api.unsubscribe_all()
            self.live_data_api.disconnect()
            logger.info("Live data stream stopped")
    
    def add_live_data_handler(self, message_type: str, handler):
        """
        Add a custom handler for live data messages.
        
        Args:
            message_type: Type of message ("trades", "quotes", "bars", "status", "error")
            handler: Callback function
        """
        self.live_data_api.add_handler(message_type, handler)
    
    # ==================== DATA SAVING METHODS ====================
    
    def fetch_and_save_historical_data(self,
                                      symbol: str,
                                      timeframe: str = "1Day", 
                                      start_date: Union[datetime, str] = None,
                                      end_date: Union[datetime, str] = None,
                                      output_file: str = None,
                                      save_raw_data: bool = False) -> str:
        """
        Fetch historical data and save to CSV with proper directory structure.
        
        Args:
            symbol: The market symbol (e.g., "AAPL", "SPY")
            timeframe: Timeframe (e.g., "1Min", "5Min", "1Hour", "1Day")
            start_date: Start date for historical data (datetime object or ISO string)
            end_date: End date for historical data (datetime object or ISO string, optional)
            output_file: Custom output file path (optional)
            save_raw_data: Whether to save raw JSON response as well
            
        Returns:
            Path to the saved CSV file
        """
        # Convert timeframe to Alpaca format if needed
        timeframe_mapping = {
            "1m": "1Min", "5m": "5Min", "15m": "15Min", "30m": "30Min",
            "1h": "1Hour", "1d": "1Day", "1w": "1Week", "1M": "1Month"
        }
        alpaca_timeframe = timeframe_mapping.get(timeframe, timeframe)
        
        # Handle both datetime objects and string inputs
        if start_date:
            if isinstance(start_date, str):
                start_str = start_date if start_date.endswith('Z') else f"{start_date}Z"
            else:
                start_str = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            start_str = None
        
        if end_date:
            if isinstance(end_date, str):
                end_str = end_date if end_date.endswith('Z') else f"{end_date}Z"
            else:
                end_str = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            end_str = None
        
        return self.historical_data_api.fetch_and_save_historical_bars(
            symbols=symbol,
            timeframe=alpaca_timeframe,
            start=start_str,
            end=end_str,
            output_file=output_file,
            save_raw_data=save_raw_data
        )
    
    def fetch_and_save_basic_plan_data(self,
                                      symbol: str,
                                      timeframe: str = "1Day",
                                      days_back: int = 30,
                                      save_raw_data: bool = False) -> str:
        """
        Fetch and save data optimized for basic plan (avoids recent data restrictions).
        
        Args:
            symbol: The market symbol (e.g., "AAPL", "SPY")
            timeframe: Timeframe (e.g., "1Min", "5Min", "1Hour", "1Day")
            days_back: Number of days to look back from a week ago
            save_raw_data: Whether to save raw JSON response as well
            
        Returns:
            Path to the saved CSV file
        """
        # Convert timeframe to Alpaca format if needed
        timeframe_mapping = {
            "1m": "1Min", "5m": "5Min", "15m": "15Min", "30m": "30Min",
            "1h": "1Hour", "1d": "1Day", "1w": "1Week", "1M": "1Month"
        }
        alpaca_timeframe = timeframe_mapping.get(timeframe, timeframe)
        
        return self.historical_data_api.fetch_and_save_basic_plan_data(
            symbols=symbol,
            timeframe=alpaca_timeframe,
            days_back=days_back,
            save_raw_data=save_raw_data
        )