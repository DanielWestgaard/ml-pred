import json
import asyncio
import websockets
import threading
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from utils.logging_utils import get_logger

logger = get_logger()

class LiveDataAPI:
    """
    Alpaca Live Market Data WebSocket API implementation.
    
    Provides real-time streaming of market data including:
    - Real-time stock trades
    - Real-time stock quotes  
    - Real-time stock bars
    - Market status updates
    
    Documentation: https://docs.alpaca.markets/docs/real-time-market-data
    """
    
    def __init__(self, key_id: str, secret_key: str, print_responses: bool = False):
        """
        Initialize the Live Data API.
        
        Args:
            key_id: Alpaca API key ID
            secret_key: Alpaca API secret key
            print_responses: Whether to print WebSocket messages for debugging
        """
        self.key_id = key_id
        self.secret_key = secret_key
        self.print_responses = print_responses
        
        # WebSocket endpoints
        self.ws_endpoint = "wss://stream.data.alpaca.markets/v2/iex"  # IEX feed (basic)
        self.ws_endpoint_sip = "wss://stream.data.alpaca.markets/v2/sip"  # SIP feed (premium)
        
        # Connection state
        self.websocket = None
        self.is_connected = False
        self.is_authenticated = False
        self.subscriptions = {
            "trades": set(),
            "quotes": set(), 
            "bars": set()
        }
        
        # Event handlers
        self.message_handlers = {
            "trades": [],
            "quotes": [],
            "bars": [],
            "status": [],
            "error": []
        }
        
        # Background task management
        self.event_loop = None
        self.websocket_task = None
        self.thread = None
        
        logger.info("Alpaca Live Data API initialized successfully")
    
    def set_premium_feed(self, premium: bool = True):
        """
        Switch between IEX (basic) and SIP (premium) data feeds.
        
        Args:
            premium: True for SIP feed (requires subscription), False for IEX feed
        """
        if premium:
            self.ws_endpoint = "wss://stream.data.alpaca.markets/v2/sip"
            logger.info("Switched to SIP premium data feed")
        else:
            self.ws_endpoint = "wss://stream.data.alpaca.markets/v2/iex"
            logger.info("Switched to IEX basic data feed")
    
    def add_handler(self, message_type: str, handler: Callable):
        """
        Add a message handler for specific data types.
        
        Args:
            message_type: Type of message ("trades", "quotes", "bars", "status", "error")
            handler: Callback function to handle the message
        """
        if message_type in self.message_handlers:
            self.message_handlers[message_type].append(handler)
            logger.info(f"Added handler for {message_type} messages")
        else:
            logger.error(f"Invalid message type: {message_type}")
    
    def remove_handler(self, message_type: str, handler: Callable):
        """
        Remove a message handler.
        
        Args:
            message_type: Type of message
            handler: Handler function to remove
        """
        if message_type in self.message_handlers:
            try:
                self.message_handlers[message_type].remove(handler)
                logger.info(f"Removed handler for {message_type} messages")
            except ValueError:
                logger.warning(f"Handler not found for {message_type}")
    
    async def _authenticate(self):
        """Authenticate the WebSocket connection."""
        auth_message = {
            "action": "auth",
            "key": self.key_id,
            "secret": self.secret_key
        }
        
        await self.websocket.send(json.dumps(auth_message))
        logger.info("Sent authentication message")
    
    async def _handle_message(self, message: str):
        """
        Handle incoming WebSocket messages.
        
        Args:
            message: Raw JSON message from WebSocket
        """
        try:
            data = json.loads(message)
            
            if self.print_responses:
                logger.debug(f"Received message: {json.dumps(data, indent=2)}")
            
            # Handle different message types
            for msg in data if isinstance(data, list) else [data]:
                msg_type = msg.get("T")
                
                if msg_type == "success":
                    if msg.get("msg") == "authenticated":
                        self.is_authenticated = True
                        logger.info("WebSocket authenticated successfully")
                        # Call status handlers
                        for handler in self.message_handlers["status"]:
                            try:
                                handler(msg)
                            except Exception as e:
                                logger.error(f"Error in status handler: {e}")
                
                elif msg_type == "error":
                    logger.error(f"WebSocket error: {msg}")
                    for handler in self.message_handlers["error"]:
                        try:
                            handler(msg)
                        except Exception as e:
                            logger.error(f"Error in error handler: {e}")
                
                elif msg_type == "subscription":
                    logger.info(f"Subscription update: {msg}")
                    for handler in self.message_handlers["status"]:
                        try:
                            handler(msg)
                        except Exception as e:
                            logger.error(f"Error in status handler: {e}")
                
                # Market data messages
                elif msg_type == "t":  # Trade
                    for handler in self.message_handlers["trades"]:
                        try:
                            handler(msg)
                        except Exception as e:
                            logger.error(f"Error in trades handler: {e}")
                
                elif msg_type == "q":  # Quote
                    for handler in self.message_handlers["quotes"]:
                        try:
                            handler(msg)
                        except Exception as e:
                            logger.error(f"Error in quotes handler: {e}")
                
                elif msg_type == "b":  # Bar
                    for handler in self.message_handlers["bars"]:
                        try:
                            handler(msg)
                        except Exception as e:
                            logger.error(f"Error in bars handler: {e}")
                
                else:
                    logger.debug(f"Unhandled message type: {msg_type}")
                    
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode WebSocket message: {e}")
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
    
    async def _websocket_loop(self):
        """Main WebSocket connection loop."""
        try:
            logger.info(f"Connecting to {self.ws_endpoint}")
            
            async with websockets.connect(self.ws_endpoint) as websocket:
                self.websocket = websocket
                self.is_connected = True
                logger.info("WebSocket connected successfully")
                
                # Authenticate
                await self._authenticate()
                
                # Wait for authentication
                auth_timeout = 10
                for _ in range(auth_timeout):
                    if self.is_authenticated:
                        break
                    await asyncio.sleep(1)
                
                if not self.is_authenticated:
                    raise Exception("Authentication failed")
                
                # Listen for messages
                async for message in websocket:
                    await self._handle_message(message)
                    
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            self.is_connected = False
            self.is_authenticated = False
            self.websocket = None
    
    def _run_websocket_in_thread(self):
        """Run WebSocket in a separate thread with its own event loop."""
        self.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.event_loop)
        
        try:
            self.websocket_task = self.event_loop.create_task(self._websocket_loop())
            self.event_loop.run_until_complete(self.websocket_task)
        except Exception as e:
            logger.error(f"Error in WebSocket thread: {e}")
        finally:
            self.event_loop.close()
    
    def connect(self):
        """Start the WebSocket connection in a background thread."""
        if self.is_connected:
            logger.warning("Already connected to WebSocket")
            return
        
        self.thread = threading.Thread(target=self._run_websocket_in_thread, daemon=True)
        self.thread.start()
        
        # Wait for connection
        for _ in range(30):  # 30 second timeout
            if self.is_connected and self.is_authenticated:
                logger.info("WebSocket connection established")
                return
            threading.Event().wait(1)
        
        logger.error("Failed to establish WebSocket connection")
    
    def disconnect(self):
        """Close the WebSocket connection."""
        if not self.is_connected:
            logger.warning("Not connected to WebSocket")
            return
        
        try:
            if self.event_loop and self.websocket_task:
                self.event_loop.call_soon_threadsafe(self.websocket_task.cancel)
            
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=5)
            
            logger.info("WebSocket disconnected")
        except Exception as e:
            logger.error(f"Error disconnecting WebSocket: {e}")
    
    async def _send_subscription_message(self, action: str, data_type: str, symbols: List[str]):
        """Send subscription/unsubscription messages."""
        if not self.is_connected or not self.websocket:
            logger.error("WebSocket not connected")
            return
        
        message = {
            "action": action,
            data_type: symbols
        }
        
        await self.websocket.send(json.dumps(message))
        logger.info(f"Sent {action} message for {data_type}: {symbols}")
    
    def _send_subscription_sync(self, action: str, data_type: str, symbols: List[str]):
        """Send subscription message synchronously."""
        if not self.event_loop:
            logger.error("Event loop not available")
            return
        
        future = asyncio.run_coroutine_threadsafe(
            self._send_subscription_message(action, data_type, symbols),
            self.event_loop
        )
        
        try:
            future.result(timeout=5)
        except Exception as e:
            logger.error(f"Error sending subscription message: {e}")
    
    def subscribe_trades(self, symbols: List[str]):
        """
        Subscribe to real-time trade data for symbols.
        
        Args:
            symbols: List of stock symbols (e.g., ["AAPL", "MSFT"])
        """
        if not isinstance(symbols, list):
            symbols = [symbols]
        
        self._send_subscription_sync("subscribe", "trades", symbols)
        self.subscriptions["trades"].update(symbols)
        logger.info(f"Subscribed to trades for: {symbols}")
    
    def subscribe_quotes(self, symbols: List[str]):
        """
        Subscribe to real-time quote data for symbols.
        
        Args:
            symbols: List of stock symbols (e.g., ["AAPL", "MSFT"])
        """
        if not isinstance(symbols, list):
            symbols = [symbols]
        
        self._send_subscription_sync("subscribe", "quotes", symbols)
        self.subscriptions["quotes"].update(symbols)
        logger.info(f"Subscribed to quotes for: {symbols}")
    
    def subscribe_bars(self, symbols: List[str]):
        """
        Subscribe to real-time minute bar data for symbols.
        
        Args:
            symbols: List of stock symbols (e.g., ["AAPL", "MSFT"])
        """
        if not isinstance(symbols, list):
            symbols = [symbols]
        
        self._send_subscription_sync("subscribe", "bars", symbols)
        self.subscriptions["bars"].update(symbols)
        logger.info(f"Subscribed to bars for: {symbols}")
    
    def unsubscribe_trades(self, symbols: List[str]):
        """Unsubscribe from trade data for symbols."""
        if not isinstance(symbols, list):
            symbols = [symbols]
        
        self._send_subscription_sync("unsubscribe", "trades", symbols)
        self.subscriptions["trades"].difference_update(symbols)
        logger.info(f"Unsubscribed from trades for: {symbols}")
    
    def unsubscribe_quotes(self, symbols: List[str]):
        """Unsubscribe from quote data for symbols."""
        if not isinstance(symbols, list):
            symbols = [symbols]
        
        self._send_subscription_sync("unsubscribe", "quotes", symbols)
        self.subscriptions["quotes"].difference_update(symbols)
        logger.info(f"Unsubscribed from quotes for: {symbols}")
    
    def unsubscribe_bars(self, symbols: List[str]):
        """Unsubscribe from bar data for symbols."""
        if not isinstance(symbols, list):
            symbols = [symbols]
        
        self._send_subscription_sync("unsubscribe", "bars", symbols)
        self.subscriptions["bars"].difference_update(symbols)
        logger.info(f"Unsubscribed from bars for: {symbols}")
    
    def unsubscribe_all(self):
        """Unsubscribe from all data streams."""
        all_symbols = set()
        for symbols in self.subscriptions.values():
            all_symbols.update(symbols)
        
        if all_symbols:
            all_symbols_list = list(all_symbols)
            self.unsubscribe_trades(all_symbols_list)
            self.unsubscribe_quotes(all_symbols_list)
            self.unsubscribe_bars(all_symbols_list)
        
        logger.info("Unsubscribed from all data streams")
    
    def get_subscriptions(self) -> Dict[str, List[str]]:
        """
        Get current subscriptions.
        
        Returns:
            Dictionary containing current subscriptions by data type
        """
        return {
            "trades": list(self.subscriptions["trades"]),
            "quotes": list(self.subscriptions["quotes"]),
            "bars": list(self.subscriptions["bars"])
        }
    
    def is_subscribed(self, symbol: str, data_type: str = None) -> bool:
        """
        Check if subscribed to a symbol for specific data type or any data type.
        
        Args:
            symbol: Stock symbol to check
            data_type: Specific data type to check ("trades", "quotes", "bars") or None for any
            
        Returns:
            True if subscribed, False otherwise
        """
        if data_type:
            return symbol in self.subscriptions.get(data_type, set())
        else:
            # Check if subscribed to any data type
            for symbols in self.subscriptions.values():
                if symbol in symbols:
                    return True
            return False
    
    # CONVENIENCE METHODS
    
    def subscribe_all_data(self, symbols: List[str]):
        """
        Convenience method to subscribe to all data types for symbols.
        
        Args:
            symbols: List of stock symbols
        """
        self.subscribe_trades(symbols)
        self.subscribe_quotes(symbols)
        self.subscribe_bars(symbols)
        logger.info(f"Subscribed to all data types for: {symbols}")
    
    def start_simple_stream(self, 
                           symbols: List[str], 
                           data_types: List[str] = None,
                           trade_handler: Callable = None,
                           quote_handler: Callable = None,
                           bar_handler: Callable = None):
        """
        Start a simple streaming session with default handlers.
        
        Args:
            symbols: List of stock symbols to stream
            data_types: List of data types ("trades", "quotes", "bars") or None for all
            trade_handler: Custom trade handler function
            quote_handler: Custom quote handler function  
            bar_handler: Custom bar handler function
        """
        # Set default data types if not specified
        if data_types is None:
            data_types = ["trades", "quotes", "bars"]
        
        # Add default handlers if not provided
        if "trades" in data_types and trade_handler:
            self.add_handler("trades", trade_handler)
        elif "trades" in data_types:
            self.add_handler("trades", self._default_trade_handler)
        
        if "quotes" in data_types and quote_handler:
            self.add_handler("quotes", quote_handler)
        elif "quotes" in data_types:
            self.add_handler("quotes", self._default_quote_handler)
        
        if "bars" in data_types and bar_handler:
            self.add_handler("bars", bar_handler)
        elif "bars" in data_types:
            self.add_handler("bars", self._default_bar_handler)
        
        # Connect and subscribe
        self.connect()
        
        # Subscribe to requested data types
        if "trades" in data_types:
            self.subscribe_trades(symbols)
        if "quotes" in data_types:
            self.subscribe_quotes(symbols)
        if "bars" in data_types:
            self.subscribe_bars(symbols)
        
        logger.info(f"Started simple stream for {symbols} with data types: {data_types}")
    
    def _default_trade_handler(self, message: Dict):
        """Default handler for trade messages."""
        symbol = message.get("S", "unknown")
        price = message.get("p", 0)
        size = message.get("s", 0)
        timestamp = message.get("t", "")
        
        logger.info(f"TRADE - {symbol}: ${price} x {size} at {timestamp}")
    
    def _default_quote_handler(self, message: Dict):
        """Default handler for quote messages."""
        symbol = message.get("S", "unknown")
        bid_price = message.get("bp", 0)
        ask_price = message.get("ap", 0)
        bid_size = message.get("bs", 0)
        ask_size = message.get("as", 0)
        
        logger.info(f"QUOTE - {symbol}: Bid ${bid_price} x {bid_size}, Ask ${ask_price} x {ask_size}")
    
    def _default_bar_handler(self, message: Dict):
        """Default handler for bar messages."""
        symbol = message.get("S", "unknown")
        open_price = message.get("o", 0)
        high_price = message.get("h", 0)
        low_price = message.get("l", 0)
        close_price = message.get("c", 0)
        volume = message.get("v", 0)
        
        logger.info(f"BAR - {symbol}: O${open_price} H${high_price} L${low_price} C${close_price} V{volume}")