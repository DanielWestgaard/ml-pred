import json
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Callable
from queue import Queue, Empty
import pandas as pd

from broker.capital_com.capitalcom import CapitalCom
import config.config as config
from data.providers.capital_com import ProviderCapitalCom


class WebSocketDataManager:
    """
    Manages real-time WebSocket data feeds from Capital.com for live trading.
    Provides seamless integration with the existing live data pipeline.
    """
    
    def __init__(self, symbols: List[str], timeframe: str = "MINUTE", 
                 buffer_size: int = 1000, **kwargs):
        """
        Initialize WebSocket data manager
        
        Args:
            symbols: List of trading symbols to subscribe to
            timeframe: Data timeframe (MINUTE, HOUR, etc.)
            buffer_size: Maximum buffer size for each symbol
        """
        self.symbols = symbols
        self.timeframe = timeframe
        self.buffer_size = buffer_size
        
        # Connection state
        self._is_connected = False
        self._is_streaming = False
        self._connection_lock = threading.RLock()
        
        # Broker and provider instances
        self.broker: Optional[CapitalCom] = None
        self.provider: Optional[ProviderCapitalCom] = None
        
        # Data management
        self.data_buffers: Dict[str, Queue] = {}
        self.latest_prices: Dict[str, Dict[str, Any]] = {}
        self.data_callbacks: List[Callable] = []
        
        # WebSocket references
        self.websockets: Dict[str, Any] = {}
        
        # Statistics
        self.messages_received = 0
        self.connection_start_time = None
        self.last_data_time = None
        
        logging.info(f"WebSocket data manager initialized for symbols: {symbols}")
    
    def add_data_callback(self, callback: Callable):
        """Add callback for new data notifications"""
        self.data_callbacks.append(callback)
    
    def start_streaming(self, jumpstart_buffer: bool = True) -> bool:
        """
        Start real-time WebSocket data streaming
        
        Args:
            jumpstart_buffer: Whether to preload historical data first
            
        Returns:
            True if streaming started successfully
        """
        try:
            with self._connection_lock:
                if self._is_streaming:
                    logging.warning("WebSocket streaming already active")
                    return True
                
                # 1. Initialize broker connection
                if not self._initialize_broker():
                    return False
                
                # 2. Jumpstart buffers with historical data if requested
                if jumpstart_buffer:
                    self._jumpstart_buffers()
                
                # 3. Start WebSocket subscriptions
                if not self._start_websocket_subscriptions():
                    return False
                
                self._is_streaming = True
                self.connection_start_time = datetime.now()
                
                logging.info("WebSocket streaming started successfully")
                return True
                
        except Exception as e:
            logging.error(f"Failed to start WebSocket streaming: {e}")
            return False
    
    def stop_streaming(self):
        """Stop WebSocket streaming and cleanup connections"""
        try:
            with self._connection_lock:
                if not self._is_streaming:
                    return
                
                logging.info("Stopping WebSocket streaming...")
                
                # Stop WebSocket connections
                for symbol, ws in self.websockets.items():
                    try:
                        if ws and hasattr(ws, 'close'):
                            ws.close()
                            logging.debug(f"Closed WebSocket for {symbol}")
                    except Exception as e:
                        logging.error(f"Error closing WebSocket for {symbol}: {e}")
                
                # End broker session
                if self.broker:
                    try:
                        self.broker.end_session()
                    except Exception as e:
                        logging.error(f"Error ending broker session: {e}")
                
                self._is_streaming = False
                self._is_connected = False
                self.websockets.clear()
                
                # Log statistics
                if self.connection_start_time:
                    uptime = datetime.now() - self.connection_start_time
                    logging.info(f"WebSocket streaming stopped - Uptime: {uptime}, "
                               f"Messages received: {self.messages_received}")
                
        except Exception as e:
            logging.error(f"Error stopping WebSocket streaming: {e}")
    
    def _initialize_broker(self) -> bool:
        """Initialize Capital.com broker connection"""
        try:
            # Initialize broker
            self.broker = CapitalCom()
            
            # Start session (automatically switches to USD_testing)
            body, headers, x_security_token, cst = self.broker.start_session()
            
            if not x_security_token or not cst:
                logging.error("Failed to establish broker session - missing tokens")
                return False
            
            # Initialize provider for historical data
            self.provider = ProviderCapitalCom()
            
            self._is_connected = True
            logging.info("Capital.com broker connection established")
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize broker: {e}")
            return False
    
    def _jumpstart_buffers(self):
        """Preload buffers with recent historical data"""
        if not self.provider:
            logging.warning("Provider not available for buffer jumpstart")
            return
        
        try:
            # Calculate date range for historical data using UTC timestamps
            # Capital.com API expects UTC, not local time
            now_utc = datetime.now(timezone.utc)
            end_date = now_utc - timedelta(minutes=5)   # 5 minutes ago in UTC
            start_date = end_date - timedelta(hours=6)  # 6 hours of data - TODO: Dynamic with the type of timeframe user has configured, now it only works with a minute data
            
            # Format as UTC timestamps (remove timezone info for string formatting)
            from_date = start_date.replace(tzinfo=None).strftime("%Y-%m-%dT%H:%M:%S")
            to_date = end_date.replace(tzinfo=None).strftime("%Y-%m-%dT%H:%M:%S")
            
            logging.info("Jumpstarting buffers with historical data...")
            
            for symbol in self.symbols:
                try:
                    # Initialize buffer for symbol
                    self.data_buffers[symbol] = Queue(maxsize=self.buffer_size)
                    
                    # Fetch historical data
                    historical_data = self.provider.fetch_maximum_available_data(
                        symbol=symbol,
                        timeframe=self.timeframe,
                        from_date=from_date,
                        to_date=to_date,
                        print_answer=False
                    )
                    
                    if historical_data:
                        # Process and add to buffer
                        processed_count = self._process_historical_data(symbol, historical_data)
                        logging.info(f"Added {processed_count} historical records to {symbol} buffer")
                    
                except Exception as e:
                    logging.error(f"Error jumpstarting buffer for {symbol}: {e}")
                    # Initialize empty buffer as fallback
                    self.data_buffers[symbol] = Queue(maxsize=self.buffer_size)
        
        except Exception as e:
            logging.error(f"Error during buffer jumpstart: {e}")
    
    def _process_historical_data(self, symbol: str, historical_data: Any) -> int:
        """Process historical data and add to buffer"""
        try:
            # Convert historical data to standard format
            if isinstance(historical_data, str):
                # Parse JSON response
                data = json.loads(historical_data)
            else:
                data = historical_data
            
            processed_count = 0
            
            # Extract OHLCV data from API response
            if 'prices' in data:
                for price_bar in data['prices']:
                    try:
                        # Convert to standard OHLCV format
                        ohlcv_data = {
                            'timestamp': price_bar.get('snapshotTimeUTC'),
                            'open': float(price_bar.get('openPrice', {}).get('bid', 0)),
                            'high': float(price_bar.get('highPrice', {}).get('bid', 0)),
                            'low': float(price_bar.get('lowPrice', {}).get('bid', 0)),
                            'close': float(price_bar.get('closePrice', {}).get('bid', 0)),
                            'volume': float(price_bar.get('lastTradedVolume', 0)),
                            'symbol': symbol,
                            'source': 'historical'
                        }
                        
                        # Add to buffer (don't block if full)
                        if not self.data_buffers[symbol].full():
                            self.data_buffers[symbol].put(ohlcv_data, block=False)
                            processed_count += 1
                        
                    except (ValueError, KeyError) as e:
                        logging.debug(f"Skipping invalid historical data point: {e}")
                        continue
            
            return processed_count
            
        except Exception as e:
            logging.error(f"Error processing historical data for {symbol}: {e}")
            return 0
    
    def _start_websocket_subscriptions(self) -> bool:
        """Start WebSocket subscriptions for all symbols"""
        try:
            for symbol in self.symbols:
                if not self._subscribe_to_symbol(symbol):
                    logging.error(f"Failed to subscribe to {symbol}")
                    return False
                
                # Initialize buffer if not already done
                if symbol not in self.data_buffers:
                    self.data_buffers[symbol] = Queue(maxsize=self.buffer_size)
            
            logging.info(f"Successfully subscribed to {len(self.symbols)} symbols")
            return True
            
        except Exception as e:
            logging.error(f"Error starting WebSocket subscriptions: {e}")
            return False
    
    def _subscribe_to_symbol(self, symbol: str) -> bool:
        """Subscribe to WebSocket data for a specific symbol"""
        try:
            # Create custom message handler for this symbol
            message_handler = self._create_message_handler(symbol)
            
            # Subscribe to live market data
            ws = self.broker.sub_live_market_data(
                symbol=symbol,
                timeframe=self.timeframe,
                message_handler=message_handler
            )
            
            if ws:
                self.websockets[symbol] = ws
                logging.info(f"Subscribed to WebSocket data for {symbol}")
                return True
            else:
                logging.error(f"Failed to create WebSocket subscription for {symbol}")
                return False
                
        except Exception as e:
            logging.error(f"Error subscribing to {symbol}: {e}")
            return False
    
    def _create_message_handler(self, symbol: str):
        """Create message handler for WebSocket data"""
        def handle_message(ws, message):
            try:
                # Parse message
                data = json.loads(message)
                
                # Check if it's OHLC data
                if data.get("destination") == "ohlc.event":
                    payload = data.get("payload", {})
                    
                    # Extract OHLCV data
                    ohlcv_data = {
                        'timestamp': payload.get('t'),  # Timestamp in milliseconds
                        'open': float(payload.get('o', 0)),
                        'high': float(payload.get('h', 0)),
                        'low': float(payload.get('l', 0)),
                        'close': float(payload.get('c', 0)),
                        'volume': float(payload.get('v', 0)) if payload.get('v') else 1.0,
                        'symbol': symbol,
                        'epic': payload.get('epic'),
                        'resolution': payload.get('resolution'),
                        'source': 'websocket'
                    }
                    
                    # Add to buffer and update latest price
                    self._process_new_data(symbol, ohlcv_data)
                    
                    self.messages_received += 1
                    self.last_data_time = datetime.now()
                    
                    # Log periodic updates
                    if self.messages_received % 100 == 0:
                        logging.debug(f"Received {self.messages_received} WebSocket messages")
                
                elif data.get("destination") == "ping":
                    # Handle ping responses
                    logging.debug("Received ping response")
                
                else:
                    # Log other message types for debugging
                    logging.debug(f"Received WebSocket message: {data.get('destination', 'unknown')}")
                    
            except Exception as e:
                logging.error(f"Error processing WebSocket message for {symbol}: {e}")
                logging.debug(f"Problematic message: {message}")
        
        return handle_message
    
    def _process_new_data(self, symbol: str, ohlcv_data: Dict[str, Any]):
        """Process new OHLCV data from WebSocket"""
        try:
            # Convert timestamp to datetime if needed
            if isinstance(ohlcv_data['timestamp'], (int, float)):
                ohlcv_data['datetime'] = datetime.fromtimestamp(ohlcv_data['timestamp'] / 1000)
            
            # Update latest price
            self.latest_prices[symbol] = ohlcv_data.copy()
            
            # Add to buffer (remove oldest if full)
            if symbol in self.data_buffers:
                if self.data_buffers[symbol].full():
                    try:
                        # Remove oldest item
                        self.data_buffers[symbol].get(block=False)
                    except Empty:
                        pass
                
                # Add new data
                self.data_buffers[symbol].put(ohlcv_data, block=False)
            
            # Notify callbacks
            self._notify_data_callbacks(symbol, ohlcv_data)
            
        except Exception as e:
            logging.error(f"Error processing new data for {symbol}: {e}")
    
    def _notify_data_callbacks(self, symbol: str, data: Dict[str, Any]):
        """Notify registered callbacks about new data"""
        for callback in self.data_callbacks:
            try:
                callback(symbol, data)
            except Exception as e:
                logging.error(f"Error in data callback: {e}")
    
    def get_latest_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest price data for symbol"""
        return self.latest_prices.get(symbol)
    
    def get_buffer_data(self, symbol: str, count: int = None) -> List[Dict[str, Any]]:
        """Get recent data from buffer"""
        if symbol not in self.data_buffers:
            return []
        
        try:
            # Convert queue to list (non-destructive)
            buffer_items = list(self.data_buffers[symbol].queue)
            
            if count:
                return buffer_items[-count:]
            return buffer_items
            
        except Exception as e:
            logging.error(f"Error getting buffer data for {symbol}: {e}")
            return []
    
    def is_connected(self) -> bool:
        """Check if WebSocket is connected and streaming"""
        return self._is_connected and self._is_streaming
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get detailed connection status"""
        status = {
            'connected': self._is_connected,
            'streaming': self._is_streaming,
            'symbols_subscribed': list(self.websockets.keys()),
            'messages_received': self.messages_received,
            'last_data_time': self.last_data_time,
            'uptime': None,
            'active_connections': len([ws for ws in self.websockets.values() if ws])
        }
        
        if self.connection_start_time:
            status['uptime'] = datetime.now() - self.connection_start_time
        
        return status
    
    def get_data_as_dataframe(self, symbol: str, count: int = None) -> Optional[pd.DataFrame]:
        """Convert buffer data to pandas DataFrame"""
        try:
            buffer_data = self.get_buffer_data(symbol, count)
            
            if not buffer_data:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(buffer_data)
            
            # Ensure datetime column
            if 'datetime' not in df.columns and 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Sort by timestamp
            if 'datetime' in df.columns:
                df = df.sort_values('datetime').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logging.error(f"Error converting buffer data to DataFrame for {symbol}: {e}")
            return None