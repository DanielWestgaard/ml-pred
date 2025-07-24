import threading
import time
import pandas as pd
import numpy as np
from collections import deque
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta, timezone
import queue

from broker.capital_com.capitalcom import CapitalCom
from data.live.websocket_data_manager import WebSocketDataManager
from data.processing.cleaning import DataCleaner
from data.processing.validation import DataValidator
from data.features.transformation import FeatureTransformer
from utils.data_utils import parse_timestamp_series
from utils.broker_utils import extract_mid_price_ohlcv
import config.config as config
from data.features.enhanced_generation import ModelAwareFeatureGenerator
from utils.logging_utils import get_logger

logger = get_logger()


class LiveDataManager:
    """
    Manages live market data streaming, buffering, and feature generation.
    Provides a clean interface for strategies to get processed live data.
    """
    
    def __init__(self, symbol: str = "GBPUSD", timeframe: str = "MINUTE", 
                 buffer_size: int = 500, feature_window: int = 200,
                 selected_features: List[str] = None, **kwargs):
        """
        Initialize live data manager
        
        Args:
            symbol: Trading symbol (e.g., "GBPUSD")
            timeframe: Data timeframe ("MINUTE", "MINUTE_5", etc.)
            buffer_size: Maximum number of data points to keep in buffer
            feature_window: Minimum data points needed for feature calculation
            selected_features: Specific features to generate (None = all features)
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.buffer_size = buffer_size
        self.feature_window = feature_window
        self.selected_features = selected_features
        
        # Data storage
        self.raw_buffer = deque(maxlen=buffer_size)
        self.processed_data = None
        self.latest_features = None
        
        # Deduplication tracking
        self._last_processed_buffer_timestamp = None
        
        # Rate limiting for repetitive log messages (avoid log spam)
        self._last_ready_check_log = None
        
        # Threading and synchronization
        self._data_lock = threading.RLock()
        self._is_streaming = False
        self._stream_thread = None
        self._processing_thread = None
        self._data_queue = queue.Queue()
        
        # Broker connection
        self.broker = None
        self._last_heartbeat = None
        self._connection_retries = 0
        self._max_retries = 5
        
        # WebSocket data manager for real-time feeds
        self.websocket_manager = None
        self._use_websocket = kwargs.get('use_websocket', True)
        
        # Callbacks for data updates
        self._data_callbacks = []
        
        logger.info(f"LiveDataManager initialized - Symbol: {symbol}, "
                    f"Timeframe: {timeframe}, Buffer: {buffer_size}")
    
    def add_data_callback(self, callback: Callable[[pd.DataFrame], None]):
        """Add callback function to be called when new processed data is available"""
        self._data_callbacks.append(callback)
    
    def start_streaming(self, dry_run: bool = False, jumpstart: bool = True) -> bool:
        """
        Start live data streaming with optional buffer jumpstart
        
        Args:
            dry_run: If True, simulate data instead of connecting to broker
            jumpstart: If True, preload buffer with historical data for faster startup
            
        Returns:
            True if streaming started successfully
        """
        if self._is_streaming:
            logger.warning("Data streaming already active")
            return True
            
        try:
            # For live mode, establish broker connection first (needed for jumpstart historical data)
            if not dry_run:
                if self._use_websocket:
                    # Initialize WebSocket data streaming
                    websocket_initialized = self._initialize_websocket_streaming()
                    if not websocket_initialized:
                        logger.warning("WebSocket initialization failed, falling back to traditional broker")
                        if not self._connect_broker():
                            return False
                    else:
                        # WebSocket initialized successfully,
                        # ONLY connect broker if WebSocketDataManager didn't establish it already
                        # or if historical data jumpstart specifically requires it
                        if not self.websocket_manager.is_connected(): # Assuming is_connected() exists or similar check
                            if not self._connect_broker():
                                logger.warning("Failed to establish broker connection for historical data, jumpstart will be disabled")
                        else:
                            logger.info("Broker connection already established by WebSocketDataManager, skipping redundant connection.")
                else:
                    # Use traditional broker connection
                    if not self._connect_broker():
                        return False
            
            # Jumpstart buffer with historical data if enabled (now after broker connection)
            # Skip jumpstart if using WebSocket mode since WebSocketDataManager already did it
            if jumpstart and not self._use_websocket:
                if not self._jumpstart_buffer(dry_run):
                    logger.warning("Buffer jumpstart failed, continuing with normal startup")
            elif jumpstart and self._use_websocket:
                logger.info("Transferring historical data from WebSocketDataManager to LiveDataManager buffer")
                self._transfer_websocket_buffer_data()
            
            # Start data processing thread
            self._is_streaming = True
            self._processing_thread = threading.Thread(
                target=self._process_data_worker, daemon=True)
            self._processing_thread.start()
            
            if dry_run:
                # Start simulation thread for testing
                logger.info("SIMULATION MODE: Starting simulated data generation")
                self._stream_thread = threading.Thread(
                    target=self._simulate_data_worker, daemon=True)
                self._stream_thread.start()
            elif self.websocket_manager and self.websocket_manager.is_connected():
                # WebSocket handles data streaming automatically via callbacks
                logger.info("LIVE MODE: Using WebSocket for real-time BTCUSD data from Capital.com")
            else:
                # Start traditional data streaming thread
                logger.info("LIVE MODE: Using traditional broker polling for real-time data")
                self._stream_thread = threading.Thread(
                    target=self._stream_data_worker, daemon=True)
                self._stream_thread.start()
            
            logger.info(f"Live data streaming started - Mode: {'simulation' if dry_run else 'live'}, "
                        f"Jumpstart: {'enabled' if jumpstart else 'disabled'}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start data streaming: {e}")
            self._is_streaming = False
            return False
    
    def stop_streaming(self):
        """Stop live data streaming and cleanup resources"""
        if not self._is_streaming:
            return
            
        logger.info("Stopping live data streaming...")
        self._is_streaming = False
        
        # Wait for threads to finish
        if self._stream_thread and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=5)
        
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=5)
        
        # Cleanup WebSocket connection
        if self.websocket_manager:
            try:
                self.websocket_manager.stop_streaming()
            except Exception as e:
                logger.error(f"Error stopping WebSocket streaming: {e}")
            finally:
                self.websocket_manager = None
        
        # Cleanup broker connection
        if self.broker:
            try:
                self.broker.end_session()
            except Exception as e:
                logger.error(f"Error closing broker session: {e}")
            finally:
                self.broker = None
        
        logger.info("Live data streaming stopped")
    
    def _connect_broker(self) -> bool:
        """Establish broker connection with retry logic"""
        for attempt in range(self._max_retries):
            try:
                # Clean up any existing broker connection first to avoid session conflicts
                if self.broker:
                    try:
                        self.broker.end_session()
                    except:
                        pass
                    self.broker = None
                
                self.broker = CapitalCom()
                self.broker.start_session()  # Now automatically switches to USD_testing
                self._connection_retries = 0
                self._last_heartbeat = datetime.now()
                logger.info("Broker connection established")
                return True
                
            except RuntimeError as e:
                # Handle specific "Request-started" errors
                if "Request-started" in str(e):
                    self._connection_retries += 1
                    logger.warning(f"Broker connection attempt {attempt + 1} failed with 'Request-started' error: {e}")
                    if attempt < self._max_retries - 1:
                        # Longer delay for "Request-started" errors
                        delay = 3 + (2 ** attempt)
                        logger.info(f"Waiting {delay} seconds before retry due to session conflict...")
                        time.sleep(delay)
                    continue
                else:
                    raise
                
            except Exception as e:
                self._connection_retries += 1
                logger.error(f"Broker connection attempt {attempt + 1} failed: {e}")
                if attempt < self._max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        logger.error("Failed to establish broker connection after retries")
        return False
    
    def _initialize_websocket_streaming(self) -> bool:
        """Initialize WebSocket data streaming"""
        try:
            # Create WebSocket manager
            self.websocket_manager = WebSocketDataManager(
                symbols=[self.symbol],
                timeframe=self.timeframe,
                buffer_size=self.buffer_size
            )
            
            # Add callback to receive WebSocket data
            self.websocket_manager.add_data_callback(self._on_websocket_data)
            
            # Start WebSocket streaming with jumpstart
            if self.websocket_manager.start_streaming(jumpstart_buffer=True):
                logger.info("WebSocket streaming initialized successfully")
                return True
            else:
                logger.error("Failed to start WebSocket streaming")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing WebSocket streaming: {e}")
            return False
    
    def _on_websocket_data(self, symbol: str, data: Dict[str, Any]):
        """Callback for WebSocket data updates"""
        try:
            if symbol == self.symbol:
                # Convert WebSocket data to standard format
                market_data = {
                    'date': data.get('datetime', datetime.now()),
                    'open': data.get('open'),
                    'high': data.get('high'),
                    'low': data.get('low'),
                    'close': data.get('close'),
                    'volume': data.get('volume', 1.0),
                    'symbol': symbol,
                    'source': 'websocket'
                }
                
                # Add to data queue for processing
                self._data_queue.put(market_data, block=False)
                
        except Exception as e:
            logger.error(f"Error processing WebSocket data callback: {e}")
    
    def _stream_data_worker(self):
        """Worker thread for streaming live data from broker"""
        while self._is_streaming:
            try:
                # Check connection health
                if not self._check_connection_health():
                    if not self._reconnect_broker():
                        time.sleep(5)
                        continue
                
                # Subscribe to live market data
                # Note: This is a simplified version - real implementation would
                # need to handle WebSocket properly
                data_point = self._fetch_latest_market_data()
                
                if data_point:
                    self._data_queue.put(data_point)
                    self._last_heartbeat = datetime.now()
                
                time.sleep(1)  # Adjust based on data frequency needs
                
            except Exception as e:
                logger.error(f"Error in data streaming: {e}")
                time.sleep(5)
    
    def _simulate_data_worker(self):
        """Worker thread for simulating live data (for testing)"""
        # Start with historical timestamp to avoid "future data" validation issues
        base_price = 1.2500
        timestamp = datetime.now() - timedelta(hours=1)  # Start 1 hour ago
        
        while self._is_streaming:
            try:
                # Generate realistic price movement
                price_change = np.random.normal(0, 0.0001)
                current_close = base_price + price_change
                
                # Generate proper OHLC relationships
                open_price = base_price
                
                # Generate high and low with proper relationships
                high_offset = abs(np.random.normal(0, 0.00003))
                low_offset = abs(np.random.normal(0, 0.00003))
                
                high_price = max(open_price, current_close) + high_offset
                low_price = min(open_price, current_close) - low_offset
                
                volume = np.random.randint(50, 500)
                
                data_point = {
                    'date': timestamp,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': current_close,
                    'volume': volume
                }
                
                self._data_queue.put(data_point)
                base_price = current_close
                timestamp += timedelta(minutes=1)
                
                time.sleep(60)  # Simulate 1-minute intervals (60 seconds)
                
            except Exception as e:
                logger.error(f"Error in data simulation: {e}")
                time.sleep(1)
    
    def _process_data_worker(self):
        """Worker thread for processing incoming data and generating features"""
        while self._is_streaming:
            try:
                # Get new data point
                data_point = self._data_queue.get(timeout=1)
                
                with self._data_lock:
                    # Add to buffer
                    self.raw_buffer.append(data_point)
                    
                    # Process if we have enough data
                    if len(self.raw_buffer) >= self.feature_window:
                        self._process_current_buffer()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing data: {e}")
                time.sleep(1)
    
    def _process_current_buffer(self):
        """Process current buffer data through cleaning and feature generation"""
        try:
            # Convert buffer to DataFrame
            df = pd.DataFrame(list(self.raw_buffer))
            
            # Add timestamp-based deduplication to prevent processing same data multiple times
            if not df.empty and 'date' in df.columns:
                latest_timestamp = df['date'].iloc[-1]
                
                # Check if we've already processed this exact timestamp
                if hasattr(self, '_last_processed_buffer_timestamp') and latest_timestamp == self._last_processed_buffer_timestamp:
                    logger.debug(f"Buffer already processed for timestamp {latest_timestamp}, skipping duplicate processing")
                    return
                
                # Mark this timestamp as processed
                self._last_processed_buffer_timestamp = latest_timestamp
                logger.debug(f"Processing buffer with latest timestamp: {latest_timestamp}")
            
            # Ensure date column exists and is properly formatted
            if 'date' not in df.columns:
                logger.error("No 'date' column found in the dataframe")
                return
                
            try:
                df['date'] = parse_timestamp_series(df['date'])
            except Exception as e:
                logger.error(f"Error parsing timestamps in dataframe: {e}")
                # Fallback to original method with error handling
                try:
                    df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
                except Exception as fallback_e:
                    logger.error(f"Fallback timestamp parsing also failed: {fallback_e}")
                    return
            df.set_index('date', inplace=True)
            
            # Filter out metadata columns that shouldn't be features
            metadata_columns = ['symbol', 'source']
            columns_to_drop = [col for col in metadata_columns if col in df.columns]
            if columns_to_drop:
                df = df.drop(columns=columns_to_drop)
                logger.debug(f"Filtered out metadata columns: {columns_to_drop}")
            
            # Clean data using the existing pipeline pattern
            cleaner = DataCleaner(df.copy(), save_cleaned_data=False)
            df_cleaned = cleaner.run()
            
            # Validate data (more lenient for live streaming)
            validator = DataValidator(df_cleaned.copy())
            validation_results = validator.run()
            if not validation_results.get('is_valid', False):
                logger.warning(f"Data validation failed: {validation_results.get('issues', [])}")
                logger.warning("Proceeding with feature generation despite validation issues (live streaming mode)")
                # Continue anyway in live mode - validation issues are often due to simulated data timestamps
            
            # Generate features
            
            # Use the same model metadata path for both feature generation and transformation
            model_metadata_path = "storage/models/2025-07-03_12:45:35/model_metadata.yml"  # TODO: Dynamic to use the models from hyppthesis/strategy config
            
            feature_generator = ModelAwareFeatureGenerator(df_cleaned.copy())
            df_with_features = feature_generator.generate_for_model(
                model_metadata_path=model_metadata_path
            )
            # logger.debug(f"After feature generation: {len(df_with_features.columns)} columns")  # debugging
            
            feature_transformer = FeatureTransformer(
                df_with_features, 
                model_metadata_path=model_metadata_path
            )
            df_with_features = feature_transformer.run()
            # logger.debug(f"After transformation: {len(df_with_features.columns)} columns")  # debugging
            
            # Store processed data
            self.processed_data = df_with_features
            self.latest_features = df_with_features.iloc[-1:].copy()
            
            # Notify callbacks
            self._notify_callbacks(self.latest_features)
            
        except Exception as e:
            logger.error(f"Error in data processing pipeline: {e}")
    
    def _fetch_latest_market_data(self) -> Optional[Dict]:
        """Fetch latest market data from broker"""
        try:
            # This is a placeholder - real implementation would use WebSocket
            # or periodic API calls to get latest price data
            if self.broker:
                # Use broker's historical data endpoint to get latest candle
                # In production, this should be WebSocket data
                pass
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return None
    
    def _check_connection_health(self) -> bool:
        """Check if broker connection is healthy"""
        # In simulation mode (no WebSocket), always return True if streaming
        if self._use_websocket == False and self._is_streaming:
            return True
        
        # For WebSocket mode, check WebSocket connection health
        if self._use_websocket and self.websocket_manager:
            return self.websocket_manager.is_connected()
        
        # For traditional broker connections
        if not self.broker or not self._last_heartbeat:
            return False
        
        # Check if we've received data recently
        time_since_heartbeat = datetime.now() - self._last_heartbeat
        return time_since_heartbeat.total_seconds() < 60
    
    def _reconnect_broker(self) -> bool:
        """Attempt to reconnect to broker"""
        logger.info("Attempting broker reconnection...")
        
        # Cleanup existing connection
        if self.broker:
            try:
                self.broker.end_session()
            except:
                pass
            self.broker = None
        
        # Attempt reconnection
        return self._connect_broker()
    
    def _jumpstart_buffer(self, dry_run: bool = False) -> bool:
        """
        Jumpstart the buffer with historical data for faster startup
        
        Args:
            dry_run: If True, generate simulated historical data
            
        Returns:
            True if jumpstart successful
        """
        try:
            logger.info("Jumpstarting buffer with historical data...")
            
            if dry_run:
                # Generate bulk simulated historical data instantly
                historical_data = self._generate_bulk_simulation_data()
            else:
                # Fetch recent historical data from broker
                historical_data = self._fetch_historical_data_from_broker()
            
            if historical_data is None or len(historical_data) == 0:
                logger.error("Failed to get historical data for jumpstart")
                return False
            
            # Add historical data to buffer
            with self._data_lock:
                for _, row in historical_data.iterrows():
                    data_point = {
                        'date': row.name,  # Index should be datetime
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close'],
                        'volume': row['volume']
                    }
                    self.raw_buffer.append(data_point)
                
                # Process the jumpstart data immediately
                if len(self.raw_buffer) >= self.feature_window:
                    self._process_current_buffer()
            
            logger.info(f"Buffer jumpstart completed - Added {len(historical_data)} historical data points")
            return True
            
        except Exception as e:
            logger.error(f"Error during buffer jumpstart: {e}")
            return False
    
    def _generate_bulk_simulation_data(self) -> Optional[pd.DataFrame]:
        """Generate bulk historical simulation data for jumpstart"""
        try:
            # Generate enough data for feature calculation + some extra
            # Need at least 250 points for volatility_long (100 period) + buffer to avoid NaN threshold
            n_points = max(self.feature_window + 50, 250)
            
            # Start from 6 hours ago to ensure historical timestamps (avoid validation issues)
            start_time = datetime.now() - timedelta(hours=6)
            
            base_price = 1.2500
            data = []
            
            for i in range(n_points):
                # Generate realistic price movement
                price_change = np.random.normal(0, 0.0001)
                current_close = base_price + price_change
                
                # Generate proper OHLC relationships
                open_price = base_price
                
                # Generate high and low with proper relationships
                high_offset = abs(np.random.normal(0, 0.00003))
                low_offset = abs(np.random.normal(0, 0.00003))
                
                high_price = max(open_price, current_close) + high_offset
                low_price = min(open_price, current_close) - low_offset
                
                volume = np.random.randint(50, 500)
                
                data.append({
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': current_close,
                    'volume': volume
                })
                
                base_price = current_close
                start_time += timedelta(minutes=1)
            
            # Create DataFrame with datetime index
            df = pd.DataFrame(data)
            df.index = pd.date_range(
                start=datetime.now() - timedelta(hours=6),
                periods=n_points,
                freq='1min'
            )
            df.index.name = 'date'
            
            logger.info(f"Generated {len(df)} simulated historical data points for jumpstart")
            return df
            
        except Exception as e:
            logger.error(f"Error generating bulk simulation data: {e}")
            return None
    
    def _fetch_historical_data_from_broker(self) -> Optional[pd.DataFrame]:
        """Fetch recent historical data from Capital.com for jumpstart"""
        try:
            if not self.broker:
                logger.error("No broker connection available for historical data fetch")
                return None
            
            # Calculate how many data points we need
            n_points = max(self.feature_window + 50, 200)
            
            # Calculate time range using UTC timestamps (Capital.com API expects UTC)
            now_utc = datetime.now(timezone.utc)
            end_time = now_utc - timedelta(minutes=2)  # 2 minutes ago in UTC
            start_time = end_time - timedelta(minutes=n_points + 10)  # Add buffer - TODO: Dynamic with the type of timeframe user has configured, now it only works with a minute data
            
            # Format times for Capital.com API (UTC format)
            from_date = start_time.replace(tzinfo=None).strftime("%Y-%m-%dT%H:%M:%S.000Z")
            to_date = end_time.replace(tzinfo=None).strftime("%Y-%m-%dT%H:%M:%S.000Z")
            
            logger.info(f"Fetching historical data from {from_date} to {to_date}")
            
            # Use Capital.com's historical data endpoint
            historical_data = self.broker.get_historical_data(
                epic=self.symbol,
                resolution=self.timeframe,
                max=n_points,
                from_date=from_date,
                to_date=to_date,
                print_answer=False
            )
            
            if historical_data is None:
                logger.error("No historical data returned from broker")
                return None
            
            # Convert Capital.com API response to DataFrame format
            if isinstance(historical_data, dict):
                # Log the structure for debugging
                logger.debug(f"Historical data keys: {list(historical_data.keys())}")
                
                if 'prices' in historical_data:
                    prices_data = historical_data['prices']
                    if not prices_data:
                        logger.error("No price data in historical response")
                        return None
                    
                    # Convert to DataFrame
                    df_data = []
                    for price_point in prices_data:
                        # Use unified mid-price extraction - no redundant conversion
                        ohlcv_data = extract_mid_price_ohlcv(price_point)
                        df_data.append({
                            'date': ohlcv_data['timestamp'],
                            'open': ohlcv_data['open'],
                            'high': ohlcv_data['high'], 
                            'low': ohlcv_data['low'],
                            'close': ohlcv_data['close'],
                            'volume': ohlcv_data['volume']
                        })
                    
                    df = pd.DataFrame(df_data)
                    
                    # Parse timestamps robustly
                    try:
                        df['date'] = parse_timestamp_series(df['date'])
                    except Exception as e:
                        logger.error(f"Error parsing timestamps in historical data: {e}")
                        # Fallback to pandas with error handling
                        try:
                            df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
                        except Exception as fallback_e:
                            logger.error(f"Fallback timestamp parsing failed: {fallback_e}")
                            return None
                    df.set_index('date', inplace=True)
                    
                    logger.info(f"Fetched {len(df)} historical data points from broker")
                    return df
                else:
                    # Check if it's an error response
                    if 'errorCode' in historical_data:
                        logger.error(f"API Error in historical data: {historical_data['errorCode']}")
                        return None
                    else:
                        logger.error(f"Unexpected historical data format - missing 'prices' key. Keys: {list(historical_data.keys())}")
                        return None
            else:
                logger.error(f"Unexpected historical data format: {type(historical_data)}")
                return None
            
        except Exception as e:
            logger.error(f"Error fetching historical data from broker: {e}")
            # In live mode, do NOT fallback to simulation data
            # Return None to indicate failure rather than generating fake data
            return None
    
    def _notify_callbacks(self, data: pd.DataFrame):
        """Notify all registered callbacks of new data"""
        for callback in self._data_callbacks:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Error in data callback: {e}")
    
    def _transfer_websocket_buffer_data(self):
        """Transfer historical data from WebSocketDataManager to LiveDataManager buffer"""
        try:
            if not self.websocket_manager:
                logger.warning("No WebSocket manager available for data transfer")
                return
            
            # Get historical data from WebSocket buffer
            historical_data = self.websocket_manager.get_buffer_data(
                symbol=self.symbol, 
                count=self.buffer_size
            )
            
            if not historical_data:
                logger.warning("No historical data available in WebSocket buffer")
                return
            
            # Add to LiveDataManager buffer
            with self._data_lock:
                for data_point in historical_data:
                    # Convert from WebSocket format to LiveDataManager format
                    formatted_data = {
                        'date': data_point.get('timestamp', data_point.get('date')),
                        'open': data_point.get('open'),
                        'high': data_point.get('high'),
                        'low': data_point.get('low'),
                        'close': data_point.get('close'),
                        'volume': data_point.get('volume', 1.0)
                    }
                    self.raw_buffer.append(formatted_data)
            
            logger.info(f"Transferred {len(historical_data)} historical data points to LiveDataManager buffer")
            
        except Exception as e:
            logger.error(f"Error transferring WebSocket buffer data: {e}")
    
    def get_latest_features(self) -> Optional[pd.DataFrame]:
        """
        Get the latest processed features
        
        Returns:
            DataFrame with latest features or None if not available
        """
        with self._data_lock:
            return self.latest_features.copy() if self.latest_features is not None else None
    
    def get_processed_data(self, n_periods: int = None) -> Optional[pd.DataFrame]:
        """
        Get processed data with features
        
        Args:
            n_periods: Number of recent periods to return (None = all)
            
        Returns:
            DataFrame with processed data or None if not available
        """
        with self._data_lock:
            if self.processed_data is None:
                return None
            
            if n_periods:
                return self.processed_data.tail(n_periods).copy()
            return self.processed_data.copy()
    
    def get_buffer_status(self) -> Dict[str, Any]:
        """Get current buffer status and statistics"""
        with self._data_lock:
            return {
                'buffer_size': len(self.raw_buffer),
                'max_buffer_size': self.buffer_size,
                'buffer_utilization': len(self.raw_buffer) / self.buffer_size,
                'has_processed_data': self.processed_data is not None,
                'latest_timestamp': list(self.raw_buffer)[-1]['date'] if self.raw_buffer else None,
                'is_streaming': self._is_streaming,
                'connection_healthy': self._check_connection_health(),
                'websocket_connected': self.websocket_manager.is_connected() if self.websocket_manager else False
            }
    
    def is_ready_for_trading(self) -> bool:
        """
        Check if data manager is ready to provide data for trading
        
        Returns:
            True if ready (streaming, enough data, features available)
        """
        with self._data_lock:
            streaming_ok = self._is_streaming
            buffer_ok = len(self.raw_buffer) >= self.feature_window
            features_ok = self.latest_features is not None
            connection_ok = self._check_connection_health()
            
            # Debug logger to see what's failing (rate limited to avoid spam)
            if not (streaming_ok and buffer_ok and features_ok and connection_ok):
                current_time = datetime.now()
                should_log = True
                
                if self._last_ready_check_log:
                    time_since_last = (current_time - self._last_ready_check_log).total_seconds()
                    should_log = time_since_last >= 30  # 30 seconds
                
                if should_log:
                    logger.debug(f"Ready check failed - Streaming: {streaming_ok}, "
                                f"Buffer: {buffer_ok} ({len(self.raw_buffer)}>={self.feature_window}), "
                                f"Features: {features_ok}, Connection: {connection_ok}")
                    self._last_ready_check_log = current_time
            
            return streaming_ok and buffer_ok and features_ok and connection_ok