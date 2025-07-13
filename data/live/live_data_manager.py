import logging
import threading
import time
import pandas as pd
import numpy as np
from collections import deque
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
import queue

from broker.capital_com.capitalcom import CapitalCom
from data.features.generation import FeatureGenerator
from data.processing.cleaning import DataCleaner
from data.processing.validation import DataValidator
import config.config as config


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
        
        # Callbacks for data updates
        self._data_callbacks = []
        
        logging.info(f"LiveDataManager initialized - Symbol: {symbol}, "
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
            logging.warning("Data streaming already active")
            return True
            
        try:
            # Jumpstart buffer with historical data if enabled
            if jumpstart:
                if not self._jumpstart_buffer(dry_run):
                    logging.warning("Buffer jumpstart failed, continuing with normal startup")
            
            if not dry_run:
                # Initialize broker connection
                if not self._connect_broker():
                    return False
            
            # Start data processing thread
            self._is_streaming = True
            self._processing_thread = threading.Thread(
                target=self._process_data_worker, daemon=True)
            self._processing_thread.start()
            
            if dry_run:
                # Start simulation thread for testing
                self._stream_thread = threading.Thread(
                    target=self._simulate_data_worker, daemon=True)
            else:
                # Start real data streaming thread
                self._stream_thread = threading.Thread(
                    target=self._stream_data_worker, daemon=True)
            
            self._stream_thread.start()
            
            logging.info(f"Live data streaming started - Mode: {'simulation' if dry_run else 'live'}, "
                        f"Jumpstart: {'enabled' if jumpstart else 'disabled'}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to start data streaming: {e}")
            self._is_streaming = False
            return False
    
    def stop_streaming(self):
        """Stop live data streaming and cleanup resources"""
        if not self._is_streaming:
            return
            
        logging.info("Stopping live data streaming...")
        self._is_streaming = False
        
        # Wait for threads to finish
        if self._stream_thread and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=5)
        
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=5)
        
        # Cleanup broker connection
        if self.broker:
            try:
                self.broker.end_session()
            except Exception as e:
                logging.error(f"Error closing broker session: {e}")
            finally:
                self.broker = None
        
        logging.info("Live data streaming stopped")
    
    def _connect_broker(self) -> bool:
        """Establish broker connection with retry logic"""
        for attempt in range(self._max_retries):
            try:
                self.broker = CapitalCom()
                self.broker.start_session()
                self.broker.switch_active_account(account_name = config.VAR_ACCOUNT_NAME_TEST, print_answer=False)  # TODO: Should be more dynamic. Maybe provide this in the config?
                self._connection_retries = 0
                self._last_heartbeat = datetime.now()
                logging.info("Broker connection established")
                return True
                
            except Exception as e:
                self._connection_retries += 1
                logging.error(f"Broker connection attempt {attempt + 1} failed: {e}")
                if attempt < self._max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        logging.error("Failed to establish broker connection after retries")
        return False
    
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
                logging.error(f"Error in data streaming: {e}")
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
                
                time.sleep(1)  # Simulate 1-minute intervals
                
            except Exception as e:
                logging.error(f"Error in data simulation: {e}")
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
                logging.error(f"Error processing data: {e}")
                time.sleep(1)
    
    def _process_current_buffer(self):
        """Process current buffer data through cleaning and feature generation"""
        try:
            # Convert buffer to DataFrame
            df = pd.DataFrame(list(self.raw_buffer))
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Clean data using the existing pipeline pattern
            cleaner = DataCleaner(df.copy(), save_cleaned_data=False)
            df_cleaned = cleaner.run()
            
            # Validate data
            validator = DataValidator(df_cleaned.copy())
            validation_results = validator.run()
            if not validation_results.get('is_valid', False):
                logging.warning("Data validation failed, skipping feature generation")
                return
            
            # Generate features
            feature_generator = FeatureGenerator(
                df_cleaned.copy(), 
                selected_features=self.selected_features
            )
            df_with_features = feature_generator.run()
            
            # Store processed data
            self.processed_data = df_with_features
            self.latest_features = df_with_features.iloc[-1:].copy()
            
            # Notify callbacks
            self._notify_callbacks(self.latest_features)
            
        except Exception as e:
            logging.error(f"Error in data processing pipeline: {e}")
    
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
            logging.error(f"Error fetching market data: {e}")
            return None
    
    def _check_connection_health(self) -> bool:
        """Check if broker connection is healthy"""
        if not self.broker or not self._last_heartbeat:
            return False
        
        # Check if we've received data recently
        time_since_heartbeat = datetime.now() - self._last_heartbeat
        return time_since_heartbeat.total_seconds() < 60
    
    def _reconnect_broker(self) -> bool:
        """Attempt to reconnect to broker"""
        logging.info("Attempting broker reconnection...")
        
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
            logging.info("Jumpstarting buffer with historical data...")
            
            if dry_run:
                # Generate bulk simulated historical data instantly
                historical_data = self._generate_bulk_simulation_data()
            else:
                # Fetch recent historical data from broker
                historical_data = self._fetch_historical_data_from_broker()
            
            if historical_data is None or len(historical_data) == 0:
                logging.error("Failed to get historical data for jumpstart")
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
            
            logging.info(f"Buffer jumpstart completed - Added {len(historical_data)} historical data points")
            return True
            
        except Exception as e:
            logging.error(f"Error during buffer jumpstart: {e}")
            return False
    
    def _generate_bulk_simulation_data(self) -> Optional[pd.DataFrame]:
        """Generate bulk historical simulation data for jumpstart"""
        try:
            # Generate enough data for feature calculation + some extra
            n_points = max(self.feature_window + 50, 100)
            
            # Start from 2 hours ago to ensure historical timestamps
            start_time = datetime.now() - timedelta(hours=2)
            
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
                start=datetime.now() - timedelta(hours=2),
                periods=n_points,
                freq='1min'
            )
            df.index.name = 'date'
            
            logging.info(f"Generated {len(df)} simulated historical data points for jumpstart")
            return df
            
        except Exception as e:
            logging.error(f"Error generating bulk simulation data: {e}")
            return None
    
    def _fetch_historical_data_from_broker(self) -> Optional[pd.DataFrame]:
        """Fetch recent historical data from Capital.com for jumpstart"""
        try:
            if not self.broker:
                logging.error("No broker connection available for historical data fetch")
                return None
            
            # Calculate how many data points we need
            n_points = max(self.feature_window + 50, 200)
            
            # Calculate time range (assuming 1-minute data)
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=n_points + 10)  # Add buffer
            
            # Format times for Capital.com API
            from_date = start_time.strftime("%Y-%m-%dT%H:%M:%S")
            to_date = end_time.strftime("%Y-%m-%dT%H:%M:%S")
            
            logging.info(f"Fetching historical data from {from_date} to {to_date}")
            
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
                logging.error("No historical data returned from broker")
                return None
            
            # Convert to DataFrame format expected by our pipeline
            # Note: This depends on Capital.com's response format
            # You may need to adjust based on actual API response structure
            
            logging.info(f"Fetched {len(historical_data)} historical data points from broker")
            return historical_data
            
        except Exception as e:
            logging.error(f"Error fetching historical data from broker: {e}")
            return None
    
    def _notify_callbacks(self, data: pd.DataFrame):
        """Notify all registered callbacks of new data"""
        for callback in self._data_callbacks:
            try:
                callback(data)
            except Exception as e:
                logging.error(f"Error in data callback: {e}")
    
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
                'connection_healthy': self._check_connection_health()
            }
    
    def is_ready_for_trading(self) -> bool:
        """
        Check if data manager is ready to provide data for trading
        
        Returns:
            True if ready (streaming, enough data, features available)
        """
        with self._data_lock:
            return (
                self._is_streaming and
                len(self.raw_buffer) >= self.feature_window and
                self.latest_features is not None and
                self._check_connection_health()
            )