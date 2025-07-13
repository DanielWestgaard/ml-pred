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
        
        # Components
        self.cleaner = DataCleaner()
        self.validator = DataValidator()
        
        logging.info(f"LiveDataManager initialized - Symbol: {symbol}, "
                    f"Timeframe: {timeframe}, Buffer: {buffer_size}")
    
    def add_data_callback(self, callback: Callable[[pd.DataFrame], None]):
        """Add callback function to be called when new processed data is available"""
        self._data_callbacks.append(callback)
    
    def start_streaming(self, dry_run: bool = False) -> bool:
        """
        Start live data streaming
        
        Args:
            dry_run: If True, simulate data instead of connecting to broker
            
        Returns:
            True if streaming started successfully
        """
        if self._is_streaming:
            logging.warning("Data streaming already active")
            return True
            
        try:
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
            
            logging.info(f"Live data streaming started - Mode: {'simulation' if dry_run else 'live'}")
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
        # Load some historical data for simulation
        base_price = 1.2500
        timestamp = datetime.now()
        
        while self._is_streaming:
            try:
                # Generate realistic OHLCV data
                price_change = np.random.normal(0, 0.0001)
                current_price = base_price + price_change
                
                # Simple OHLC simulation
                high = current_price + abs(np.random.normal(0, 0.00005))
                low = current_price - abs(np.random.normal(0, 0.00005))
                volume = np.random.randint(50, 500)
                
                data_point = {
                    'date': timestamp,
                    'open': base_price,
                    'high': high,
                    'low': low,
                    'close': current_price,
                    'volume': volume
                }
                
                self._data_queue.put(data_point)
                base_price = current_price
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
            
            # Clean data
            df_cleaned = self.cleaner.run(df.copy())
            
            # Validate data
            validation_results = self.validator.run(df_cleaned.copy())
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