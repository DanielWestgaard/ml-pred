import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Iterator
from pathlib import Path
from utils.logging_utils import get_logger

logger = get_logger()

class BacktestDataManager:
    """
    Manages historical data iteration for backtesting.
    Replaces live data streaming with historical data playback.
    """
    
    def __init__(self, data_path: str, start_date: str = None, end_date: str = None,
                 buffer_size: int = 200, **kwargs):
        """
        Initialize backtest data manager
        
        Args:
            data_path: Path to processed historical data CSV
            start_date: Start date for backtesting (YYYY-MM-DD)
            end_date: End date for backtesting (YYYY-MM-DD)
            buffer_size: Number of historical rows to maintain for feature generation
        """
        self.data_path = Path(data_path)
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
        self.buffer_size = buffer_size
        
        # Data storage
        self.full_data = None
        self.backtest_data = None
        self.current_index = 0
        self.data_buffer = []
        
        # Load and prepare data
        self._load_data()
        self._prepare_backtest_data()
        
        logger.info(f"Backtest data manager initialized - {len(self.backtest_data)} rows "
                   f"from {self.backtest_data.index[0]} to {self.backtest_data.index[-1]}")
    
    def _load_data(self):
        """Load historical data from CSV"""
        try:
            if not self.data_path.exists():
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
            logger.info(f"Loading historical data from: {self.data_path}")
            self.full_data = pd.read_csv(self.data_path)
            
            # Ensure timestamp column is properly formatted
            if 'timestamp' in self.full_data.columns:
                self.full_data['timestamp'] = pd.to_datetime(self.full_data['timestamp'])
                self.full_data.set_index('timestamp', inplace=True)
            elif self.full_data.index.name != 'timestamp':
                # Try to use index as timestamp if it looks like datetime
                try:
                    self.full_data.index = pd.to_datetime(self.full_data.index)
                    self.full_data.index.name = 'timestamp'
                except:
                    logger.warning("Could not parse index as timestamp, using as-is")
            
            # Sort by timestamp to ensure proper order
            self.full_data.sort_index(inplace=True)
            
            logger.info(f"Loaded {len(self.full_data)} rows spanning "
                       f"{self.full_data.index[0]} to {self.full_data.index[-1]}")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _prepare_backtest_data(self):
        """Prepare data slice for backtesting based on date range"""
        if self.full_data is None:
            raise ValueError("No data loaded")
        
        # Apply date filters
        data_slice = self.full_data.copy()
        
        if self.start_date:
            data_slice = data_slice[data_slice.index >= self.start_date]
            logger.info(f"Filtered data from start_date: {self.start_date}")
        
        if self.end_date:
            data_slice = data_slice[data_slice.index <= self.end_date]
            logger.info(f"Filtered data to end_date: {self.end_date}")
        
        if len(data_slice) == 0:
            raise ValueError(f"No data found in date range {self.start_date} to {self.end_date}")
        
        self.backtest_data = data_slice
        
        # Initialize buffer with first buffer_size rows (needed for feature generation)
        buffer_end = min(self.buffer_size, len(self.backtest_data))
        self.data_buffer = [
            self._create_data_point(i, self.backtest_data.iloc[i])
            for i in range(buffer_end)
        ]
        self.current_index = buffer_end
        
        logger.info(f"Backtest data prepared: {len(self.backtest_data)} rows, "
                   f"buffer initialized with {len(self.data_buffer)} rows")
    
    def _create_data_point(self, index: int, row: pd.Series) -> Dict[str, Any]:
        """
        Create standardized data point from DataFrame row
        
        Args:
            index: Row index in backtest data
            row: Pandas Series for the row
            
        Returns:
            Standardized data point dictionary
        """
        return {
            'timestamp': row.name,  # Index should be timestamp
            'index': index,
            'open': row.get('open', row.get('open_og', 0.0)),
            'high': row.get('high', 0.0),
            'low': row.get('low', 0.0),
            'close': row.get('close', row.get('close_og', 0.0)),
            'volume': row.get('volume', 0.0),
            'mid_price': row.get('mid_price', row.get('close', 0.0)),
            # Include all other columns as features
            **{col: row.get(col, 0.0) for col in row.index if col not in 
               ['open', 'high', 'low', 'close', 'volume', 'mid_price', 'open_og', 'close_og']}
        }
    
    def get_next_data_point(self) -> Optional[Dict[str, Any]]:
        """
        Get next data point in backtest sequence
        
        Returns:
            Next data point or None if end of data reached
        """
        if self.current_index >= len(self.backtest_data):
            return None
        
        # Get current row
        current_row = self.backtest_data.iloc[self.current_index]
        data_point = self._create_data_point(self.current_index, current_row)
        
        # Add to buffer and maintain buffer size
        self.data_buffer.append(data_point)
        if len(self.data_buffer) > self.buffer_size:
            self.data_buffer.pop(0)
        
        self.current_index += 1
        
        return data_point
    
    def get_current_data_buffer(self) -> pd.DataFrame:
        """
        Get current data buffer as DataFrame for feature generation
        
        Returns:
            DataFrame with recent historical data for feature calculation
        """
        if not self.data_buffer:
            return pd.DataFrame()
        
        # Convert buffer to DataFrame
        buffer_df = pd.DataFrame(self.data_buffer)
        
        # Set timestamp as index
        if 'timestamp' in buffer_df.columns:
            buffer_df.set_index('timestamp', inplace=True)
        
        return buffer_df
    
    def get_latest_data_point(self) -> Optional[Dict[str, Any]]:
        """Get the most recent data point"""
        return self.data_buffer[-1] if self.data_buffer else None
    
    def get_current_price(self) -> float:
        """Get current market price (close price of latest data point)"""
        latest = self.get_latest_data_point()
        return latest['close'] if latest else 0.0
    
    def has_more_data(self) -> bool:
        """Check if more data is available for backtesting"""
        return self.current_index < len(self.backtest_data)
    
    def get_progress_info(self) -> Dict[str, Any]:
        """
        Get current progress information
        
        Returns:
            Progress info dictionary
        """
        total_rows = len(self.backtest_data)
        processed_rows = self.current_index
        progress_pct = (processed_rows / total_rows) * 100 if total_rows > 0 else 0
        
        current_timestamp = None
        if self.data_buffer:
            current_timestamp = self.data_buffer[-1]['timestamp']
        
        return {
            'processed_rows': processed_rows,
            'total_rows': total_rows,
            'progress_percentage': progress_pct,
            'current_timestamp': current_timestamp,
            'remaining_rows': total_rows - processed_rows,
            'has_more_data': self.has_more_data()
        }
    
    def peek_next_data_points(self, count: int = 1) -> List[Dict[str, Any]]:
        """
        Peek at next data points without advancing the iterator
        
        Args:
            count: Number of data points to peek ahead
            
        Returns:
            List of upcoming data points
        """
        peek_data = []
        
        for i in range(count):
            peek_index = self.current_index + i
            if peek_index >= len(self.backtest_data):
                break
            
            row = self.backtest_data.iloc[peek_index]
            data_point = self._create_data_point(peek_index, row)
            peek_data.append(data_point)
        
        return peek_data
    
    def skip_to_timestamp(self, target_timestamp: pd.Timestamp):
        """
        Skip to a specific timestamp in the data
        
        Args:
            target_timestamp: Target timestamp to skip to
        """
        # Find the index for target timestamp
        try:
            target_index = self.backtest_data.index.get_loc(target_timestamp)
            if isinstance(target_index, slice):
                target_index = target_index.start
            
            # Reset to target position
            self.current_index = max(self.buffer_size, target_index)
            
            # Rebuild buffer
            buffer_start = max(0, self.current_index - self.buffer_size)
            buffer_end = self.current_index
            
            self.data_buffer = [
                self._create_data_point(i, self.backtest_data.iloc[i])
                for i in range(buffer_start, buffer_end)
            ]
            
            logger.info(f"Skipped to timestamp: {target_timestamp}")
            
        except KeyError:
            logger.warning(f"Timestamp {target_timestamp} not found in data")
    
    def reset(self):
        """Reset data manager to beginning"""
        self._prepare_backtest_data()
        logger.info("Backtest data manager reset to beginning")
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary of available data
        
        Returns:
            Data summary dictionary
        """
        if self.backtest_data is None:
            return {'error': 'No data loaded'}
        
        numeric_columns = self.backtest_data.select_dtypes(include=[np.number]).columns
        
        return {
            'total_rows': len(self.backtest_data),
            'date_range': {
                'start': str(self.backtest_data.index[0]),
                'end': str(self.backtest_data.index[-1])
            },
            'columns': list(self.backtest_data.columns),
            'numeric_columns': list(numeric_columns),
            'data_completeness': {
                col: (1 - self.backtest_data[col].isna().sum() / len(self.backtest_data)) * 100
                for col in numeric_columns[:10]  # Show first 10 columns
            },
            'buffer_size': self.buffer_size,
            'current_progress': self.get_progress_info()
        }