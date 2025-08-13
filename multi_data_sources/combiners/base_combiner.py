from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class BaseCombiner(ABC):
    """
    Abstract base class for data source combiners.
    
    Defines the interface for combining different data formats (CSV market data, 
    JSON event data, etc.) into unified time-indexed datasets for analysis.
    
    Key Design Principles:
    - Preserve original data storage formats
    - Create time-aligned feature sets
    - Support both historical and live data flows
    - Enable systematic testing of data source combinations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the combiner with optional configuration.
        
        Args:
            config: Configuration dictionary for combiner-specific settings
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def combine_data(self, symbol: str, timeframe: str, date_range: Dict[str, str], 
                    data_sources: List[str]) -> pd.DataFrame:
        """
        Combine multiple data sources into a unified time-indexed DataFrame.
        
        Args:
            symbol: Trading symbol (e.g., 'GBPUSD', 'CLOV')
            timeframe: Time granularity (e.g., '1m', '5m', '1h', '1d')  
            date_range: Dictionary with 'start' and 'end' date strings
            data_sources: List of data source identifiers to combine
            
        Returns:
            pd.DataFrame: Time-indexed DataFrame with combined features
            
        Raises:
            ValueError: If required data sources are not available
            FileNotFoundError: If required data files are missing
        """
        pass
    
    @abstractmethod 
    def get_available_combinations(self, symbol: str) -> Dict[str, List[str]]:
        """
        Get available data source combinations for a given symbol.
        
        Args:
            symbol: Trading symbol to check
            
        Returns:
            Dictionary mapping data source types to available options
        """
        pass
    
    def validate_inputs(self, symbol: str, timeframe: str, date_range: Dict[str, str], 
                       data_sources: List[str]) -> bool:
        """
        Validate input parameters before combining data.
        
        Args:
            symbol: Trading symbol
            timeframe: Time granularity
            date_range: Date range dictionary
            data_sources: List of data sources
            
        Returns:
            bool: True if inputs are valid
            
        Raises:
            ValueError: If any input is invalid
        """
        # Validate symbol
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        
        # Validate timeframe
        valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
        if timeframe not in valid_timeframes:
            raise ValueError(f"Timeframe must be one of {valid_timeframes}")
        
        # Validate date range
        if not isinstance(date_range, dict) or 'start' not in date_range or 'end' not in date_range:
            raise ValueError("date_range must be a dict with 'start' and 'end' keys")
        
        try:
            start_date = pd.to_datetime(date_range['start'])
            end_date = pd.to_datetime(date_range['end'])
            if start_date >= end_date:
                raise ValueError("Start date must be before end date")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid date format in date_range: {e}")
        
        # Validate data sources
        if not data_sources or not isinstance(data_sources, list):
            raise ValueError("data_sources must be a non-empty list")
        
        return True
    
    def _load_market_data(self, symbol: str, timeframe: str, date_range: Dict[str, str]) -> pd.DataFrame:
        """
        Load market data from CSV files.
        
        Args:
            symbol: Trading symbol
            timeframe: Time granularity
            date_range: Date range to load
            
        Returns:
            pd.DataFrame: Time-indexed market data
        """
        from pathlib import Path
        import glob
        
        # Construct path pattern for market data
        data_dir = Path(f"storage/data/capital_com/{symbol}/raw/{timeframe}")
        
        if not data_dir.exists():
            raise FileNotFoundError(f"Market data directory not found: {data_dir}")
        
        # Find CSV files in date range
        csv_files = glob.glob(str(data_dir / "*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")
        
        # Load and combine CSV files
        dfs = []
        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path)
                
                # Handle both 'Date' (raw datasets) and 'date' (processed datasets)
                date_column = None
                if 'Date' in df.columns:
                    date_column = 'Date'
                elif 'date' in df.columns:
                    date_column = 'date'
                else:
                    self.logger.warning(f"No 'Date' or 'date' column found in {file_path}")
                    continue
                
                df[date_column] = pd.to_datetime(df[date_column])
                
                # Standardize column name to 'Date' for consistency
                if date_column == 'date':
                    df.rename(columns={'date': 'Date'}, inplace=True)
                
                dfs.append(df)
            except Exception as e:
                self.logger.warning(f"Error loading {file_path}: {e}")
        
        if not dfs:
            raise ValueError("No valid market data files could be loaded")
        
        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df = combined_df.sort_values('Date').drop_duplicates(subset=['Date'])
        
        # Filter by date range
        start_date = pd.to_datetime(date_range['start'])
        end_date = pd.to_datetime(date_range['end'])
        mask = (combined_df['Date'] >= start_date) & (combined_df['Date'] <= end_date)
        filtered_df = combined_df.loc[mask].copy()
        
        # Set index
        filtered_df.set_index('Date', inplace=True)
        
        self.logger.info(f"Loaded {len(filtered_df)} market data points for {symbol}")
        return filtered_df
    
    def _save_combined_data(self, combined_df: pd.DataFrame, symbol: str, 
                           timeframe: str, date_range: Dict[str, str], 
                           data_sources: List[str]) -> str:
        """
        Save combined dataset to storage.
        
        Args:
            combined_df: Combined DataFrame
            symbol: Trading symbol
            timeframe: Time granularity
            date_range: Date range
            data_sources: Data sources used
            
        Returns:
            str: Path to saved file
        """
        from pathlib import Path
        from datetime import datetime
        
        # Create output directory
        output_dir = Path(f"storage/combined_features/{symbol}/{timeframe}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        start_str = date_range['start'].replace('-', '')
        end_str = date_range['end'].replace('-', '')
        sources_str = '_'.join(sorted(data_sources))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"{start_str}_{end_str}_{sources_str}_{timestamp}.csv"
        filepath = output_dir / filename
        
        # Save with metadata
        combined_df.to_csv(filepath)
        
        # Save metadata
        metadata = {
            'symbol': symbol,
            'timeframe': timeframe,
            'date_range': date_range,
            'data_sources': data_sources,
            'total_rows': len(combined_df),
            'date_created': datetime.now().isoformat(),
            'filepath': str(filepath)
        }
        
        metadata_file = filepath.with_suffix('.json')
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Combined dataset saved: {filepath}")
        return str(filepath)