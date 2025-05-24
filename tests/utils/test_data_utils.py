import pandas as pd
import numpy as np
import os
import tempfile
from datetime import datetime, timedelta, timezone

def create_sample_ohlcv_data(rows=100, with_issues=False, frequency='1h'):
    """
    Create sample OHLCV data for testing.
    
    Parameters:
    -----------
    rows : int
        Number of rows in the dataset
    with_issues : bool
        If True, inject data quality issues for testing cleaning methods
    frequency : str
        Time frequency of the data ('1H' for hourly, '1D' for daily, etc.)
        
    Returns:
    --------
    pd.DataFrame with OHLCV data
    """
    # Create date range
    end_date = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(hours=rows if frequency == '1h' else rows * 24)
    dates = pd.date_range(start=start_date, end=end_date, freq=frequency)
    
    # Base prices (random walk)
    np.random.seed(42)  # For reproducibility
    base_price = 100.0
    returns = np.random.normal(0, 0.01, len(dates))
    price_multipliers = np.cumprod(1 + returns)
    base_prices = base_price * price_multipliers
    
    # Create OHLCV data
    df = pd.DataFrame(index=dates)
    
    # Valid OHLC values
    volatility = base_prices * 0.02
    df['open'] = base_prices * (1 + np.random.normal(0, 0.005, len(dates)))
    df['high'] = df['open'] + volatility * np.random.random(len(dates))
    df['low'] = df['open'] - volatility * np.random.random(len(dates))
    df['close'] = df['low'] + (df['high'] - df['low']) * np.random.random(len(dates))
    
    # Ensure OHLC relationships (high >= open,close >= low)
    for i in range(len(df)):
        high = max(df['open'].iloc[i], df['close'].iloc[i], df['high'].iloc[i])
        low = min(df['open'].iloc[i], df['close'].iloc[i], df['low'].iloc[i])
        df.loc[df.index[i], 'high'] = high
        df.loc[df.index[i], 'low'] = low
    
    # Volume as function of price volatility
    price_range = df['high'] - df['low']
    df['volume'] = (price_range * 1000000 * (1 + np.random.random(len(dates)))).astype(int)
    
    # Convert index to column for saving to CSV
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'date'}, inplace=True)
    
    if with_issues:
        # Inject various data issues for testing cleaning functions
        
        # 1. Missing values (10% of the data)
        mask = np.random.random(len(df)) < 0.1
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df.loc[mask, col] = np.nan
        
        # 2. Duplicate dates (5% of the data)
        duplicate_indices = np.random.choice(df.index[10:], size=int(len(df) * 0.05), replace=False)
        for idx in duplicate_indices:
            df.loc[idx, 'date'] = df.loc[idx-1, 'date']
        
        # 3. Outliers in prices (3% of the data)
        outlier_indices = np.random.choice(df.index, size=int(len(df) * 0.03), replace=False)
        for idx in outlier_indices:
            multiplier = np.random.choice([0.1, 10])  # Either very low or very high
            for col in ['open', 'high', 'low', 'close']:
                df.loc[idx, col] = df.loc[idx, col] * multiplier
        
        # 4. Outliers in volume (2% of the data)
        volume_outlier_indices = np.random.choice(df.index, size=int(len(df) * 0.02), replace=False)
        for idx in volume_outlier_indices:
            df.loc[idx, 'volume'] = df.loc[idx, 'volume'] * 50
        
        # 5. Invalid OHLC relationships (5% of the data)
        ohlc_invalid_indices = np.random.choice(df.index, size=int(len(df) * 0.05), replace=False)
        for idx in ohlc_invalid_indices:
            # Make low higher than open/close or high lower than open/close
            if np.random.random() < 0.5:
                df.loc[idx, 'low'] = df.loc[idx, 'open'] * 1.1
            else:
                df.loc[idx, 'high'] = df.loc[idx, 'open'] * 0.9
        
        # 6. Negative or zero values (2% of the data)
        negative_indices = np.random.choice(df.index, size=int(len(df) * 0.02), replace=False)
        for idx in negative_indices:
            if np.random.random() < 0.5:
                # Negative volume
                df.loc[idx, 'volume'] = -df.loc[idx, 'volume']
            else:
                # Zero price
                col = np.random.choice(['open', 'high', 'low', 'close'])
                df.loc[idx, col] = 0
    
    return df

def save_temp_csv(df):
    """
    Save dataframe to a temporary CSV file and return the file path.
    """
    fd, path = tempfile.mkstemp(suffix='.csv')
    os.close(fd)
    df.to_csv(path, index=False)
    return path

def clean_temp_file(path):
    """
    Delete the temporary file.
    """
    if os.path.exists(path):
        os.remove(path)

def get_test_data_path():
    """Returns path to test data directory"""
    # Assuming tests are in a "tests" directory at the same level as the source code
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_data_dir = os.path.join(current_dir, 'test_data')
    
    # Create directory if it doesn't exist
    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)
    
    return test_data_dir

def mock_model_utils():
    """Create a mock for model_utils that might be imported in feature selection"""
    import sys
    from unittest.mock import MagicMock
    
    # Create mock module
    mock_module = MagicMock()
    
    # Set up preprocess_features_for_xgboost to return the input dataframe
    mock_module.preprocess_features_for_xgboost = lambda df, enable_categorical: df
    
    # Add to sys.modules to be importable
    sys.modules['utils.model_utils'] = mock_module
    return mock_module