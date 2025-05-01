import datetime
import logging
import re
import os
from pathlib import Path
import pandas as pd

import config.config as config


def generate_filename(symbol, timeframe, start_date, end_date, is_raw=True, 
                     data_source=None, processing_info=None, extension='csv'):
    """
    Generate a standardized filename for financial data.
    
    Parameters:
    -----------
    symbol : str
        The trading symbol or asset identifier
    timeframe : str
        The timeframe of the data (e.g., '1min', '1h', '1d')
    start_date : datetime.datetime or str
        The start date of the data
    end_date : datetime.datetime or str
        The end date of the data
    is_raw : bool, default=True
        Flag indicating if the data is raw or processed
    data_source : str, optional
        The source of the data (e.g., 'yahoo', 'bloomberg')
    processing_info : str, optional
        Additional information about processing steps (only used if is_raw=False)
    extension : str, default='csv'
        The file extension without the dot
    """
    # Sanitize symbol (remove non-alphanumeric characters except underscore)
    symbol = re.sub(r'[^\w]', '_', symbol)
    
    # Standardize timeframe format
    timeframe = timeframe.lower().replace('minute', 'm').replace('hour', 'h').replace('day', 'd').replace('_', '')
    
    # Convert dates to datetime objects if they are strings
    if isinstance(start_date, str):
        try:
            # Try common date formats
            for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%Y-%m-%dT%H:%M:%S']:  #2020-02-24T00:00:00
                try:
                    start_date = datetime.datetime.strptime(start_date, fmt)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(f"Could not parse start_date: {start_date}")
        except ValueError as e:
            raise ValueError(f"Could not parse start_date: {e}")
            
    if isinstance(end_date, str):
        try:
            for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%Y-%m-%dT%H:%M:%S']:
                try:
                    end_date = datetime.datetime.strptime(end_date, fmt)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(f"Could not parse end_date: {end_date}")
        except ValueError as e:
            raise ValueError(f"Could not parse end_date: {e}")
    
    # Format dates as YYYYMMDD
    start_str = start_date.strftime('%Y%m%d')
    end_str = end_date.strftime('%Y%m%d')
    
    # Base name components
    components = [symbol, timeframe, start_str, end_str]
    
    # Add data source if provided
    if data_source:
        sanitized_source = re.sub(r'[^\w]', '_', data_source)
        components.append(sanitized_source)
    
    base_name = "_".join(components)
    
    # Add prefix based on whether the data is raw or processed
    if is_raw:
        prefix = "raw"
    else:
        prefix = "processed"
        # Add processing info if provided and data is processed
        if processing_info:
            sanitized_info = re.sub(r'[^\w]', '_', processing_info)
            prefix = f"{prefix}_{sanitized_info}"
    
    # Create the final filename
    filename = f"{prefix}_{base_name}.{extension}"
    
    return filename

# More enhanced and should be more widely used
def save_financial_data(data, file_type, **kwargs):
    """
    Save financial data as either raw or processed files.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The data to be saved
    file_type : str
        Type of data: 'raw' or 'processed'
    **kwargs :
        For raw files:
            symbol, timeframe, start_date, end_date : required for filename generation
            data_source, extension : optional for filename generation
            location : optional directory to save file
            
        For processed files:
            raw_filename : required - the original raw filename (can be a full path)
            processing_info : optional - additional processing information
            location : optional directory to save file
    
    Returns:
    --------
    str
        The path to the saved file
    """
    import os
    import logging
    import re
    
    # Generate filename based on file_type
    if file_type == 'raw':
        # Validate required parameters for raw files
        required_params = ['symbol', 'timeframe', 'start_date', 'end_date']
        missing = [p for p in required_params if p not in kwargs]
        if missing:
            raise ValueError(f"Missing required parameters for raw file: {', '.join(missing)}")
        
        # Generate raw filename
        filename = generate_filename(
            symbol=kwargs['symbol'],
            timeframe=kwargs['timeframe'],
            start_date=kwargs['start_date'],
            end_date=kwargs['end_date'],
            is_raw=True,
            data_source=kwargs.get('data_source'),
            extension=kwargs.get('extension', 'csv')
        )
        
        # Set location for raw files
        location = kwargs.get('location', config.CAPCOM_RAW_DATA_DIR)
        
    elif file_type == 'processed':
        # Validate we have a raw filename for processed files
        if 'raw_filename' not in kwargs or not kwargs['raw_filename']:
            raise ValueError("Raw filename is required for processed files")
        
        # Extract just the filename if a full path is provided
        raw_filename = os.path.basename(kwargs['raw_filename'])
        
        # Validate it's a raw filename
        if not raw_filename.startswith("raw_"):
            raise ValueError("Raw filename must start with 'raw_'")
        
        # Convert to processed filename
        filename = raw_filename.replace("raw_", "processed_", 1)
        
        # Add processing info if provided
        if processing_info := kwargs.get('processing_info'):
            # Split into base name and extension
            name_parts = filename.rsplit('.', 1)
            base_name = name_parts[0]
            extension = name_parts[1] if len(name_parts) > 1 else 'csv'
            
            # Sanitize and add processing info
            sanitized_info = re.sub(r'[^\w]', '_', processing_info)
            filename = f"{base_name}_{sanitized_info}.{extension}"
        
        # Set location for processed files
        location = kwargs.get('location', config.CAPCOM_PROCESSED_DATA_DIR)
        
    else:
        raise ValueError(f"Invalid file_type: {file_type}. Must be 'raw' or 'processed'")
    
    # Ensure directory exists
    os.makedirs(location, exist_ok=True)
    
    # Create full file path
    file_path = os.path.join(location, filename)
    
    # Save data to file
    data.to_csv(file_path, index=False)
    
    # Log and return file path
    logging.info(f"File saved at: {file_path}")
    return file_path

def get_file_path(filename, base_dir=None, create_dirs=True):
    """
    Generate a full path for a file and optionally create the directories.
    
    Parameters:
    -----------
    filename : str
        The generated filename
    base_dir : str, optional
        The base directory to save files. Defaults to current working directory.
    create_dirs : bool, default=True
        Whether to create directories if they don't exist
    """
    if not base_dir:
        base_dir = os.getcwd()
    
    # Determine if it's raw or processed from the filename
    if filename.startswith("raw_"):
        sub_dir = "raw_data"
    elif filename.startswith("processed_"):
        sub_dir = "processed_data"
    else:
        sub_dir = "other_data"
    
    # Create the directory path
    dir_path = os.path.join(base_dir, sub_dir)
    
    # Create directories if they don't exist and create_dirs is True
    if create_dirs and not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    # Return the full file path
    return os.path.join(dir_path, filename)

# Have a more enhanced version, but sometimes simpler might be better to use at that time
def save_data_file(data:pd.DataFrame, type:str, filename:str, location:str=None):
    logging.info("Saving file...")
    
    if location is None:
        if type == 'processed':
            location = sys_config.CAPCOM_PROCESSED_DATA_DIR
        else:  # raw
            location = sys_config.CAPCOM_RAW_DATA_DIR
    else:
        location = location
        
    # Ensure the directory exists
    os.makedirs(location, exist_ok=True)
    # Create the full file path
    file_path = os.path.join(location, filename)
    
    # Write content to file
    data.to_csv(file_path, index=False)

    logging.info(f"File saved at: {file_path}")

def extract_file_metadata(filename):
    """
    Extract metadata from a raw or processed filename.
    
    Parameters:
    -----------
    filename : str
        The filename to parse (can be a full path)
        
    Returns:
    --------
    dict
        Dictionary containing extracted metadata like instrument, timeframe, dates
    """
    # Get just the filename if full path was provided
    filename = os.path.basename(filename)
    
    # Use regex to extract components from filenames like:
    # raw_GBPUSD_m5_20240101_20250101.csv or
    # processed_GBPUSD_m5_20240101_20250101.csv
    pattern = r'(?:raw|processed|meta)_([A-Z]+)_([a-z0-9]+)_(\d+)_(\d+)(?:\..*)?'
    match = re.match(pattern, filename)
    
    if not match:
        logging.warning(f"Could not extract metadata from filename: {filename}")
        return None
    
    # Extract the components
    instrument, timeframe, start_date, end_date = match.groups()
    
    return {
        'instrument': instrument,
        'timeframe': timeframe,
        'start_date': start_date,
        'end_date': end_date,
        'is_raw': filename.startswith('raw_'),
        'is_processed': filename.startswith('processed_'),
        'is_meta': filename.startswith('meta_'),
        'base_name': f"{instrument}_{timeframe}_{start_date}_{end_date}"
    }

def generate_derived_filename(processed_file, file_type, extension='csv'):
    """
    Generate a filename for a derived file based on a processed data file.
    
    Parameters:
    -----------
    processed_file : str
        Path to the processed file that this is derived from
    file_type : str
        Type of derived file (e.g., 'feature_importance', 'category_importance', 'model_performance')
    extension : str
        File extension (without the dot)
    
    Returns:
    --------
    str
        The derived filename
    """
    # Extract metadata from the processed filename
    metadata = extract_file_metadata(processed_file)
    if not metadata:
        # Fallback to simple naming if extraction fails
        base_name = os.path.splitext(os.path.basename(processed_file))[0]
        return f"{file_type}_{base_name}.{extension}"
    
    # Create derived filename with the pattern: file_type_BASE_NAME.extension
    return f"{file_type}_{metadata['base_name']}.{extension}"

def get_derived_file_path(processed_file, file_type, base_dir=None, sub_dir=None, extension='csv'):
    """Get the full path for a derived file."""
    # Generate the derived filename
    filename = generate_derived_filename(processed_file, file_type, extension)
    
    # Determine the base directory
    if not base_dir:
        base_dir = os.path.dirname(processed_file)
    
    # Create the full directory path
    if sub_dir:
        dir_path = os.path.join(base_dir, sub_dir)
    else:
        # Don't add another subdirectory if we're already in the correct category directory
        current_dir = os.path.basename(base_dir)
        if (file_type in ['feature_importance', 'category_importance'] and current_dir == 'features') or \
           (file_type in ['model_performance'] and current_dir == 'performance'):
            dir_path = base_dir
        # Otherwise use the default subdirectory logic
        elif file_type in ['feature_importance', 'category_importance', 'selected_features']:
            dir_path = os.path.join(base_dir, 'features')
        elif file_type in ['model_performance', 'prediction_results', 'backtest_results']:
            dir_path = os.path.join(base_dir, 'performance')
        elif file_type in ['meta', 'metadata']:
            dir_path = base_dir  # Keep metadata in same directory as processed file
        else:
            # Default subdirectory based on file type
            dir_path = os.path.join(base_dir, file_type.split('_')[0] + 's')
    
    # Ensure the directory exists
    os.makedirs(dir_path, exist_ok=True)
    
    # Return the full file path
    return os.path.join(dir_path, filename)

# Data column related
def _find_column_by_pattern(df, pattern):
    """
    Find a column in the dataframe that matches the given pattern.
    
    Args:
        df: DataFrame to search in
        pattern: String pattern to match in column names
        
    Returns:
        The name of the column if found, None otherwise
    """
    # First check for exact match (which would be faster)
    if pattern in df.columns:
        return pattern
        
    # Then look for pattern in column names
    for col in df.columns:
        if pattern in col.lower():
            return col
            
    # Return None if no match is found
    return None

def get_price_columns(df):
    """
    Find all the necessary price columns (open, high, low, close) in the dataframe.
    
    Args:
        df: DataFrame to search in
        
    Returns:
        Dictionary mapping standard names to actual column names found
    """
    columns = {}
    
    # Find each price column
    columns['open'] = _find_column_by_pattern(df, 'open')
    columns['high'] = _find_column_by_pattern(df, 'high')
    columns['low'] = _find_column_by_pattern(df, 'low')
    columns['close'] = _find_column_by_pattern(df, 'close')
    
    return columns