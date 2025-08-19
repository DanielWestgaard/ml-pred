import datetime
import re
import os
import pandas as pd
import sys
from utils.logging_utils import get_logger
from config import config

logger = get_logger()


def parse_timestamp_robust(timestamp_value):
    """
    Robust timestamp parsing function that handles multiple formats:
    - Unix timestamps (seconds and milliseconds)
    - ISO 8601 strings
    - Other common datetime formats
    
    Args:
        timestamp_value: The timestamp to parse (str, int, or float)
        logger: Optional logger instance for debugging
        
    Returns:
        pd.Timestamp: Parsed timestamp
        
    Raises:
        ValueError: If timestamp cannot be parsed
    """        
    if pd.isna(timestamp_value):
        return pd.NaT
        
    # Convert to string for consistent processing
    timestamp_str = str(timestamp_value).strip()
    
    # Handle empty or 'nan' strings
    if not timestamp_str or timestamp_str.lower() in ['nan', 'nat', 'none']:
        return pd.NaT
    
    try:
        # Try to convert to numeric (Unix timestamp)
        if timestamp_str.replace('.', '').isdigit():
            timestamp_numeric = float(timestamp_str)
            
            # Check if it's milliseconds (13+ digits) or seconds (10 digits)
            if timestamp_numeric > 1e12:  # Milliseconds
                logger.debug(f"Parsing Unix millisecond timestamp: {timestamp_str}")
                return pd.to_datetime(timestamp_numeric, unit='ms')
            elif timestamp_numeric > 1e9:  # Seconds
                logger.debug(f"Parsing Unix second timestamp: {timestamp_str}")
                return pd.to_datetime(timestamp_numeric, unit='s')
            else:
                # Too small to be a valid Unix timestamp, might be a different format
                raise ValueError(f"Numeric timestamp too small: {timestamp_numeric}")
        
        # Try common datetime string formats
        datetime_formats = [
            '%Y-%m-%dT%H:%M:%S',      # ISO 8601 basic
            '%Y-%m-%dT%H:%M:%S.%f',   # ISO 8601 with microseconds
            '%Y-%m-%dT%H:%M:%SZ',     # ISO 8601 with Z
            '%Y-%m-%dT%H:%M:%S.%fZ',  # ISO 8601 with microseconds and Z
            '%Y-%m-%d %H:%M:%S',      # Standard datetime
            '%Y-%m-%d %H:%M:%S.%f',   # Standard datetime with microseconds
            '%Y-%m-%d',               # Date only
            '%Y/%m/%d %H:%M:%S',      # Alternative format
            '%m/%d/%Y %H:%M:%S',      # US format
        ]
        
        for fmt in datetime_formats:
            try:
                parsed = datetime.datetime.strptime(timestamp_str, fmt)
                logger.debug(f"Parsed timestamp with format {fmt}: {timestamp_str}")
                return pd.Timestamp(parsed)
            except ValueError:
                continue
        
        # Last resort: let pandas handle it with mixed format
        logger.debug(f"Using pandas mixed format parsing for: {timestamp_str}")
        return pd.to_datetime(timestamp_str, format='mixed')
        
    except Exception as e:
        logger.error(f"Failed to parse timestamp '{timestamp_str}': {e}")
        raise ValueError(f"Unable to parse timestamp: {timestamp_str}")


def parse_timestamp_series(series):
    """
    Apply robust timestamp parsing to an entire pandas Series.
    
    Args:
        series: pandas Series containing timestamps
        logger: Optional logger instance
        
    Returns:
        pd.Series: Series with parsed timestamps
    """
    logger.debug(f"Parsing timestamp series with {len(series)} values")
    
    # Try vectorized approach first for performance
    try:
        # Check if all values are numeric (Unix timestamps)
        if series.dtype in ['int64', 'float64']:
            # Check if values are in millisecond range
            sample_value = float(str(series.iloc[0]))
            if sample_value > 1e12:
                logger.debug("Series appears to be Unix millisecond timestamps")
                return pd.to_datetime(series, unit='ms')
            elif sample_value > 1e9:
                logger.debug("Series appears to be Unix second timestamps")
                return pd.to_datetime(series, unit='s')
        elif series.astype(str).str.replace('.', '').str.isdigit().all():
            # All string values are numeric
            numeric_series = pd.to_numeric(series, errors='coerce')
            sample_value = numeric_series.iloc[0]
            if sample_value > 1e12:
                logger.debug("Series appears to be Unix millisecond timestamps (string format)")
                return pd.to_datetime(numeric_series, unit='ms')
            elif sample_value > 1e9:
                logger.debug("Series appears to be Unix second timestamps (string format)")  
                return pd.to_datetime(numeric_series, unit='s')
    except (ValueError, TypeError, IndexError):
        pass
    
    # Try pandas mixed format parsing
    try:
        logger.debug("Attempting pandas mixed format parsing")
        return pd.to_datetime(series, format='mixed', errors='coerce')
    except Exception as e:
        logger.warning(f"Mixed format parsing failed: {e}")
    
    # Fallback to element-wise parsing
    logger.debug("Falling back to element-wise timestamp parsing")
    return series.apply(lambda x: parse_timestamp_robust(x, logger))


def generate_filename(symbol, timeframe, start_date, end_date, is_raw=True, 
                     data_source=None, processing_info=None, extension='csv'):
    """
    Generate a standardized filename for financial data using new naming convention.
    
    New format: 'YYYY-MM-DD_YYYY-MM-DD.csv' (start_date_end_date.csv)
    Old format: 'raw_SYMBOL_timeframe_YYYYMMDD_YYYYMMDD.csv' (deprecated)
    
    Parameters:
    -----------
    symbol : str
        The trading symbol or asset identifier
    timeframe : str
        The timeframe of the data (e.g., '1m', '5m', '1h', '1d')
    start_date : datetime.datetime or str
        The start date of the data
    end_date : datetime.datetime or str
        The end date of the data
    is_raw : bool, default=True
        Flag indicating if the data is raw or processed (used for directory structure)
    data_source : str, optional
        The source of the data (e.g., 'capital_com', 'alpha_vantage')
    processing_info : str, optional
        Additional information about processing steps (deprecated in new format)
    extension : str, default='csv'
        The file extension without the dot
        
    Returns:
    --------
    str
        The filename in format: 'YYYY-MM-DD_YYYY-MM-DD.csv'
    """
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
    
    # Format dates as YYYY-MM-DD for new naming convention
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    # Create the final filename using new format: start_date_end_date.extension
    filename = f"{start_str}_{end_str}.{extension}"
    
    return filename


def check_input_type(input_data):
    """
    Check if the provided input is a string (CSV path), a pandas DataFrame,
    or something else. If it's a string, also verify it's a CSV file.
    
    Parameters:
    -----------
    input_data : str or pandas.DataFrame
        Either a path to a CSV file or a pandas DataFrame
    
    Returns:
    --------
    str
        'csv_path' if input is a string pointing to a CSV file,
        'dataframe' if input is a pandas DataFrame
    
    Raises:
    -------
    TypeError
        If input is neither a string nor a pandas DataFrame
    ValueError
        If input is a string but not a CSV file path
    """
    if isinstance(input_data, str):
        # Check if the string has a .csv extension
        if not input_data.lower().endswith('.csv'):
            logger.error(f"Provided input must be a (string) path to a CSV file (with .csv extension). You provided: {input_data}  ... exiting!")
            sys.exit()
        
        # Check if the file exists
        if not os.path.isfile(input_data):
            logger.error(f"CSV file does not exist: {input_data}  ... exiting!")
            sys.exit()
        
        return 'csv_path'
    elif isinstance(input_data, pd.DataFrame):
        return 'dataframe'
    else:
        logger.error(f"Input must be either a string (CSV path) or a pandas DataFrame. You provided {input_data}  ... exiting!")
        sys.exit()
    
def check_and_return_df(input_data) -> pd.DataFrame:
    """
    Method that checks what kind of input type it is, and returns a DataFrame if it exists.
    """
    logger.debug(f"Loading {input_data}")
    # Load dataset based on format
    try:
        input_type = check_input_type(input_data)
        
        if input_type == 'csv_path':
            # Load data from CSV file
            data = pd.read_csv(input_data, index_col=False)
            # logger.debug("Provided raw dataset was an existing csv-file.")
        else:  # input_type == 'dataframe'
            # Use DataFrame directly
            data = input_data.copy()
            # logger.debug("Provided raw dataset was a DataFrame.")
        original_data = data.copy()
        return data, original_data
    except (TypeError, ValueError) as e:
        logger.error(f"Error checking/loading df: {e}")
        return None
    

def check_validation(validation_results_is_valid : bool, validation_results_issues):
    """
    Method specifically designed to check the validation.py's results and gives 
    a descriptive logger based on the results.
    """
    
    if validation_results_is_valid:
        logger.info("Validation of cleaned data was valid!")
    else:
        logger.warning("Validation of data was not valid!")
        logger.warning(f"Validation issues: {validation_results_issues}")
        

def save_processed_file(filepath, processed_data: pd.DataFrame):
    """
    Save a processed version of the file using symbol-based directory structure.
    
    Structure: storage/data/capital_com/{SYMBOL}/processed/{timeframe}/filename.csv
    
    Note: As the method is saving the data, it's setting the index_label to 'date'. Using the Data Pipeline,
    the cleaning file (method timestamp_alignment()) sets 'date' as index, but when saving file with 'date'
    as index "the column gets 'lost' since indices don't have column names in the same way regular columns do.
    Hence, we have to "manually name it" when storing.
    
    Args:
        filepath: Original raw filepath: "storage/data/capital_com/GBPUSD/raw/5m/2024-01-01_2025-01-01.csv"
        processed_data: DataFrame to save
        
    Returns:
        str: Path where the processed file was saved
    """
    logger.info("Saving processed file ...")
    
    # Convert raw path to processed path
    output_path = filepath.replace('/raw/', '/processed/')
    processed_dir = os.path.dirname(output_path)
    
    # Ensure the processed directory exists
    os.makedirs(processed_dir, exist_ok=True)
    
    # Save the dataframe
    try:  # If df has 'date' as index (done in cleaning), we want to preserve the 'date' name of the column
        processed_data.to_csv(output_path, index=True, index_label='date')
    except:  # Might be failing because date is not the index? Saving standard just in case
        processed_data.to_csv(output_path, index=True)
    
    logger.info(f"Successfully saved file! {output_path}")
    
    return output_path




def get_data_file_path(symbol: str, timeframe: str, start_date: str, end_date: str, 
                      is_raw: bool = True, data_source: str = 'capital_com') -> str:
    """
    Generate complete file path for data using new symbol-based structure.
    
    Args:
        symbol: Trading symbol (e.g., 'GBPUSD', 'BTCUSD')
        timeframe: Data timeframe (e.g., '1m', '5m', '1h')
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        is_raw: Whether this is raw or processed data
        data_source: Data source ('capital_com' or 'alpha_vantage')
    
    Returns:
        str: Complete file path
    """
    # Generate filename
    filename = generate_filename(symbol, timeframe, start_date, end_date, 
                                is_raw=is_raw, extension='csv')
    
    if data_source == 'capital_com':
        if is_raw:
            directory = config.get_capcom_raw_data_dir(symbol, timeframe)
        else:
            directory = config.get_capcom_processed_data_dir(symbol, timeframe)
    elif data_source == 'alpaca':
        if is_raw:
            directory = config.get_alpaca_raw_data_dir(symbol, timeframe)
        else:
            directory = config.get_alpaca_processed_data_dir(symbol, timeframe)
    else:
        # Handle other data sources if needed
        raise ValueError(f"Unsupported data source: {data_source}")
    
    return os.path.join(directory, filename)


########## Utility used in data/features/target_hypothesis.py
def generate_misssing_required_features_for_target(missing_features : list):
    """
    Helper method generating any missing features in the processed dataset.
    
    Args:
        missing_features: List of features missing that's required for the generation of target hypothesis feature.
    
    Returns:
        pandas.dataframe: with missing features. This way, the actual method don't need to add them in its own dataframe, and can use this returning one as just a "temporary df".
    """
    
    # Try to generate the feature directly
    
        # Sometimes the feature has its own dependent features needed to be generated, so this could be a fallback or first/second test?
    
    # Returning resulting features
