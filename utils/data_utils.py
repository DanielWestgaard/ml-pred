import datetime
import logging
import re
import os
from pathlib import Path
import pandas as pd
import sys
import pandas as pd

import config.config as config


def parse_timestamp_robust(timestamp_value, logger=None):
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
    if logger is None:
        logger = logging.getLogger(__name__)
        
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


def parse_timestamp_series(series, logger=None):
    """
    Apply robust timestamp parsing to an entire pandas Series.
    
    Args:
        series: pandas Series containing timestamps
        logger: Optional logger instance
        
    Returns:
        pd.Series: Series with parsed timestamps
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
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
            logging.error(f"Provided input must be a (string) path to a CSV file (with .csv extension). You provided: {input_data}  ... exiting!")
            sys.exit()
        
        # Check if the file exists
        if not os.path.isfile(input_data):
            logging.error(f"CSV file does not exist: {input_data}  ... exiting!")
            sys.exit()
        
        return 'csv_path'
    elif isinstance(input_data, pd.DataFrame):
        return 'dataframe'
    else:
        logging.error(f"Input must be either a string (CSV path) or a pandas DataFrame. You provided {input_data}  ... exiting!")
        sys.exit()
    
def check_and_return_df(input_data) -> pd.DataFrame:
    """
    Method that checks what kind of input type it is, and returns a DataFrame if it exists.
    """
    logging.debug(f"Loading {input_data}")
    # Load dataset based on format
    try:
        input_type = check_input_type(input_data)
        
        if input_type == 'csv_path':
            # Load data from CSV file
            data = pd.read_csv(input_data, index_col=False)
            # logging.debug("Provided raw dataset was an existing csv-file.")
        else:  # input_type == 'dataframe'
            # Use DataFrame directly
            data = input_data.copy()
            # logging.debug("Provided raw dataset was a DataFrame.")
        original_data = data.copy()
        return data, original_data
    except (TypeError, ValueError) as e:
        logging.error(f"Error checking/loading df: {e}")
        return None
    

def check_validation(validation_results_is_valid : bool, validation_results_issues):
    """
    Method specifically designed to check the validation.py's results and gives 
    a descriptive logging based on the results.
    """
    
    if validation_results_is_valid:
        logging.info("Validation of cleaned data was valid!")
    else:
        logging.warning("Validation of data was not valid!")
        logging.warning(f"Validation issues: {validation_results_issues}")
        

def save_processed_file(filepath, processed_data : pd.DataFrame):
    """
    Save a processed version of the file:
    - Replace 'raw' with 'processed' in the filename (or add 'processed_' if 'raw' isn't found)
    - Save to ../processed/ directory
    
    Note: As the method is saving the data, it's setting the index_label to 'date'. Using the Data Pipeline,
    the cleaning file (method timestamp_alignment()) sets 'date' as index, but when saving file with 'date'
    as index "the column gets 'lost' since indices don't have column names in the same way regular columns do.
    Hence, we have to "manually name it" when storing.
    
    Args:
        filepath: Original filepath, e.g. "storage/data/capital_com/raw/raw_GBPUSD_m5_20240101_20250101.csv"
        
    Returns:
        str: Path where the processed file was saved
    """
    logging.info("Saving processed file ...")
    
    # Extract filename from filepath
    filename = os.path.basename(filepath)
    
    # Replace "raw" with "processed" in the filename
    if "raw" in filename:
        processed_filename = filename.replace("raw", "processed")
    else:
        # If "raw" isn't in the filename, add "processed_" prefix
        processed_filename = "processed_" + filename
    
    # Get parent directory of the original file's directory
    parent_dir = os.path.dirname(os.path.dirname(filepath))
    
    # Create the processed directory path
    processed_dir = os.path.join(parent_dir, "processed")
    
    # Ensure the processed directory exists
    os.makedirs(processed_dir, exist_ok=True)
    
    # Create the full output path
    output_path = os.path.join(processed_dir, processed_filename)
    
    # Save the dataframe
    try:  # If df has 'date' as index (done in cleaning), we want to preserve the 'date' name of the column
        processed_data.to_csv(output_path, index=True, index_label='date')
    except:  # Might be failing because date is not the index? Saving standard just in case
        processed_data.to_csv(output_path, index=True)
    
    logging.info(f"Successfully saved file! {output_path}")
    
    return output_path


def check_feature_specification(selected_features : list, method : str):
    """
    This is a method to determine the scope of feature selection, and is meant to be used in feature generation methods.
    This is what I mean with the weird wording: { feature_name_1: [feature_value_1, feature_value_2], feature_name_2: []}
    
    The approach is as follows:
        - Is the method being ran with or without a hypothesis (meaning is the selected_features array empty)?
            - If the list is empty, this function returns True (as it is no hypotheis ran) and None (since the method naturally has no specified feature values).
            - If the list has values (we're assuming the method is called because it is in the list): Does 'this' feature (eg. time_of_day) have specified
            excactly what (granular) features it wants to be generated (eg. hour_sin, hour_cos)? Or is the list empty - Meaning all features of the will be
            generated anyway (in that method).
                - If the feature values (eg. time_of_day) does not have feature names (eg. hour_sin, hour_cos), function returns True (as all features of the method
                then is meant to be generated), and the feature values (selected_features[method]) (though this will be an empty list: '[]').
                - If the feature values (eg. time_of_day) has feature names (eg. hour_sin, hour_cos), function returns False (as not all features of the method will
                be generated), and the these values (selected_features[method]). Meaning only those feature names will be generated.
    
    Args:
        - selected_features: The whole list of key-value pairs. Eg. Looks like this: { time_of_day: ['hour_sin', 'hour_cos'], 'time_of_week': [], 'moving_average': [] }
        - method: The feature/method you are using/calculating the feature/featuers of. Eg. 'time_of_day' or 'moving_averages'
    
    Returns:
        - bool: wether or not all features are to be generated, or if there are provided specific features within method to be generated
            True => Generate all features (in the method).
            False => Only generate the features of selected_features[method].
        - values of feature: list of the features values, selected_features[method]. Eg. selected_features['time_of_day']
            If the selected_features is empty, method returns None. If it exists, we just return that list.
    """
    logging.debug(f"----- Inside check_feature_specification() to determine how to generate hypotheses feature: {method} -----")
    # First, check if the whole list is full or empty.
    if selected_features is None:  # List is empty ->  All features are to be generated
        logging.debug(f"selected_features (the whole list) is empty. All features are to be generated of method {method}")
        return True, None
    else:  # List has specified features to generate
        # As the method has been (manually) called, I'm ASSUMING this feature is in the list (so won't check)
        values = selected_features[method]
        if not values:  # List of feature values is empty -> Not (granularly) specified features, meaning all features in that feature method will be generated
            logging.debug(f"List of feature ({method}) values is empty: {values}. Will still generate all features.")
            return True, values
        else:  # Not all features are to be generated, but specified ones
            logging.debug(f"List of feature ({method}) values contains values: {values}. Will only generate those.")
            return False, values