import datetime
import logging
import re
import os
from pathlib import Path
import pandas as pd
import sys

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
    
def check_and_return_df(input_data):
    """
    Method that checks what kind of input type it is, and returns a DataFrame if it exists.
    """
    # Load dataset based on format
    try:
        input_type = check_input_type(input_data)
        
        if input_type == 'csv_path':
            # Load data from CSV file
            data = pd.read_csv(input_data, index_col=False)
            logging.debug("Provided raw dataset was an existing csv-file.")
        else:  # input_type == 'dataframe'
            # Use DataFrame directly
            data = input_data.copy()
            logging.debug("Provided raw dataset was a DataFrame.")
        original_data = data.copy()
        return data, original_data
    except (TypeError, ValueError) as e:
        logging.error(f"Error: {e}")
        return None