from datetime import datetime, timedelta
import http.client
import json
import os
import sys
import logging
import time
import base64
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from dateutil.relativedelta import relativedelta
import pandas as pd

import config.config as config
from utils.broker_utils import convert_json_to_ohlcv_csv
import utils.data_utils as data_utils


# Global variables
conn = http.client.HTTPSConnection(config.CAPCOM_DEMO_URL)

def historical_prices(X_SECURITY_TOKEN, CST,
                      epic, resolution, from_date, to_date,  # format 2022-02-24T00:00:00
                      max, print_answer):
    """
    Returns historical prices for a particular instrument. The maximum number of the values in answer is max = 1000.
    
    Args:
        epic: Instrument epic/symbol. Ex. GOLD, GBPUSD.
        resolution: Defines the resolution of requested prices. Possible values are MINUTE, MINUTE_5, MINUTE_15, MINUTE_30, HOUR, HOUR_4, DAY, WEEK.
        from_date: 
        to_date
        max: 
        The maximum number of the values in answer. Default = 10, max = 1000
    Returns:

    """
    
    payload = ''
    headers = {
        'X-SECURITY-TOKEN': X_SECURITY_TOKEN,
        'CST': CST
    }
    
    conn.request("GET", f"/api/v1/prices/{epic}?resolution={resolution}&max={max}&from={from_date}&to={to_date}", payload, headers)
    #conn.request("GET", "/api/v1/prices/SILVER?resolution=MINUTE&max=10&from=2020-02-24T00:00:00&to=2020-02-24T01:00:00", payload, headers)
    res = conn.getresponse()
    data = res.read()

    parsed_data = json.loads(data.decode("utf-8"))
    if print_answer:
        print(json.dumps(parsed_data, indent=4))
    return parsed_data


def calculate_optimal_chunk_size(resolution, max_points=900):
    """
    Calculate optimal chunk size based on resolution to stay under max_points limit.
    Using 900 instead of 1000 to leave some buffer for API limitations.
    """
    # Minutes per interval for each resolution
    minutes_per_interval = {
        "MINUTE": 1,
        "MINUTE_5": 5,
        "MINUTE_15": 15,
        "MINUTE_30": 30,
        "HOUR": 60,
        "HOUR_4": 240,
        "DAY": 1440,  # 24 * 60
        "WEEK": 10080  # 7 * 24 * 60
    }
    
    # Calculate how many minutes we can fetch to stay under max_points
    interval_minutes = minutes_per_interval.get(resolution, 1)
    total_minutes_for_max_points = max_points * interval_minutes
    
    # Convert to timedelta
    return timedelta(minutes=total_minutes_for_max_points)


def fetch_chunk_with_retry(X_SECURITY_TOKEN, CST, epic, resolution, 
                          from_date, to_date, max_retries=3, print_answer=False):
    """
    Fetch a chunk of data with automatic retry using smaller chunks if it fails.
    """
    current_from = from_date
    current_to = to_date
    
    for attempt in range(max_retries):
        try:
            # Format dates for API call
            from_str = current_from.strftime("%Y-%m-%dT%H:%M:%S")
            to_str = current_to.strftime("%Y-%m-%dT%H:%M:%S")
            
            logging.debug(f"Attempt {attempt + 1}: Fetching from {from_str} to {to_str}")
            
            # Call the API
            chunk_data = historical_prices(
                X_SECURITY_TOKEN, 
                CST, 
                epic, 
                resolution, 
                from_str, 
                to_str, 
                max=1000, 
                print_answer=print_answer
            )
            
            # Check if we got an error
            if "errorCode" in chunk_data:
                if chunk_data["errorCode"] == "error.invalid.max.daterange":
                    # Reduce the time range by half and try again
                    time_diff = current_to - current_from
                    current_to = current_from + time_diff / 2
                    logging.warning(f"Date range too large, reducing to {current_to.strftime('%Y-%m-%dT%H:%M:%S')}")
                    continue
                else:
                    # Other error, return it
                    logging.error(f"API Error: {chunk_data}")
                    return chunk_data, current_to
            else:
                # Success!
                return chunk_data, current_to
                
        except Exception as e:
            logging.error(f"Error in attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                return {"error": str(e)}, current_to
            
            # Reduce time range and try again
            time_diff = current_to - current_from
            current_to = current_from + time_diff / 2
    
    return {"error": "Max retries exceeded"}, current_to


def fetch_and_save_historical_prices(X_SECURITY_TOKEN, CST, epic, resolution, 
                                    from_date, to_date, output_file,
                                    print_answer, save_raw_data):
    """
    Fetches historical price data over an extended period and saves it to a CSV file.
    Improved version with better chunking and error handling.
    
    Args:
        X_SECURITY_TOKEN: Security token for API authentication
        CST: CST value for API authentication
        epic: Instrument epic/symbol. Ex. GOLD, GBPUSD
        resolution: Defines the resolution of requested prices
        from_date: Start date in format "YYYY-MM-DDTHH:MM:SS"
        to_date: End date in format "YYYY-MM-DDTHH:MM:SS"
        output_file: Path to the output CSV file
        print_answer: Whether to print individual API responses
        save_raw_data: Save a copy of the raw JSON data (for debugging)
        
    Returns:
        Dictionary containing all the raw price data (with 'prices' key)
    """
    # Fetch the data using our improved function
    logging.info(f"Fetching historical data for {epic} from {from_date} to {to_date}")
    
    # Process in chunks and collect all data
    current_from = datetime.strptime(from_date, "%Y-%m-%dT%H:%M:%S")
    to_datetime = datetime.strptime(to_date, "%Y-%m-%dT%H:%M:%S")
    
    # Calculate optimal chunk size based on resolution
    chunk_size = calculate_optimal_chunk_size(resolution)
    logging.info(f"Using chunk size of {chunk_size} for resolution {resolution}")
    
    # Initialize the collection of all price data
    all_prices = []
    successful_chunks = 0
    failed_chunks = 0
    
    # Process the date range in chunks
    while current_from < to_datetime:
        # Calculate the end date for this chunk
        chunk_end = min(current_from + chunk_size, to_datetime)
        
        logging.info(f"Processing chunk from {current_from.strftime('%Y-%m-%d %H:%M')} to {chunk_end.strftime('%Y-%m-%d %H:%M')}")
        
        # Fetch this chunk with retry logic
        chunk_data, actual_end = fetch_chunk_with_retry(
            X_SECURITY_TOKEN, 
            CST, 
            epic, 
            resolution, 
            current_from, 
            chunk_end,
            print_answer=print_answer
        )
        
        # Process the result
        if "prices" in chunk_data and len(chunk_data["prices"]) > 0:
            all_prices.extend(chunk_data["prices"])
            successful_chunks += 1
            logging.info(f"✓ Successfully fetched {len(chunk_data['prices'])} data points")
        elif "errorCode" in chunk_data:
            failed_chunks += 1
            logging.warning(f"✗ Failed to get data: {chunk_data['errorCode']}")
            
            # If it's still a date range error even after retries, skip this period
            if chunk_data["errorCode"] == "error.invalid.max.daterange":
                logging.warning(f"Skipping period {current_from.strftime('%Y-%m-%d %H:%M')} to {chunk_end.strftime('%Y-%m-%d %H:%M')} - date range too large")
        else:
            failed_chunks += 1
            logging.warning(f"✗ Unexpected response format: {chunk_data}")
        
        # Move to the next chunk (use actual_end in case it was reduced during retries)
        current_from = max(actual_end, current_from + timedelta(minutes=1))  # Ensure we always make progress
    
    logging.info(f"Fetching completed: {successful_chunks} successful chunks, {failed_chunks} failed chunks")
    logging.info(f"Total data points collected: {len(all_prices)}")
    
    # Prepare the complete data object
    complete_data = {
        "prices": all_prices,
        "instrument": epic,
        "resolution": resolution,
        "from": from_date,
        "to": to_date,
        "total_points": len(all_prices),
        "successful_chunks": successful_chunks,
        "failed_chunks": failed_chunks
    }
    
    # Save the raw JSON data if requested
    if save_raw_data:
        raw_json_file = os.path.join(config.CAPCOM_RESPONSE_JSON_DIR, f"response_raw.json")
        with open(raw_json_file, 'w') as f:
            json.dump(complete_data, f, indent=4)
        logging.info(f"Raw data saved to {raw_json_file}")
    
    # Generate output filename
    output_file_name = data_utils.generate_filename(symbol=epic, timeframe=resolution, 
                                               start_date=from_date, end_date=to_date, is_raw=True)
    output_file = os.path.join(config.CAPCOM_RAW_DATA_DIR, output_file_name)
    
    # Save the data to OHLCV CSV format
    if all_prices:
        logging.info(f"Converting {len(all_prices)} data points to OHLCV format")
        convert_json_to_ohlcv_csv(complete_data, output_file)
        logging.info(f"OHLCV data saved to {output_file}")
        
        # Print summary
        first_timestamp = all_prices[0]['snapshotTimeUTC'] if all_prices else "N/A"
        last_timestamp = all_prices[-1]['snapshotTimeUTC'] if all_prices else "N/A"
        logging.info(f"Data range: {first_timestamp} to {last_timestamp}")
    else:
        logging.error("No data to save.")
    
    return complete_data


def fetch_maximum_available_data(X_SECURITY_TOKEN, CST, epic, resolution, 
                                from_date, to_date, output_file=None,
                                print_answer=False, save_raw_data=False):
    """
    Attempts to fetch as much historical data as possible within the given date range.
    If data is not available for certain periods, it will skip them and continue.
    
    This is a wrapper around fetch_and_save_historical_prices with additional
    logic to handle cases where data might not be available for the full requested range.
    """
    try:
        logging.info(f"Attempting to fetch maximum available data for {epic}")
        logging.info(f"Requested range: {from_date} to {to_date}")
        
        result = fetch_and_save_historical_prices(
            X_SECURITY_TOKEN, CST, epic, resolution,
            from_date, to_date, output_file,
            print_answer, save_raw_data
        )
        
        if result["total_points"] > 0:
            logging.info(f"Successfully collected {result['total_points']} data points")
            return result
        else:
            logging.warning("No data was collected for the requested period")
            
            # Try a more recent period if the original request failed completely
            recent_from = datetime.now() - timedelta(days=7)
            recent_from_str = recent_from.strftime("%Y-%m-%dT%H:%M:%S")
            recent_to_str = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            
            logging.info(f"Trying to fetch recent data from {recent_from_str} to {recent_to_str}")
            
            return fetch_and_save_historical_prices(
                X_SECURITY_TOKEN, CST, epic, resolution,
                recent_from_str, recent_to_str, output_file,
                print_answer, save_raw_data
            )
            
    except Exception as e:
        logging.error(f"Error in fetch_maximum_available_data: {e}")
        return {"error": str(e), "prices": [], "total_points": 0}