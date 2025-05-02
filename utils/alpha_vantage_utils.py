import csv
import json
import logging
import os

import config.config as config
import utils.shared_utils as shared_util

import csv
import json
import logging
import os

def convert_market_data_to_csv(json_data, output_file_name:str, output_dir:str = None):
    """Convert response from Alpha Vantage API to .csv-file format."""
    # If json_data is a string, parse it to a dictionary
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data
        
    if output_dir is None:
        output_dir = config.ALPVAN_RAW_DATA_DIR # Default directory if not specified
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)    
    
    output = os.path.join(output_dir, output_file_name)
    
    # Open the CSV file for writing
    with open(output, 'w', newline='') as csvfile:
        # Create CSV writer
        fieldnames = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write the header
        writer.writeheader()
        
        # Dynamically find the time series key
        time_series_key = None
        for key in data.keys():
            if key.startswith("Time Series"):
                time_series_key = key
                break
        
        if not time_series_key:
            logging.error("Could not find time series data in the JSON")
            return None
        
        # Process each time entry in the Time Series data
        time_series = data.get(time_series_key, {})
        
        for timestamp, price_data in time_series.items():
            # Extract prices and volume from the current time entry
            open_price = float(price_data.get('1. open', 0))
            high_price = float(price_data.get('2. high', 0))
            low_price = float(price_data.get('3. low', 0))
            close_price = float(price_data.get('4. close', 0))
            volume = int(price_data.get('5. volume', 0))
            
            # Write the row to the CSV
            writer.writerow({
                'Date': timestamp,
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close': close_price,
                'Volume': volume
            })
    
    logging.info(f"Data successfully converted and saved to {output}")
    return output