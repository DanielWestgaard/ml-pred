import csv
import os
from datetime import datetime
import sys
import json

import config.config as config
from utils.logging_utils import get_logger

logger = get_logger()


########## HANDLING ##########

def on_open(ws, subscription_message):
    logger.info("WebSocket connection opened")
    # Send the subscription message
    ws.send(json.dumps(subscription_message))

def on_message(ws, message):
    parsed = json.dumps(message, indent=4)
    logger.debug(f"WebSocket message received: {message}")

def on_message_improved(ws, message):
    try:
        # Parse the JSON message
        data = json.loads(message)
        
        # Print the formatted JSON with indentation for readability
        formatted_json = json.dumps(data, indent=4)
        
        # Log the formatted JSON with visual separation
        logger.debug(f"\n{'-'*50}\n{formatted_json}\n{'-'*50}")
    except Exception as e:
        logger.error(f"Error processing WebSocket message: {e}")
        logger.debug(f"Original message: {message}")

def on_message_pretty(ws, message):
    try:
        # Parse the JSON message
        data = json.loads(message)
        
        # Check if it's an OHLC event
        if data.get("destination") == "ohlc.event":
            # Extract the payload
            payload = data.get("payload", {})
            
            # Convert timestamp to readable datetime
            timestamp_ms = payload.get("t")
            if timestamp_ms:
                timestamp = datetime.datetime.fromtimestamp(timestamp_ms / 1000)
                formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            else:
                formatted_time = "N/A"
            
            # Create a formatted output
            ohlc_info = (f"\n{'='*50}\n"
                        f"OHLC Update for {payload.get('epic')}\n"
                        f"Time: {formatted_time}\n"
                        f"Resolution: {payload.get('resolution')}\n"
                        f"Type: {payload.get('type')}\n"
                        f"Price Type: {payload.get('priceType')}\n"
                        f"{'-'*20}\n"
                        f"Open:  {payload.get('o')}\n"
                        f"High:  {payload.get('h')}\n"
                        f"Low:   {payload.get('l')}\n"
                        f"Close: {payload.get('c')}\n"
                        f"{'='*50}")
            logger.info(ohlc_info)
        else:
            # For other types of messages, just log them nicely formatted
            logger.debug(f"\nReceived message:\n{json.dumps(data, indent=4)}")
    except Exception as e:
        logger.error(f"Error processing WebSocket message: {e}")
        logger.debug(f"Original message: {message}")

def on_error(ws, error):
    logger.error(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    logger.info(f"WebSocket connection closed - Status: {close_status_code}, Message: {close_msg}")


########## API ##########

def get_account_id_by_name(json_data : json, account_name):
    """ Retrieves the account ID based on the account name. """
    # Parse the JSON string to a Python dictionary  
    if json_data is None:
        logger.error("json_data is None - API call may have failed")
        return None, None
        
    try:
        if isinstance(json_data, str):
            parsed_data = json.loads(json_data)
        elif isinstance(json_data, dict):
            # Already parsed JSON
            parsed_data = json_data
        else:
            logger.error(f"Unexpected data type for json_data: {type(json_data)}, content: {json_data}")
            return None, None
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"Failed to parse JSON data: {e}. Data type: {type(json_data)}, Data: {json_data}")
        return None, None
    
    # Iterate through accounts to find the matching accountName
    for account in parsed_data["accounts"]:
        if account["accountName"] == account_name:
            logger.info(f"Found accountId matching account {account_name}!")
            return account["accountId"], account_name
    
    logger.warning(f"Did not find accountId that matched account name {account_name}")
    # Return None if no match is found
    return None, None

def extract_deal_reference(json_data, key_string_to_extract):
    """ Extracting and returning the deal reference / dealID from confirmed positions. """
    # Parse the JSON string to a Python dictionary
    parsed_data = json.loads(json_data)
    # Extract the dealReference value
    value = parsed_data.get(key_string_to_extract)  # "dealReference"
    #logger.info(f"Successfully extracted {key_string_to_extract}: {value}")
    
    return value

def process_positions(json_response):
    """
    Process a trading positions JSON response.
    
    Args:
        json_response (str or dict): JSON string or dictionary containing positions data
    
    Returns:
        list: List of all deal IDs
    """
    # Parse the JSON if it's a string
    if isinstance(json_response, str):
        data = json.loads(json_response)
    else:
        data = json_response
    
    # Get the positions array
    positions = data.get('positions', [])
    
    # Count active positions
    position_count = len(positions)
    logger.info(f"Number of active positions: {position_count}")
    
    # Extract all deal IDs
    deal_ids = []
    for position in positions:
        if 'position' in position and 'dealId' in position['position']:
            deal_ids.append(position['position']['dealId'])
    
    return deal_ids

# Not sure where to place this method. Was thinking in the data processing pipeline, 
# but could also be more of a utility func for when taking in live market data.
def convert_json_to_ohlcv_csv(json_data, output_file):
    """
    Converts the provided JSON price data to OHLCV format and saves it to a CSV file.
    
    Args:
        json_data (dict or str): The JSON data containing price information
        output_file (str): Path to the output CSV file
    """
    # If json_data is a string, parse it to a dictionary
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data
    
    # Open the CSV file for writing
    with open(output_file, 'w', newline='') as csvfile:
        # Create CSV writer
        fieldnames = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write the header
        writer.writeheader()
        
        # Process each price entry
        for price in data['prices']:
            # Extract UTC timestamp (using snapshotTimeUTC for consistency)
            timestamp = price['snapshotTimeUTC']
            
            # Calculate mid prices (average of bid and ask) with rounding
            decimals = config.PRICE_DECIMALS  # Adjust based on the asset's typical price granularity
            open_price = round((price['openPrice']['bid'] + price['openPrice']['ask']) / 2, decimals)
            high_price = round((price['highPrice']['bid'] + price['highPrice']['ask']) / 2, decimals)
            low_price = round((price['lowPrice']['bid'] + price['lowPrice']['ask']) / 2, decimals)
            close_price = round((price['closePrice']['bid'] + price['closePrice']['ask']) / 2, decimals)
            volume = price['lastTradedVolume']
            
            # Write the row to the CSV
            writer.writerow({
                'Date': timestamp,
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close': close_price,
                'Volume': volume
            })
    
    logger.info(f"Data successfully converted and saved to {output_file}")

def get_capital_from_json(json_data : str, account_name):
    """
    Method to extrect the capital of the active account.
    """
    try:
        parsed_data = json.loads(json_data)
        for account in parsed_data["accounts"]:
            if account["accountName"] == account_name:
                available_balance = account["balance"]["available"]
                logger.info(f"Successfully found available account balance USD {available_balance} to account {account_name}!")
                return available_balance
    except:
        logger.error("Unable to find available balance from provided json data! Returning static capital 1000.")
        return 1000