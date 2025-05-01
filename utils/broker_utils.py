import csv
import os
import logging
from datetime import datetime
import sys
import json

import config.constants.system_config as sys_config
import config.constants.market_config as mark_config
import broker.capital_com.capitalcom as broker


def load_secrets_alpaca(file_path="secrets/secrets.txt"):
    desired_keys={"alpaca_secret_key_paper", "alpaca_api_key_paper"}
    secrets = {}
    
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if "=" in line:  # Ensure valid format
                key, value = line.split("=", 1)  # Split at the first '='
                if key in desired_keys:
                    secrets[key] = value
    
    api_key = secrets.get('alpaca_api_key_paper')
    secret_key = secrets.get('alpaca_secret_key_paper')

    return secrets, api_key, secret_key

def load_secrets(file_path="secrets/secrets.txt"):
    """
    Utility functions to extract the following keys: API_KEY_CAP, PASSWORD_CAP, EMAIL.
    
    Returns:
        secrets{}, API_KEY_CAP, PASSWORD_CAP, EMAIL
    """
    desired_keys = {"API_KEY_CAP", "PASSWORD_CAP", "EMAIL"}
    secrets = {}
    
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if "=" in line:  # Ensure valid format
                key, value = line.split("=", 1)  # Split at the first '='
                if key in desired_keys:
                    secrets[key] = value
    
    api_key = secrets.get('API_KEY_CAP')
    password = secrets.get('PASSWORD_CAP')
    email = secrets.get('EMAIL')
    logging.info(f"Loaded credentials. Length: {len(secrets)}")

    return secrets, api_key, password, email


# Related to handling

def on_open(ws, subscription_message):
    print("Connection opened")
    # Send the subscription message
    ws.send(json.dumps(subscription_message))

def on_message(ws, message):
    parsed = json.dumps(message, indent=4)
    print(f"Received: {message}")

def on_message_improved(ws, message):
    try:
        # Parse the JSON message
        data = json.loads(message)
        
        # Print the formatted JSON with indentation for readability
        formatted_json = json.dumps(data, indent=4)
        
        # Add a separator line for visual clarity between messages
        print("\n" + "-"*50)
        print(formatted_json)
        print("-"*50)
    except Exception as e:
        print(f"Error processing message: {e}")
        print(f"Original message: {message}")

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
            print("\n" + "="*50)
            print(f"OHLC Update for {payload.get('epic')}")
            print(f"Time: {formatted_time}")
            print(f"Resolution: {payload.get('resolution')}")
            print(f"Type: {payload.get('type')}")
            print(f"Price Type: {payload.get('priceType')}")
            print("-"*20)
            print(f"Open:  {payload.get('o')}")
            print(f"High:  {payload.get('h')}")
            print(f"Low:   {payload.get('l')}")
            print(f"Close: {payload.get('c')}")
            print("="*50)
        else:
            # For other types of messages, just print them nicely formatted
            print(f"\nReceived message:\n{json.dumps(data, indent=4)}")
    except Exception as e:
        print(f"Error processing message: {e}")
        print(f"Original message: {message}")

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("Connection closed")


# Related to API

def get_account_id_by_name(json_data : json, account_name):
    """ Retrieves the account ID based on the account name. """
    # Parse the JSON string to a Python dictionary
    parsed_data = json.loads(json_data)
    
    # Iterate through accounts to find the matching accountName
    for account in parsed_data["accounts"]:
        if account["accountName"] == account_name:
            logging.info(f"Found accountId matching account {account_name}!")
            return account["accountId"], account_name
    
    logging.warning(f"Did not find accountId that matched account name {account_name}")
    # Return None if no match is found
    return None, None

def extract_deal_reference(json_data, key_string_to_extract):
    """ Extracting and returning the deal reference / dealID from confirmed positions. """
    # Parse the JSON string to a Python dictionary
    parsed_data = json.loads(json_data)
    # Extract the dealReference value
    value = parsed_data.get(key_string_to_extract)  # "dealReference"
    #logging.info(f"Successfully extracted {key_string_to_extract}: {value}")
    
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
    print(f"Number of active positions: {position_count}")
    
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
            decimals = mark_config.PRICE_DECIMALS  # Adjust based on the asset's typical price granularity
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
    
    logging.info(f"Data successfully converted and saved to {output_file}")

def get_capital_from_json(json_data : str, account_name):
    """
    Method to extrect the capital of the active account.
    """
    try:
        parsed_data = json.loads(json_data)
        for account in parsed_data["accounts"]:
            if account["accountName"] == account_name:
                available_balance = account["balance"]["available"]
                logging.info(f"Successfully found available account balance USD {available_balance} to account {account_name}!")
                return available_balance
    except:
        logging.error("Unable to find available balance from provided json data! Returning static capital 1000.")
        return 1000