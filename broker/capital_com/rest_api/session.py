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

import config.config as config


# Global variables
conn = http.client.HTTPSConnection(config.CAPCOM_DEMO_URL)

def start_session(email, password, api_key, use_encryption=True, print_answer=False):
    """
    Method to create a new (trading) session, obtaining session tokens for subsequent API access.
    Session is active for 10 minutes. In case your inactivity is longer than this period then you need to create a new session
    
    Args:
        email: email related to capital.com
        password: password or encrypted password for the selected account (stored in secrets.txt)
        api_key: api_key for the selected account (stored in secrets.txt)
        use_encryption: Boolean
        print_answer: If true, prints response body and headers. Default is False
        
    Returns:
        Path to the trained model run
    """
    logging.info("About to start a new session.")
    if use_encryption:
        logging.info("Using encrypted password to start session.")
        payload = json.dumps({
            "identifier": email,
            "password": password,
            "encryptedPassword": "true"
        })
    else:
        logging.info("Using unecnrypted password to start session!")
        payload = json.dumps({
            "identifier": email,
            "password": password,
        })
    headers = {
        'X-CAP-API-KEY': api_key,
        'Content-Type': 'application/json'
    }

    # Create connection object and send POST request
    conn.request("POST", "/api/v1/session", payload, headers)
    res = conn.getresponse()

    # Read response body
    body = res.read().decode("utf-8")
    # Extract response headers
    response_headers = res.getheaders()  # Returns a list of tuples [(header_name, header_value), ...]
    # Convert headers to a dictionary for easier access
    headers_dict = dict(response_headers)
    # Retrieve specific headers
    x_security_token = headers_dict.get('X-SECURITY-TOKEN')
    cst = headers_dict.get('CST')
    
    if print_answer:
        # Print response body and headers
        logging.info("Response Body:")
        logging.info(body)
        logging.info("\nResponse Headers:")
        for header, value in headers_dict.items():
            logging.info(f"{header}: {value}")

    return body, headers_dict, x_security_token, cst  # Return body and headers if needed

def end_session(X_SECURITY_TOKEN, CST):
    """Method to properly end session with Capital.com's API."""
    payload = ''
    headers = {
        'X-SECURITY-TOKEN': f'{X_SECURITY_TOKEN}',
        'CST': f'{CST}'
    }
    try:
        conn.request("DELETE", "/api/v1/session", payload, headers)
        res = conn.getresponse()
        data = res.read()
        logging.info(f"Successfully ended session: {data.decode('utf-8')}")
    except Exception as e:
        logging.error("Unable to properly end_session(): ", e)

def session_details(X_SECURITY_TOKEN, CST, print_answer):
    """Returns the user's session details."""
    payload = ''
    headers = { 'X-SECURITY-TOKEN': X_SECURITY_TOKEN, 'CST': CST }
    
    conn.request("GET", "/api/v1/session", payload, headers)
    res = conn.getresponse()
    data = res.read()
    if print_answer:
        parsed_data = json.loads(data.decode("utf-8"))
        print(json.dumps(parsed_data, indent=4))
    
def get_encryption_key(api_key):
    """
    Get the encryption key to use in order to send the API key password in an encrypted form.
    
    Args:
        api_key: api_key for the selected account (stored in secrets.txt)
    
    Returns:
        encryption_key and timestamp needed to encrypt the password.
    """
    payload = ''
    headers = {
        'X-CAP-API-KEY': api_key
    }
    conn.request("GET", "/api/v1/session/encryptionKey", payload, headers)
    res = conn.getresponse()
    data = res.read()
    
    # Parse the JSON response
    response_json = json.loads(data.decode("utf-8"))
    
    # Extract the encryption key and timestamp
    encryption_key = response_json["encryptionKey"]
    timestamp = response_json["timeStamp"]
    
    return encryption_key, timestamp

def encrypt_password(password, api_key):
    """
    Using the encryptionKey and timeStamp parameters (recieved from get_encryption_key()) to encrypt API key password using the AES encryption method.
    
    Args:
        password: password or encrypted password for the selected account (stored in secrets.txt)
        api_key: api_key for the selected account (stored in secrets.txt)
        
    Returns:
        Encrypted password
    """
    encryption_key, timestamp = get_encryption_key(api_key)
    try:
        # Concatenate password and timestamp with separator
        input_data = (password + "|" + str(timestamp)).encode('utf-8')
        
        # Base64 encode the input
        input_data = base64.b64encode(input_data)
        
        # Decode the encryption key from base64 and create public key
        key_bytes = base64.b64decode(encryption_key.encode('utf-8'))
        public_key = serialization.load_der_public_key(
            key_bytes,
            backend=default_backend()
        )
        
        # Encrypt using RSA with PKCS#1 padding
        output = public_key.encrypt(
            input_data,
            padding.PKCS1v15()
        )
        
        # Base64 encode the result
        output = base64.b64encode(output)
        
        # Return as string
        return output.decode('utf-8')
    except Exception as e:
        raise RuntimeError(e)

def switch_active_account(account_id, account_name, X_SECURITY_TOKEN, CST, print_answer):
    """Switch active account based on account ID."""
    payload = json.dumps({
        "accountId": account_id
    })
    headers = {
        'X-SECURITY-TOKEN': X_SECURITY_TOKEN,
        'CST': CST,
        'Content-Type': 'application/json'
    }
    conn.request("PUT", "/api/v1/session", payload, headers)
    res = conn.getresponse()
    data = res.read()
    if print_answer:
        parsed_data = json.loads(data.decode("utf-8"))
        print(json.dumps(parsed_data, indent=4))
    logging.info(f"Successfully switched to account {account_name}!")