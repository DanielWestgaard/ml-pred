import http.client
import json
import os
import sys
import time
import base64
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

import config.config as config
from utils.logging_utils import get_logger

logger = get_logger()


# Note: Using connection-per-request instead of global connection to avoid state conflicts

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
    logger.info("About to start a new session.")
    if use_encryption:
        logger.debug("Using encrypted password to start session.")
        payload = json.dumps({
            "identifier": email,
            "password": password,
            "encryptedPassword": "true"
        })
    else:
        logger.warning("Using unecnrypted password to start session!")
        payload = json.dumps({
            "identifier": email,
            "password": password,
        })
    headers = {
        'X-CAP-API-KEY': api_key,
        'Content-Type': 'application/json'
    }

    # Create a fresh connection for this request to avoid state conflicts
    conn = http.client.HTTPSConnection(config.CAPCOM_DEMO_URL)
    
    try:
        # Send POST request with retry logic for "Request-started" errors
        max_retries = 3
        for attempt in range(max_retries):
            try:
                conn.request("POST", "/api/v1/session", payload, headers)
                res = conn.getresponse()
                body = res.read().decode("utf-8")
                
                # Check for "Request-started" error
                if "Request-started" in body:
                    logger.warning(f"Capital.com 'Request-started' error on attempt {attempt + 1}, retrying...")
                    if attempt < max_retries - 1:
                        time.sleep(1 + attempt)  # Exponential backoff
                        # Create new connection for retry
                        conn.close()
                        conn = http.client.HTTPSConnection(config.CAPCOM_DEMO_URL)
                        continue
                    else:
                        logger.error("Max retries exceeded for 'Request-started' error")
                        raise RuntimeError("Unable to start session: Request-started error persists")
                
                # Success - extract response headers
                response_headers = res.getheaders()
                headers_dict = dict(response_headers)
                x_security_token = headers_dict.get('X-SECURITY-TOKEN')
                cst = headers_dict.get('CST')
                
                if print_answer:
                    logger.info("Response Body:")
                    logger.info(body)
                    logger.info("\nResponse Headers:")
                    for header, value in headers_dict.items():
                        logger.info(f"{header}: {value}")

                return body, headers_dict, x_security_token, cst
                
            except Exception as e:
                logger.error(f"Session start attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1 + attempt)
                    # Create new connection for retry
                    conn.close()
                    conn = http.client.HTTPSConnection(config.CAPCOM_DEMO_URL)
                else:
                    raise
                    
    finally:
        # Close the connection after the request is complete
        # The session tokens remain valid for 10 minutes regardless of connection state
        try:
            conn.close()
        except:
            pass

def end_session(X_SECURITY_TOKEN, CST):
    """Method to properly end session with Capital.com's API."""
    # Create a fresh connection for ending session
    conn = http.client.HTTPSConnection(config.CAPCOM_DEMO_URL)
    
    try:
        payload = ''
        headers = {
            'X-SECURITY-TOKEN': f'{X_SECURITY_TOKEN}',
            'CST': f'{CST}'
        }
        
        conn.request("DELETE", "/api/v1/session", payload, headers)
        res = conn.getresponse()
        data = res.read()
        logger.info(f"Successfully ended session: {data.decode('utf-8')}")
        
    except Exception as e:
        logger.error(f"Unable to properly end_session(): {e}")
        
    finally:
        # Always close the connection
        try:
            conn.close()
        except:
            pass

def session_details(X_SECURITY_TOKEN, CST, print_answer):
    """Returns the user's session details."""
    # Create a fresh connection for session details
    conn = http.client.HTTPSConnection(config.CAPCOM_DEMO_URL)
    
    try:
        payload = ''
        headers = { 'X-SECURITY-TOKEN': X_SECURITY_TOKEN, 'CST': CST }
        
        conn.request("GET", "/api/v1/session", payload, headers)
        res = conn.getresponse()
        data = res.read()
        if print_answer:
            parsed_data = json.loads(data.decode("utf-8"))
            logger.debug(f"Session response: {json.dumps(parsed_data, indent=4)}")
            
    finally:
        # Always close the connection
        try:
            conn.close()
        except:
            pass
    
def get_encryption_key(api_key):
    """
    Get the encryption key to use in order to send the API key password in an encrypted form.
    
    Args:
        api_key: api_key for the selected account (stored in secrets.txt)
    
    Returns:
        encryption_key and timestamp needed to encrypt the password.
    """
    # Create a fresh connection for encryption key
    conn = http.client.HTTPSConnection(config.CAPCOM_DEMO_URL)
    
    try:
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
        
    finally:
        # Always close the connection
        try:
            conn.close()
        except:
            pass

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
    # Create a fresh connection for account switching
    conn = http.client.HTTPSConnection(config.CAPCOM_DEMO_URL)
    
    try:
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
            logger.debug(f"Session response: {json.dumps(parsed_data, indent=4)}")
        logger.info(f"Successfully switched to account {account_name}!")
        
    finally:
        # Always close the connection
        try:
            conn.close()
        except:
            pass