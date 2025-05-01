import http.client
import json
import os
import sys
import logging
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config.config as config

# Global variables
conn = http.client.HTTPSConnection(config.CAPCOM_DEMO_URL)

def list_all_accounts(X_SECURITY_TOKEN, CST, print_answer):
    """Returns a list of accounts belonging to the logged-in client."""
    payload = ''
    headers = {
        'X-SECURITY-TOKEN': X_SECURITY_TOKEN,
        'CST': CST
    }
    conn.request("GET", "/api/v1/accounts", payload, headers)
    res = conn.getresponse()
    data = res.read()
    if print_answer:
        parsed_data = json.loads(data.decode("utf-8"))
        print(json.dumps(parsed_data, indent=4))
    else:
        logging.info("Will not print out response of all accounts.")
    return data.decode("utf-8")