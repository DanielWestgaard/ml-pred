import http.client
import os
import sys
import logging
import time
import base64
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
import json

from utils.broker_utils import extract_deal_reference
import config.config as config


conn = http.client.HTTPSConnection(config.CAPCOM_DEMO_URL)

# POSITIONS
def all_positions(X_SECURITY_TOKEN, CST, print_answer):
    """Returns all open positions for the active account"""
    payload = ''
    headers = {
        'X-SECURITY-TOKEN': X_SECURITY_TOKEN,
        'CST': CST,
    }
    conn.request("GET", "/api/v1/positions", payload, headers)
    res = conn.getresponse()
    data = res.read()
    if print_answer:
        parsed_data = json.loads(data.decode("utf-8"))
        print(json.dumps(parsed_data, indent=4))
    return data.decode("utf-8")

def create_new_position(X_SECURITY_TOKEN, CST, 
                     symbol, direction, size, stop_amount=None, profit_amount=None, stop_level=None, profit_level=None,
                     print_answer=True):
    """
    Create orders and positions.
    Note: The deal reference you get as "confirmation" from successfully creating a new position
    is not the same dealReference the order has (when active) and not the same as dealId.
    
    Args:
        X_SECURITY_TOKEN: Account token or account id identifying the client's current account/session
        CST: Access token identifying the client
        symbol: Instrument epic identifier. Ex. SILVER
        direction: Deal direction. Must be BUY or SELL
        size: Deal size. Ex. 1
        stop_amount: Loss amount when a stop loss will be triggered (dollars). Ex. 4
        profit_amount: Profit amount when a take profit will be triggered (dollars). Ex. 20
        print_answer: If true, prints response body and headers. Default is False
    
    Return:
        Deal Reference / deal ID
    """
    if stop_amount is not None and profit_amount is not None:  # have added dollar amount of SL/TP
        logging.info("Recieved request of using dollar amout as SL/TP.")
        payload = json.dumps({
            "epic": symbol,
            "direction": direction,
            "size": size,
            "guaranteedStop": False,
            "stopAmount": stop_amount, 
            "profitAmount": profit_amount
        })
    elif stop_level is not None and profit_level is not None:  # have added price level/value as SL/TP
        logging.info("Recieved request of using price level as SL/TP.")
        payload = json.dumps({
            "epic": symbol,
            "direction": direction,
            "size": size,
            "guaranteedStop": False,
            "stopLevel": stop_level,
            "profitLevel": profit_level
        })
    else:  # Not having any stop loss or take profit
        logging.warning("Did not recieve SL/TP values!")
        payload = json.dumps({
            "epic": symbol,
            "direction": direction,
            "size": size,
            "guaranteedStop": False,
        })
        
    headers = {
        'X-SECURITY-TOKEN': X_SECURITY_TOKEN,
        'CST': CST,
        'Content-Type': 'application/json'
    }
    conn.request("POST", "/api/v1/positions", payload, headers)
    res = conn.getresponse()
    data = res.read()
    if print_answer:
        parsed_data = json.loads(data.decode("utf-8"))
        print(json.dumps(parsed_data, indent=4))
    
    deal_ref = extract_deal_reference(data.decode("utf-8"), "dealReference")
    # looks like the response deal reference value from placing a successfull trade is not the one related to the actual position
    logging.info(print(f"Deal Reference from confirmed position: {deal_ref}"))
    
    return deal_ref
    
def close_position(X_SECURITY_TOKEN, CST, dealID, print_answer):
    """
    Close the position of trade related with dealId for a confirmed trade.
    Note: The deal reference you get as "confirmation" from successfully creating a new position
    is not the same dealReference the order has (when active) and not the same as dealId.
    
    Args:
        X_SECURITY_TOKEN: Account token or account id identifying the client's current account/session
        CST: Access token identifying the client
        dealID: Permanent deal reference for a confirmed trade
        print_answer: If true, prints response body and headers. Default is False
    
    Return:
        Response from closed position.
    """
    payload = ''
    headers = {
        'X-SECURITY-TOKEN': X_SECURITY_TOKEN,
        'CST': CST,
    }
    print(f"I have dealid : /api/v1/positions/{dealID}")
    conn.request("DELETE", f"/api/v1/positions/{dealID}", payload, headers)
    res = conn.getresponse()
    data = res.read()
    if print_answer:
        parsed_data = json.loads(data.decode("utf-8"))
        print(json.dumps(parsed_data, indent=4))
    return data.decode("utf-8")

def update_position(X_SECURITY_TOKEN, CST, dealID, print_answer,
                    stop_amount=None, profit_amount=None, stop_level=None, profit_level=None,):
    if stop_amount is not None and profit_amount is not None:  # have added dollar amount of SL/TP
        logging.info("Recieved request of using dollar amout as SL/TP.")
        payload = json.dumps({
            "guaranteedStop": False,
            "stopAmount": stop_amount,
            "profitAmount": profit_amount
        })
    elif stop_level is not None and profit_level is not None:  # have added price level/value as SL/TP
        logging.info("Recieved request of using price level as SL/TP.")
        payload = json.dumps({
            "guaranteedStop": False,
            "stopLevel": stop_level,
            "profitLevel": profit_level
        })
    else:  # Not having any stop loss or take profit
        logging.error("Did not recieve SL/TP values! What is there to change then?")
        return None
    
    headers = {
        'X-SECURITY-TOKEN': X_SECURITY_TOKEN,
        'CST': CST,
        'Content-Type': 'application/json'
    }
    conn.request("PUT", f"/api/v1/positions/{dealID}", payload, headers)
    res = conn.getresponse()
    data = res.read()
    if print_answer:
        parsed_data = json.loads(data.decode("utf-8"))
        print(json.dumps(parsed_data, indent=4))
    return data.decode("utf-8")

# ORDERS - not really needed for current scope?
def all_orders(X_SECURITY_TOKEN, CST, print_answer=True):
    """ Returns all open working orders for the active account. """
    payload = ''
    headers = {
        'X-SECURITY-TOKEN': X_SECURITY_TOKEN,
        'CST': CST,
    }
    conn.request("GET", "/api/v1/workingorders", payload, headers)
    res = conn.getresponse()
    data = res.read()
    if print_answer:
        parsed_data = json.loads(data.decode("utf-8"))
        print(json.dumps(parsed_data, indent=4))
    return data.decode("utf-8")

def create_new_order(X_SECURITY_TOKEN, CST, print_answer=True):
    pass

def delete_order(X_SECURITY_TOKEN, CST, print_answer=True):
    pass
