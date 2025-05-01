"""
According to Capital.com's documentation, websocket is the way one can subscribe to live market data.
"""

import logging
import http
import json
import os
import sys
import threading
import time
import websocket
from functools import partial

from utils.broker_utils import on_close, on_error, on_message, on_open, on_message_improved


# Global variables
ws_url = "wss://api-streaming-capital.backend-capital.com/connect"

def sub_live_market_data(X_SECURITY_TOKEN, CST, epic, resolution, custom_message_handler=None):
    """
    Subscribe to the candlestick bars updates by mentioning the epics, resolutions, bar type, and a custom message handler.
    
    Args:
        X_SECURITY_TOKEN: Security token
        CST: CST value
        epic: Trading instrument/epic
        resolution: Data resolution
        custom_message_handler: Custom function to handle incoming messages
        
    Returns:
        WebSocket connection object
    """
    # Subscription payload
    subscription_message = {
        "destination": "OHLCMarketData.subscribe",
        "correlationId": "3",
        "cst": CST,
        "securityToken": X_SECURITY_TOKEN,
        "payload": {
            "epics": [
                epic
            ],
            "resolutions": [
                resolution
            ],
            "type": "classic"
        }
    }
    
    # This uses functools.partial to create a new function with the parameter pre-filled
    custom_on_open = partial(on_open, subscription_message=subscription_message)
    # Determine which message handler to use
    message_handler = custom_message_handler if custom_message_handler else on_message_improved
    
    # Create a WebSocket connection
    ws = websocket.WebSocketApp(
        ws_url,
        header={
            "CST": CST,
            "X-SECURITY-TOKEN": X_SECURITY_TOKEN
        },
        on_open=custom_on_open,
        on_message=message_handler,
        on_error=on_error,
        on_close=on_close
    )

    # Start the WebSocket connection in a separate thread
    ws_thread = threading.Thread(target=ws.run_forever)
    ws_thread.daemon = True  # Allow the thread to exit when the main program exits
    ws_thread.start()
    
    # Give the WebSocket time to connect
    time.sleep(1)
    
    return ws

def unsub_live_market_data(X_SECURITY_TOKEN, CST, epic, resolution, ws):
    """
    Unsubscribe from candlestick bars updates for specific epics, resolutions and bar types.
    """
    # Unsubscription payload
    unsubscription_message = {
        "destination": "OHLCMarketData.unsubscribe",
        "correlationId": "4",
        "cst": CST,
        "securityToken": X_SECURITY_TOKEN,
        "payload": {
            "epics": epic,
            "resolutions": resolution,
            "types": "classic"
        }
    }
    
    # Send the unsubscription message if the WebSocket is open
    if ws.sock and ws.sock.connected:
        ws.send(json.dumps(unsubscription_message))
        print(f"Sent unsubscription request for {', '.join(epic)}")
    else:
        print("WebSocket is not connected. Cannot send unsubscribe request.")
        
def send_ping(ws, X_SECURITY_TOKEN, CST):
    """
    Send a ping message to keep the WebSocket connection alive.
    """
    ping_message = {
        "destination": "ping",
        "correlationId": str(int(time.time())),  # Use timestamp as correlation ID
        "cst": CST,
        "securityToken": X_SECURITY_TOKEN
    }
    
    if ws.sock and ws.sock.connected:
        ws.send(json.dumps(ping_message))
        logging.debug(f"Ping sent at {time.strftime('%H:%M:%S')}")
        return True
    else:
        logging.error("WebSocket is not connected. Cannot send ping.")
        return False

def setup_ping_timer(ws, X_SECURITY_TOKEN, CST, interval_seconds=120):
    """
    Set up a timer to ping the server every 2 minutes
    """
    def ping_task():
        if send_ping(ws, X_SECURITY_TOKEN, CST):
            # Schedule the next ping only if the current one was successful
            threading.Timer(interval_seconds, ping_task).start()
    
    # Start the first ping task
    threading.Timer(interval_seconds, ping_task).start()
    logging.info(f"Automatic ping scheduled every {interval_seconds} seconds")
