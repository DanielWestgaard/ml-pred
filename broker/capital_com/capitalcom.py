from abc import ABC
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
from datetime import datetime

from broker.base_interface import BaseBroker
from utils import shared_utils
import utils.broker_utils as br_util
import config.config as config
import broker.capital_com.rest_api.account as account
import broker.capital_com.rest_api.session as session
import broker.capital_com.rest_api.trading as trading
import broker.capital_com.rest_api.markets_info as markets_info
import broker.capital_com.web_socket.web_socket as web_socket


class CapitalCom(BaseBroker):
    """Used to interact with implemented API endpoints."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the broker with configuration. Will always try to encrypt the password first,
        but if that fails it will try to fall back to plain password.
        """
        
        try: 
            # Getting secrets
            self.secrets = shared_utils.load_secrets(desired_keys={'API_KEY_CAP', 'PASSWORD_CAP', 'EMAIL'})
            self.api_key = self.secrets.get('API_KEY_CAP')
            self.password = self.secrets.get('PASSWORD_CAP')
            self.email = self.secrets.get('EMAIL')
            try:
                # Encrypting password
                self.enc_pass = session.encrypt_password(self.password, self.api_key)
            except Exception as e:
                logging.warning(f"Could not initiate BaseBroker with encrypted password. Using plain password. Error: {e}")
        except Exception as e:
            logging.warning(f"Could not initiate BaseBroker with secrets. Error: {e}")
            
    # =================== SESSION METHODS ==================
    
    def start_session(self, email=None, password=None, api_key=None, use_encryption=True, print_answer=False):
        """Starting session with the broker."""
        self.body, self.headers_dict, self.x_security_token, self.cst = session.start_session(email=email or self.email,
                                     password=password or self.enc_pass,
                                     api_key=api_key or self.api_key,
                                     use_encryption=use_encryption,
                                     print_answer=print_answer)
        return self.body, self.headers_dict, self.x_security_token, self.cst  # Just in case I need to use them outside
        
    def end_session(self, X_SECURITY_TOKEN=None, CST=None):
        """End the session with the broker."""
        return session.end_session(X_SECURITY_TOKEN=X_SECURITY_TOKEN or self.x_security_token, CST=CST or self.cst)
    
    def session_details(self, X_SECURITY_TOKEN=None, CST=None, print_answer=False):
        return session.session_details(X_SECURITY_TOKEN=X_SECURITY_TOKEN or self.x_security_token, CST=CST or self.cst, print_answer=print_answer)
    
    def switch_active_account(self, account_id=None, account_name=None, X_SECURITY_TOKEN=None, CST=None, print_answer=False):
        if account_id is None or self.all_accounts is None:
            logging.info("AccountID and/or all_accounts is None. Initializing them now.")
            self.all_accounts = account.list_all_accounts(X_SECURITY_TOKEN=X_SECURITY_TOKEN or self.x_security_token, CST=CST or self.cst, print_answer=print_answer)
            self.account_id, self.account_name = br_util.get_account_id_by_name(self.all_accounts, account_name=account_name or config.VAR_ACCOUNT_NAME_TEST)
        if self.account_id is None and self.account_name is None:
            logging.error("Error switching active account! Unable to switch!")
            return None
        return session.switch_active_account(self.account_id, self.account_name, X_SECURITY_TOKEN=X_SECURITY_TOKEN or self.x_security_token, CST=CST or self.cst, print_answer=print_answer)
            
    # ==================== DATA METHODS ====================
    
    def get_historical_data(self, epic:str, resolution:str, 
                            from_date:str, to_date:str,  # format 2022-02-24T00:00:00
                            X_SECURITY_TOKEN=None, CST=None,
                            max=1000, print_answer=False):
        """Fetch historical OHLCV data."""
        return markets_info.historical_prices(X_SECURITY_TOKEN=X_SECURITY_TOKEN or self.x_security_token, CST=CST or self.cst,
                                              epic=epic, resolution=resolution, from_date=from_date, to_date=to_date,
                                              max=max, print_answer=print_answer)
    
    def fetch_and_save_historical_prices(self, epic:str, resolution:str, 
                                    from_date:str, to_date:str,  # format 2022-02-24T00:00:00
                                    output_file=None,
                                    X_SECURITY_TOKEN=None, CST=None,
                                    print_answer=False, save_raw_data=False):
        return markets_info.fetch_and_save_historical_prices(X_SECURITY_TOKEN=X_SECURITY_TOKEN or self.x_security_token, CST=CST or self.cst,
                                                             epic=epic, resolution=resolution, from_date=from_date, to_date=to_date,
                                                             output_file=output_file, print_answer=print_answer, save_raw_data=save_raw_data)
    
    def fetch_maximum_available_data(self, epic:str, resolution:str, 
                                    from_date:str, to_date:str,  # format 2022-02-24T00:00:00
                                    output_file=None,
                                    X_SECURITY_TOKEN=None, CST=None,
                                    print_answer=False, save_raw_data=False):
        """
        Fetch as much historical data as possible within the given date range.
        Uses improved chunking and error handling to get maximum available data.
        """
        return markets_info.fetch_maximum_available_data(
            X_SECURITY_TOKEN=X_SECURITY_TOKEN or self.x_security_token, 
            CST=CST or self.cst,
            epic=epic, 
            resolution=resolution, 
            from_date=from_date, 
            to_date=to_date,
            output_file=output_file, 
            print_answer=print_answer, 
            save_raw_data=save_raw_data
        )
    
    # ==================== ACCOUNT METHODS ====================
    
    def list_all_accounts(self, X_SECURITY_TOKEN=None, CST=None, print_answer=False):
        """Returns a list of accounts belonging to the logged-in client."""
        self.all_accounts = account.list_all_accounts(X_SECURITY_TOKEN=X_SECURITY_TOKEN or self.x_security_token, CST=CST or self.cst, print_answer=print_answer)
        return self.all_accounts
    
    def get_account_capital(self):
        """
        Method to extrect the capital of the active account.
        Important! Assuming that the code already has switched to active account USD_testing!!
        """
        try:
            if self.all_accounts is not None:  # Will throw error if now initialized
                return br_util.get_capital_from_json(json_data=self.all_accounts, account_name=self.account_name)
        except Exception as e:
            logging.error("Unable to fetch account capital! Must switch active account first! ")
            return None
    
    def get_positions(self) -> List[Dict]:
        """Get current open positions."""
        logging.info("Getting positions")
        return []  # Return empty list
    
    def get_orders(self) -> List[Dict]:
        """Get current pending orders."""
        logging.info("Getting orders")
        return []  # Return empty list
    
    # ==================== TRADING METHODS ====================
    
    def all_positions(self, X_SECURITY_TOKEN=None, CST=None, print_answer=True):
        """Returns all open positions for the active account."""
        return trading.all_positions(X_SECURITY_TOKEN=X_SECURITY_TOKEN or self.x_security_token, CST=CST or self.cst,print_answer=print_answer)
    
    def place_market_order(self, symbol, direction, size, stop_amount=None, profit_amount=None, stop_level=None, profit_level=None,
                            X_SECURITY_TOKEN=None, CST=None,
                            print_answer=True):
        """
        Create orders and positions.
        Note: The deal reference you get as "confirmation" from successfully creating a new position
        is not the same dealReference the order has (when active) and not the same as dealId.
        
        Args:
            symbol: Instrument epic identifier. Ex. SILVER
            direction: Deal direction. Must be BUY or SELL
            size: Deal size. Ex. 1
            stop_amount: Loss amount when a stop loss will be triggered. Ex. 4
            profit_amount: Profit amount when a take profit will be triggered. Ex. 20
            print_answer: If true, prints response body and headers. Default is False
        
        Return:
            Deal Reference / deal ID
        """
        return trading.create_new_position(X_SECURITY_TOKEN=X_SECURITY_TOKEN or self.x_security_token, CST=CST or self.cst, print_answer=print_answer,
                                        symbol=symbol, direction=direction, size=size, stop_amount=stop_amount, profit_amount=profit_amount, stop_level=stop_level, profit_level=profit_level)
    
    def place_limit_order(self,
                         symbol: str,
                         side: str,
                         quantity: float,
                         price: float,
                         take_profit: Optional[float] = None,
                         stop_loss: Optional[float] = None) -> Dict:
        """Place a limit order."""
        logging.info(f"Placing limit order for {symbol}")
        return {}  # Return empty dict
    
    def close_order(self, X_SECURITY_TOKEN=None, CST=None, print_answer=True) -> bool:
        """Close an existing order. This går ut ifra that the model/we only have one active trade open at a time."""
        logging.error("Not made yet")
        return False
    
    def close_all_orders(self, X_SECURITY_TOKEN=None, CST=None, print_answer=True) -> bool:
        """Close an existing order. This is based on the principle that the model/we only have one active trade open at a time."""
        try:
            all_positions_after_new = trading.all_positions(X_SECURITY_TOKEN=X_SECURITY_TOKEN or self.x_security_token, CST=CST or self.cst, print_answer=print_answer)
            
            dealIds = br_util.process_positions(all_positions_after_new)
            
            logging.info(f"About to close ALL active positions...")
            
            for dealId in dealIds:
                logging.info(f"Closing trade/position with dealId: {dealId}")
                trading.close_position(X_SECURITY_TOKEN=X_SECURITY_TOKEN or self.x_security_token, CST=CST or self.cst, print_answer=print_answer,
                                       dealID=dealId)
            return True
        except Exception as e:
            logging.error(f"Unable to close all positions. Error: {e}")
            return False
    
    def modify_position(self, dealId=None, stop_amount=None, profit_amount=None, stop_level=None, profit_level=None,
                        X_SECURITY_TOKEN=None, CST=None,
                        print_answer=True) -> bool:
        """Modify an existing position. Note!! As with the close all order method, this wil modify the 'last' dealID, so really it is only meant for one active position. Must find a better solution later.
        The issue is that creating and placing an order, doesn't give you the right dealreference - it is only temporary. So to close a position, we need to get all active positions from active positions, and
        extract all dealIds there. So i have no way of filtering based on a position. I think this will work for now, but should be solved later. Perhaps based on the time it was placed, amount or symbol?"""
        try:
            if dealId is None:
                all_positions_after_new = trading.all_positions(X_SECURITY_TOKEN=X_SECURITY_TOKEN or self.x_security_token, CST=CST or self.cst, print_answer=print_answer)
                
                dealIds = br_util.process_positions(all_positions_after_new)  # Getting all dealIDs
                if len(dealIds) >= 2:
                    logging.error(f"There are {len(dealIds)} active positions, which is more than 1! Will not confinue to modify! Either specify the dealId, or close not-relevant positions.")
                    return False
                for deal_ID in dealIds:
                    dealId = deal_ID
            trading.update_position(X_SECURITY_TOKEN=X_SECURITY_TOKEN or self.x_security_token, CST=CST or self.cst, print_answer=print_answer,
                                    stop_amount=stop_amount, profit_amount=profit_amount, stop_level=stop_level, profit_level=profit_level, dealID=dealId)
            return True
        except Exception as e:
            logging.error(f"Unable to modify position: {e}")
            return False
        
    # ==================== LIVE DATA ====================
    
    def sub_live_market_data(self, symbol, timeframe, message_handler=None, X_SECURITY_TOKEN=None, CST=None):
        try:
            logging.info("About to initiate subscribtion to live market data...")
            
            # Initiating the subscription
            ws = web_socket.sub_live_market_data(X_SECURITY_TOKEN=X_SECURITY_TOKEN or self.x_security_token,
                                                 CST=CST or self.cst,
                                                 epic=symbol,
                                                 resolution=timeframe,
                                                 custom_message_handler=message_handler)
            # Set up automatic pinging every 1 minutes (60 seconds) to make sure we don't time out
            web_socket.setup_ping_timer(ws=ws,
                                        X_SECURITY_TOKEN=X_SECURITY_TOKEN or self.x_security_token,
                                        CST=CST or self.cst,
                                        interval_seconds=60)
            # Simple "hack" so that the program isn't terminated
            # try:
            #     while True:
            #         time.sleep(1)
            # except KeyboardInterrupt:
            #     print("Program terminated by user")
            
            return True
        except Exception as e:
            logging.error(f"Error occured: {e}")
            
            return False
