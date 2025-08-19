import requests
import json
from typing import Dict, Optional, Union
from utils.logging_utils import get_logger

logger = get_logger()

class TradingAPI:
    """
    Actual implementation of the possibilities with trading API. Called from alpaca.py.
    These methods and logic will wollow the naming provided about the Trading API in
    the documentation: https://docs.alpaca.markets/reference/
    
    TODO: In the future, if this file get overly populated with methods, it could be an idea
    to divide into files based on the "type". Eg. accounts.py, assets.py, etc.
    """
    
    def __init__(self, key_id, secret_key, print_responses):
        """Initialize the Alpaca Trading API."""
        
        self.key_id = key_id
        self.secret_key = secret_key
        self.print_responses = print_responses

        # Use paper trading endpoint by default, can be overridden for live trading
        self.endpoint = "https://paper-api.alpaca.markets/v2"
        self.headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": f"{self.key_id}",
            "APCA-API-SECRET-KEY": f"{self.secret_key}"
        }
        
        logger.info("Alpaca Trading API initialized successfully")
    
    def set_live_trading(self, live: bool = True):
        """
        Switch between paper and live trading endpoints.
        
        Args:
            live: True for live trading, False for paper trading
        """
        if live:
            self.endpoint = "https://api.alpaca.markets/v2"
            logger.warning("Switched to LIVE TRADING mode")
        else:
            self.endpoint = "https://paper-api.alpaca.markets/v2"
            logger.info("Switched to paper trading mode")
    
    # ACCOUNTS
    
    def get_account(self) -> dict:
        """
        Returns your account details including buying power, cash, and positions value.
        
        Returns:
            Dictionary containing account information
            
        Documentation: https://docs.alpaca.markets/reference/getaccount-1
        """
        try:
            response = requests.get(
                url=f"{self.endpoint}/account", 
                headers=self.headers
            )
            
            if self.print_responses:
                logger.debug(f"Account response status: {response.status_code}")
                logger.debug(f"Account response: {json.dumps(response.json(), indent=4)}")
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get account: {response.status_code} - {response.text}")
                return {"error": response.text, "status_code": response.status_code}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            return {"error": str(e)}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode response: {str(e)}")
            return {"error": f"Invalid JSON response: {str(e)}"}
    
    # ASSETS
    
    # COROPORATE ACTIONS
    
    # ORDERS
    
    def create_an_order(self, 
                       symbol: str, 
                       qty: float, 
                       side: str, 
                       order_type: str = "market",
                       time_in_force: str = "day",
                       limit_price: float = None,
                       stop_price: float = None,
                       trail_price: float = None,
                       trail_percent: float = None,
                       extended_hours: bool = False,
                       client_order_id: str = None,
                       order_class: str = "simple",
                       take_profit: dict = None,
                       stop_loss: dict = None) -> dict:
        """
        Places a new order for the given account. An order request may be rejected if the account
        is not authorized for trading, or if the tradable balance is insufficient to fill the order.
        
        Args:
            symbol: The asset symbol to trade (e.g., "AAPL", "SPY")
            qty: Number of shares to trade (can be fractional for supported assets)
            side: "buy" or "sell"
            order_type: "market", "limit", "stop", "stop_limit", "trailing_stop"
            time_in_force: "day", "gtc" (good til canceled), "ioc" (immediate or cancel), "fok" (fill or kill)
            limit_price: Required for limit and stop_limit orders
            stop_price: Required for stop and stop_limit orders
            trail_price: Trail amount in dollars for trailing stop orders
            trail_percent: Trail amount in percentage for trailing stop orders
            extended_hours: Whether to allow trading during extended hours
            client_order_id: Client unique order ID (optional)
            order_class: "simple", "bracket", "oco", "oto"
            take_profit: Dict with "limit_price" for bracket orders
            stop_loss: Dict with "stop_price" and optionally "limit_price" for bracket orders
        
        Returns:
            Dictionary containing order information or error details
            
        Documentation: https://docs.alpaca.markets/reference/postorder
        """
        
        # Build the basic payload
        payload = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force
        }
        
        # Add optional parameters based on order type
        if order_type in ["limit", "stop_limit"] and limit_price is not None:
            payload["limit_price"] = str(limit_price)
            
        if order_type in ["stop", "stop_limit"] and stop_price is not None:
            payload["stop_price"] = str(stop_price)
            
        if order_type == "trailing_stop":
            if trail_price is not None:
                payload["trail_price"] = str(trail_price)
            elif trail_percent is not None:
                payload["trail_percent"] = str(trail_percent)
            else:
                raise ValueError("Trailing stop orders require either trail_price or trail_percent")
        
        # Add other optional parameters
        if extended_hours:
            payload["extended_hours"] = extended_hours
            
        if client_order_id:
            payload["client_order_id"] = client_order_id
            
        if order_class != "simple":
            payload["order_class"] = order_class
            
        # Add bracket order parameters
        if order_class == "bracket":
            if take_profit:
                payload["take_profit"] = take_profit
            if stop_loss:
                payload["stop_loss"] = stop_loss
        
        try:
            response = requests.post(
                url=f"{self.endpoint}/orders", 
                json=payload, 
                headers=self.headers
            )
            
            if self.print_responses:
                logger.debug(f"Order request payload: {json.dumps(payload, indent=4)}")
                logger.debug(f"Order response status: {response.status_code}")
                logger.debug(f"Order response: {json.dumps(response.json(), indent=4)}")
            
            # Check for successful response
            if response.status_code == 201:
                logger.info(f"Order placed successfully: {response.json().get('id', 'unknown')}")
                return response.json()
            else:
                logger.error(f"Order failed with status {response.status_code}: {json.dumps(response.text, indent=4)}")
                return {"error": response.text, "status_code": response.status_code}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            return {"error": str(e)}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode response: {str(e)}")
            return {"error": f"Invalid JSON response: {str(e)}"}
    
    def get_all_orders(self, 
                      status: str = None,
                      limit: int = 50,
                      after: str = None,
                      until: str = None,
                      direction: str = "desc",
                      nested: bool = True,
                      symbols: str = None) -> dict:
        """
        Retrieves a list of orders for the account, filtered by the supplied query parameters.
        
        Args:
            status: Order status to filter by ("open", "closed", "all")
            limit: Maximum number of orders to return (1-500, default 50)
            after: Date string (RFC3339) or order ID for pagination
            until: Date string (RFC3339) for filtering orders before this time
            direction: "asc" or "desc" for chronological order
            nested: Include nested multi-leg orders
            symbols: Comma-separated symbols to filter by
            
        Returns:
            Dictionary containing list of orders
            
        Documentation: https://docs.alpaca.markets/reference/getallorders-1
        """
        
        # Build query parameters
        params = {
            "limit": str(limit),
            "direction": direction,
            "nested": str(nested).lower()
        }
        
        if status:
            params["status"] = status
        if after:
            params["after"] = after
        if until:
            params["until"] = until
        if symbols:
            params["symbols"] = symbols
            
        try:
            response = requests.get(
                url=f"{self.endpoint}/orders", 
                params=params,
                headers=self.headers
            )
            
            if self.print_responses:
                logger.debug(f"Get orders response status: {response.status_code}")
                logger.debug(f"Get orders response: {json.dumps(response.json(), indent=4)}")
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get orders: {response.status_code} - {response.text}")
                return {"error": response.text, "status_code": response.status_code}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            return {"error": str(e)}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode response: {str(e)}")
            return {"error": f"Invalid JSON response: {str(e)}"}
    
    def cancel_order(self, order_id: str) -> dict:
        """
        Cancel an existing order by its ID.
        
        Args:
            order_id: The ID of the order to cancel
            
        Returns:
            Dictionary containing cancellation confirmation or error
            
        Documentation: https://docs.alpaca.markets/reference/deleteorder
        """
        try:
            response = requests.delete(
                url=f"{self.endpoint}/orders/{order_id}",
                headers=self.headers
            )
            
            if self.print_responses:
                logger.debug(f"Cancel order response status: {response.status_code}")
                
            if response.status_code == 204:
                logger.info(f"Order {order_id} cancelled successfully")
                return {"success": True, "order_id": order_id}
            else:
                logger.error(f"Failed to cancel order {order_id}: {response.status_code} - {response.text}")
                return {"error": response.text, "status_code": response.status_code}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            return {"error": str(e)}

    def get_order(self, order_id: str) -> dict:
        """
        Get a specific order by its ID.
        
        Args:
            order_id: The ID of the order to retrieve
            
        Returns:
            Dictionary containing order details
            
        Documentation: https://docs.alpaca.markets/reference/getorder
        """
        try:
            response = requests.get(
                url=f"{self.endpoint}/orders/{order_id}",
                headers=self.headers
            )
            
            if self.print_responses:
                logger.debug(f"Get order response status: {response.status_code}")
                logger.debug(f"Get order response: {json.dumps(response.json(), indent=4)}")
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get order {order_id}: {response.status_code} - {response.text}")
                return {"error": response.text, "status_code": response.status_code}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            return {"error": str(e)}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode response: {str(e)}")
            return {"error": f"Invalid JSON response: {str(e)}"}

    # POSITIONS
    
    def get_all_positions(self) -> dict:
        """
        Get all open positions for the account.
        
        Returns:
            Dictionary containing list of positions
            
        Documentation: https://docs.alpaca.markets/reference/getallpositions
        """
        try:
            response = requests.get(
                url=f"{self.endpoint}/positions",
                headers=self.headers
            )
            
            if self.print_responses:
                logger.debug(f"Get positions response status: {response.status_code}")
                logger.debug(f"Get positions response: {json.dumps(response.json(), indent=4)}")
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get positions: {response.status_code} - {response.text}")
                return {"error": response.text, "status_code": response.status_code}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            return {"error": str(e)}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode response: {str(e)}")
            return {"error": f"Invalid JSON response: {str(e)}"}

    def get_position(self, symbol: str) -> dict:
        """
        Get a specific position by symbol.
        
        Args:
            symbol: The asset symbol (e.g., "AAPL", "SPY")
            
        Returns:
            Dictionary containing position details
            
        Documentation: https://docs.alpaca.markets/reference/getposition
        """
        try:
            response = requests.get(
                url=f"{self.endpoint}/positions/{symbol}",
                headers=self.headers
            )
            
            if self.print_responses:
                logger.debug(f"Get position response status: {response.status_code}")
                logger.debug(f"Get position response: {json.dumps(response.json(), indent=4)}")
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get position for {symbol}: {response.status_code} - {response.text}")
                return {"error": response.text, "status_code": response.status_code}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            return {"error": str(e)}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode response: {str(e)}")
            return {"error": f"Invalid JSON response: {str(e)}"}

    def close_position(self, symbol: str, qty: str = None, percentage: str = None) -> dict:
        """
        Close a position (liquidate).
        
        Args:
            symbol: The asset symbol to close
            qty: Number of shares to close (optional, closes all if not specified)
            percentage: Percentage of position to close (optional)
            
        Returns:
            Dictionary containing liquidation order details
            
        Documentation: https://docs.alpaca.markets/reference/deleteposition
        """
        params = {}
        if qty is not None:
            params["qty"] = str(qty)
        if percentage is not None:
            params["percentage"] = str(percentage)
            
        try:
            response = requests.delete(
                url=f"{self.endpoint}/positions/{symbol}",
                params=params,
                headers=self.headers
            )
            
            if self.print_responses:
                logger.debug(f"Close position response status: {response.status_code}")
                logger.debug(f"Close position response: {json.dumps(response.json(), indent=4)}")
            
            if response.status_code == 200:
                logger.info(f"Position {symbol} closed successfully")
                return response.json()
            else:
                logger.error(f"Failed to close position {symbol}: {response.status_code} - {response.text}")
                return {"error": response.text, "status_code": response.status_code}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            return {"error": str(e)}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode response: {str(e)}")
            return {"error": f"Invalid JSON response: {str(e)}"}

    # PORTFOLIO HISTORY
    
    # WATCHLISTS
    
    # ACCOUNT CONFIGURATIONS
    
    # ACCOUNT ACTIVITIES
    
    # CALENDAR
    
    # CLOCK
    
    # CRYPTO FUNDING
    
    # CONVENIENCE METHODS
    
    def place_market_order(self, symbol: str, side: str, qty: float, **kwargs) -> dict:
        """
        Convenience method to place a market order.
        
        Args:
            symbol: Asset symbol (e.g., "AAPL", "SPY")
            side: "buy" or "sell"
            qty: Number of shares
            **kwargs: Additional parameters for create_an_order
            
        Returns:
            Dictionary containing order information
        """
        return self.create_an_order(
            symbol=symbol,
            side=side,
            qty=qty,
            order_type="market",
            **kwargs
        )
    
    def place_limit_order(self, symbol: str, side: str, qty: float, limit_price: float, **kwargs) -> dict:
        """
        Convenience method to place a limit order.
        
        Args:
            symbol: Asset symbol (e.g., "AAPL", "SPY")
            side: "buy" or "sell"
            qty: Number of shares
            limit_price: Limit price for the order
            **kwargs: Additional parameters for create_an_order
            
        Returns:
            Dictionary containing order information
        """
        return self.create_an_order(
            symbol=symbol,
            side=side,
            qty=qty,
            order_type="limit",
            limit_price=limit_price,
            **kwargs
        )
    
    def place_bracket_order(self, 
                           symbol: str, 
                           side: str, 
                           qty: float,
                           take_profit_price: float = None,
                           stop_loss_price: float = None,
                           **kwargs) -> dict:
        """
        Convenience method to place a bracket order with take profit and stop loss.
        
        Args:
            symbol: Asset symbol (e.g., "AAPL", "SPY")
            side: "buy" or "sell"
            qty: Number of shares
            take_profit_price: Take profit price
            stop_loss_price: Stop loss price
            **kwargs: Additional parameters for create_an_order
            
        Returns:
            Dictionary containing order information
        """
        take_profit = {"limit_price": str(take_profit_price)} if take_profit_price else None
        stop_loss = {"stop_price": str(stop_loss_price)} if stop_loss_price else None
        
        return self.create_an_order(
            symbol=symbol,
            side=side,
            qty=qty,
            order_type="market",
            order_class="bracket",
            take_profit=take_profit,
            stop_loss=stop_loss,
            **kwargs
        )