import requests
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from utils.logging_utils import get_logger

logger = get_logger()

class HistoricalDataAPI:
    """
    Alpaca Historical Market Data API implementation.
    
    Provides access to historical stock bars, quotes, trades, and snapshots.
    Supports both basic and premium data subscriptions.
    
    Documentation: https://docs.alpaca.markets/docs/historical-stock-data
    """
    
    def __init__(self, key_id: str, secret_key: str, print_responses: bool = False):
        """
        Initialize the Historical Data API.
        
        Args:
            key_id: Alpaca API key ID
            secret_key: Alpaca API secret key
            print_responses: Whether to print API responses for debugging
        """
        self.key_id = key_id
        self.secret_key = secret_key
        self.print_responses = print_responses
        
        # Market Data API has its own endpoint
        self.endpoint = "https://data.alpaca.markets/v2"
        self.headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": self.key_id,
            "APCA-API-SECRET-KEY": self.secret_key
        }
        
        logger.info("Alpaca Historical Data API initialized successfully")
    
    def get_stock_bars(self, 
                      symbols: Union[str, List[str]], 
                      timeframe: str = "1Day",
                      start: Optional[str] = None,
                      end: Optional[str] = None,
                      limit: Optional[int] = None,
                      page_token: Optional[str] = None,
                      adjustment: str = "raw",
                      feed: Optional[str] = None,
                      sort: str = "asc") -> Dict:
        """
        Get historical stock bars (OHLCV data).
        
        Args:
            symbols: Single symbol string or list of symbols (e.g., "AAPL" or ["AAPL", "MSFT"])
            timeframe: Bar timeframe - "1Min", "5Min", "15Min", "30Min", "1Hour", "1Day", "1Week", "1Month"
            start: Start time (RFC3339 format or "2022-01-01T00:00:00Z")
            end: End time (RFC3339 format)
            limit: Maximum number of bars to return (default 1000, max 10000)
            page_token: Token for pagination
            adjustment: Price adjustment type - "raw", "split", "dividend", "all"
            feed: Data feed (premium subscribers) - "iex", "sip"
            sort: Sort order - "asc" or "desc"
            
        Returns:
            Dictionary containing bars data with metadata
        """
        # Handle single symbol or list of symbols
        if isinstance(symbols, str):
            symbols_param = symbols
        else:
            symbols_param = ",".join(symbols)
        
        # Build query parameters
        params = {
            "symbols": symbols_param,
            "timeframe": timeframe,
            "adjustment": adjustment,
            "sort": sort
        }
        
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        if limit:
            params["limit"] = str(limit)
        if page_token:
            params["page_token"] = page_token
        if feed:
            params["feed"] = feed
        
        try:
            response = requests.get(
                url=f"{self.endpoint}/stocks/bars",
                params=params,
                headers=self.headers
            )
            
            if self.print_responses:
                logger.debug(f"Stock bars request: {response.url}")
                logger.debug(f"Stock bars response status: {response.status_code}")
                if response.status_code == 200:
                    logger.debug(f"Stock bars response: {json.dumps(response.json(), indent=2)}")
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get stock bars: {response.status_code} - {response.text}")
                return {"error": response.text, "status_code": response.status_code}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            return {"error": str(e)}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode response: {str(e)}")
            return {"error": f"Invalid JSON response: {str(e)}"}
    
    def get_stock_quotes(self,
                        symbols: Union[str, List[str]],
                        start: Optional[str] = None,
                        end: Optional[str] = None,
                        limit: Optional[int] = None,
                        page_token: Optional[str] = None,
                        feed: Optional[str] = None,
                        sort: str = "asc") -> Dict:
        """
        Get historical stock quotes (bid/ask data).
        
        Args:
            symbols: Single symbol string or list of symbols
            start: Start time (RFC3339 format)
            end: End time (RFC3339 format)
            limit: Maximum number of quotes to return
            page_token: Token for pagination
            feed: Data feed (premium subscribers)
            sort: Sort order - "asc" or "desc"
            
        Returns:
            Dictionary containing quotes data
        """
        if isinstance(symbols, str):
            symbols_param = symbols
        else:
            symbols_param = ",".join(symbols)
        
        params = {
            "symbols": symbols_param,
            "sort": sort
        }
        
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        if limit:
            params["limit"] = str(limit)
        if page_token:
            params["page_token"] = page_token
        if feed:
            params["feed"] = feed
        
        try:
            response = requests.get(
                url=f"{self.endpoint}/stocks/quotes",
                params=params,
                headers=self.headers
            )
            
            if self.print_responses:
                logger.debug(f"Stock quotes response status: {response.status_code}")
                if response.status_code == 200:
                    logger.debug(f"Stock quotes response: {json.dumps(response.json(), indent=2)}")
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get stock quotes: {response.status_code} - {response.text}")
                return {"error": response.text, "status_code": response.status_code}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            return {"error": str(e)}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode response: {str(e)}")
            return {"error": f"Invalid JSON response: {str(e)}"}
    
    def get_stock_trades(self,
                        symbols: Union[str, List[str]],
                        start: Optional[str] = None,
                        end: Optional[str] = None,
                        limit: Optional[int] = None,
                        page_token: Optional[str] = None,
                        feed: Optional[str] = None,
                        sort: str = "asc") -> Dict:
        """
        Get historical stock trades data.
        
        Args:
            symbols: Single symbol string or list of symbols
            start: Start time (RFC3339 format)
            end: End time (RFC3339 format)
            limit: Maximum number of trades to return
            page_token: Token for pagination
            feed: Data feed (premium subscribers)
            sort: Sort order - "asc" or "desc"
            
        Returns:
            Dictionary containing trades data
        """
        if isinstance(symbols, str):
            symbols_param = symbols
        else:
            symbols_param = ",".join(symbols)
        
        params = {
            "symbols": symbols_param,
            "sort": sort
        }
        
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        if limit:
            params["limit"] = str(limit)
        if page_token:
            params["page_token"] = page_token
        if feed:
            params["feed"] = feed
        
        try:
            response = requests.get(
                url=f"{self.endpoint}/stocks/trades",
                params=params,
                headers=self.headers
            )
            
            if self.print_responses:
                logger.debug(f"Stock trades response status: {response.status_code}")
                if response.status_code == 200:
                    logger.debug(f"Stock trades response: {json.dumps(response.json(), indent=2)}")
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get stock trades: {response.status_code} - {response.text}")
                return {"error": response.text, "status_code": response.status_code}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            return {"error": str(e)}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode response: {str(e)}")
            return {"error": f"Invalid JSON response: {str(e)}"}
    
    def get_latest_bars(self, symbols: Union[str, List[str]], feed: Optional[str] = None) -> Dict:
        """
        Get the latest minute bars for symbols.
        
        Args:
            symbols: Single symbol string or list of symbols
            feed: Data feed (premium subscribers)
            
        Returns:
            Dictionary containing latest bars data
        """
        if isinstance(symbols, str):
            symbols_param = symbols
        else:
            symbols_param = ",".join(symbols)
        
        params = {"symbols": symbols_param}
        if feed:
            params["feed"] = feed
        
        try:
            response = requests.get(
                url=f"{self.endpoint}/stocks/bars/latest",
                params=params,
                headers=self.headers
            )
            
            if self.print_responses:
                logger.debug(f"Latest bars response status: {response.status_code}")
                if response.status_code == 200:
                    logger.debug(f"Latest bars response: {json.dumps(response.json(), indent=2)}")
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get latest bars: {response.status_code} - {response.text}")
                return {"error": response.text, "status_code": response.status_code}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            return {"error": str(e)}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode response: {str(e)}")
            return {"error": f"Invalid JSON response: {str(e)}"}
    
    def get_snapshots(self, symbols: Union[str, List[str]], feed: Optional[str] = None) -> Dict:
        """
        Get market snapshots for symbols (latest quote, trade, minute bar, etc.).
        
        Args:
            symbols: Single symbol string or list of symbols
            feed: Data feed (premium subscribers)
            
        Returns:
            Dictionary containing snapshot data
        """
        if isinstance(symbols, str):
            symbols_param = symbols
        else:
            symbols_param = ",".join(symbols)
        
        params = {"symbols": symbols_param}
        if feed:
            params["feed"] = feed
        
        try:
            response = requests.get(
                url=f"{self.endpoint}/stocks/snapshots",
                params=params,
                headers=self.headers
            )
            
            if self.print_responses:
                logger.debug(f"Snapshots response status: {response.status_code}")
                if response.status_code == 200:
                    logger.debug(f"Snapshots response: {json.dumps(response.json(), indent=2)}")
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get snapshots: {response.status_code} - {response.text}")
                return {"error": response.text, "status_code": response.status_code}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            return {"error": str(e)}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode response: {str(e)}")
            return {"error": f"Invalid JSON response: {str(e)}"}
    
    # CONVENIENCE METHODS
    
    def get_daily_bars(self, 
                      symbols: Union[str, List[str]], 
                      days_back: int = 30,
                      feed: str = "iex",
                      **kwargs) -> Dict:
        """
        Convenience method to get daily bars for the last N days.
        
        Args:
            symbols: Single symbol string or list of symbols
            days_back: Number of days to look back (default 30)
            feed: Data feed to use ("iex" for basic plan, "sip" for premium)
            **kwargs: Additional parameters for get_stock_bars
            
        Returns:
            Dictionary containing daily bars data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        return self.get_stock_bars(
            symbols=symbols,
            timeframe="1Day",
            start=start_date.strftime("%Y-%m-%dT00:00:00Z"),
            end=end_date.strftime("%Y-%m-%dT23:59:59Z"),
            feed=feed,
            **kwargs
        )
    
    def get_minute_bars(self, 
                       symbols: Union[str, List[str]], 
                       hours_back: int = 24,
                       **kwargs) -> Dict:
        """
        Convenience method to get minute bars for the last N hours.
        
        Args:
            symbols: Single symbol string or list of symbols
            hours_back: Number of hours to look back (default 24)
            **kwargs: Additional parameters for get_stock_bars
            
        Returns:
            Dictionary containing minute bars data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=hours_back)
        
        return self.get_stock_bars(
            symbols=symbols,
            timeframe="1Min",
            start=start_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            end=end_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            **kwargs
        )
    
    def bars_to_dataframe(self, bars_response: Dict, symbol: str = None) -> Optional[pd.DataFrame]:
        """
        Convert bars API response to pandas DataFrame.
        
        Args:
            bars_response: Response from get_stock_bars
            symbol: Specific symbol to extract (if multiple symbols in response)
            
        Returns:
            DataFrame with OHLCV data or None if error
        """
        try:
            if "error" in bars_response:
                logger.error(f"Cannot convert error response to DataFrame: {bars_response['error']}")
                return None
            
            if "bars" not in bars_response:
                logger.warning("No bars data found in response")
                return None
            
            bars_data = bars_response["bars"]
            
            # If specific symbol requested
            if symbol and symbol in bars_data:
                bars_list = bars_data[symbol]
            elif len(bars_data) == 1:
                # Single symbol in response
                symbol_key = next(iter(bars_data))
                bars_list = bars_data[symbol_key]
            else:
                logger.error("Multiple symbols found. Please specify symbol parameter")
                return None
            
            if not bars_list:
                logger.warning("No bars data found for symbol")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(bars_list)
            
            # Convert timestamp to datetime index
            if 't' in df.columns:
                df['timestamp'] = pd.to_datetime(df['t'])
                df.set_index('timestamp', inplace=True)
                df.drop('t', axis=1, inplace=True)
            
            # Rename columns to standard OHLCV format
            column_mapping = {
                'o': 'open',
                'h': 'high', 
                'l': 'low',
                'c': 'close',
                'v': 'volume',
                'n': 'trade_count',
                'vw': 'vwap'
            }
            
            df.rename(columns=column_mapping, inplace=True)
            
            logger.info(f"Created DataFrame with {len(df)} bars")
            return df
            
        except Exception as e:
            logger.error(f"Error converting bars to DataFrame: {str(e)}")
            return None
    
    def get_historical_bars_basic_plan(self, 
                                      symbols: Union[str, List[str]], 
                                      timeframe: str = "1Day",
                                      days_back: int = 30) -> Dict:
        """
        Get historical bars optimized for basic plan (free tier).
        Uses IEX feed and avoids recent data restrictions.
        
        Args:
            symbols: Single symbol string or list of symbols
            timeframe: Bar timeframe (default "1Day")
            days_back: Number of days to look back from a week ago (to avoid recent data restrictions)
            
        Returns:
            Dictionary containing historical bars data
        """
        # End date is a week ago to avoid recent data restrictions
        end_date = datetime.now() - timedelta(days=7)
        start_date = end_date - timedelta(days=days_back)
        
        return self.get_stock_bars(
            symbols=symbols,
            timeframe=timeframe,
            start=start_date.strftime("%Y-%m-%dT00:00:00Z"),
            end=end_date.strftime("%Y-%m-%dT23:59:59Z"),
            feed="iex",  # Use IEX feed for basic plan
            adjustment="raw"
        )