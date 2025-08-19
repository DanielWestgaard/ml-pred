import requests
import json
import pandas as pd
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from utils.logging_utils import get_logger
import utils.data_utils as data_utils
from utils.broker_utils import convert_alpaca_bars_to_ohlcv_csv

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
    
    # DATA SAVING METHODS
    
    def fetch_and_save_historical_bars(self, 
                                      symbols: Union[str, List[str]], 
                                      timeframe: str = "1Day",
                                      start: Optional[str] = None,
                                      end: Optional[str] = None,
                                      output_file: str = None,
                                      save_raw_data: bool = False,
                                      **kwargs) -> str:
        """
        Fetch historical bars and save to CSV file with proper directory structure.
        
        Args:
            symbols: Single symbol string or list of symbols
            timeframe: Bar timeframe (e.g., "1Min", "5Min", "1Hour", "1Day")
            start: Start time (RFC3339 format)
            end: End time (RFC3339 format)
            output_file: Custom output file path (optional)
            save_raw_data: Whether to save raw JSON response as well
            **kwargs: Additional parameters for get_stock_bars
            
        Returns:
            Path to the saved CSV file
        """
        # Handle single symbol
        if isinstance(symbols, list):
            if len(symbols) != 1:
                raise ValueError("Only single symbol supported for saving. Use fetch_and_save_multiple_symbols for multiple symbols.")
            symbol = symbols[0]
        else:
            symbol = symbols
        
        # Get the bars data
        bars_data = self.get_stock_bars(
            symbols=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            **kwargs
        )
        
        if "error" in bars_data:
            raise Exception(f"Failed to fetch data: {bars_data['error']}")
        
        # Extract date range for filename
        if "bars" in bars_data and symbol in bars_data["bars"] and bars_data["bars"][symbol]:
            first_bar = bars_data["bars"][symbol][0]
            last_bar = bars_data["bars"][symbol][-1]
            start_date = first_bar['t'][:10]  # Extract YYYY-MM-DD
            end_date = last_bar['t'][:10]
        else:
            raise ValueError("No data found to save")
        
        # Generate output file path if not provided
        if output_file is None:
            # Map Alpaca timeframes to our internal format
            timeframe_mapping = {
                "1Min": "1m", "5Min": "5m", "15Min": "15m", "30Min": "30m",
                "1Hour": "1h", "1Day": "1d", "1Week": "1w", "1Month": "1M"
            }
            normalized_timeframe = timeframe_mapping.get(timeframe, timeframe.lower())
            
            output_file = data_utils.get_data_file_path(
                symbol=symbol,
                timeframe=normalized_timeframe,
                start_date=start_date,
                end_date=end_date,
                is_raw=True,
                data_source='alpaca'
            )
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save the data to OHLCV CSV format
        if bars_data["bars"][symbol]:
            logger.info(f"Converting {len(bars_data['bars'][symbol])} Alpaca bars to OHLCV format")
            convert_alpaca_bars_to_ohlcv_csv(bars_data, output_file, symbol=symbol)
            logger.info(f"OHLCV data saved to {output_file}")
            
            # Print summary
            first_timestamp = bars_data["bars"][symbol][0]['t'] if bars_data["bars"][symbol] else "N/A"
            last_timestamp = bars_data["bars"][symbol][-1]['t'] if bars_data["bars"][symbol] else "N/A"
            logger.info(f"Data range: {first_timestamp} to {last_timestamp}")
            
            # Save raw JSON if requested
            if save_raw_data:
                raw_file = output_file.replace('.csv', '_raw.json')
                with open(raw_file, 'w') as f:
                    json.dump(bars_data, f, indent=2)
                logger.info(f"Raw JSON data saved to {raw_file}")
                
        else:
            logger.error("No data to save.")
            raise ValueError("No data to save")
        
        return output_file
    
    def fetch_and_save_multiple_symbols(self,
                                       symbols: List[str],
                                       timeframe: str = "1Day", 
                                       start: Optional[str] = None,
                                       end: Optional[str] = None,
                                       save_raw_data: bool = False,
                                       **kwargs) -> List[str]:
        """
        Fetch and save historical bars for multiple symbols.
        
        Args:
            symbols: List of symbols to fetch
            timeframe: Bar timeframe
            start: Start time (RFC3339 format)
            end: End time (RFC3339 format)
            save_raw_data: Whether to save raw JSON responses
            **kwargs: Additional parameters for get_stock_bars
            
        Returns:
            List of saved file paths
        """
        saved_files = []
        
        for symbol in symbols:
            try:
                logger.info(f"Fetching and saving data for {symbol}")
                output_file = self.fetch_and_save_historical_bars(
                    symbols=symbol,
                    timeframe=timeframe,
                    start=start,
                    end=end,
                    save_raw_data=save_raw_data,
                    **kwargs
                )
                saved_files.append(output_file)
                logger.info(f"Completed {symbol}: {output_file}")
                
            except Exception as e:
                logger.error(f"Failed to fetch/save data for {symbol}: {str(e)}")
                continue
        
        logger.info(f"Completed batch save: {len(saved_files)}/{len(symbols)} symbols successful")
        return saved_files
    
    def fetch_and_save_basic_plan_data(self,
                                      symbols: Union[str, List[str]],
                                      timeframe: str = "1Day",
                                      days_back: int = 30,
                                      save_raw_data: bool = False) -> Union[str, List[str]]:
        """
        Convenience method to fetch and save data optimized for basic plan.
        
        Args:
            symbols: Single symbol or list of symbols
            timeframe: Bar timeframe
            days_back: Number of days to look back from a week ago
            save_raw_data: Whether to save raw JSON responses
            
        Returns:
            File path (single symbol) or list of file paths (multiple symbols)
        """
        if isinstance(symbols, str):
            # Single symbol
            bars_data = self.get_historical_bars_basic_plan([symbols], timeframe, days_back)
            
            if "error" in bars_data:
                raise Exception(f"Failed to fetch data: {bars_data['error']}")
            
            # Extract date range
            if "bars" in bars_data and symbols in bars_data["bars"] and bars_data["bars"][symbols]:
                first_bar = bars_data["bars"][symbols][0]
                last_bar = bars_data["bars"][symbols][-1]
                start_date = first_bar['t'][:10]
                end_date = last_bar['t'][:10]
                
                # Generate output file path
                timeframe_mapping = {
                    "1Min": "1m", "5Min": "5m", "15Min": "15m", "30Min": "30m",
                    "1Hour": "1h", "1Day": "1d", "1Week": "1w", "1Month": "1M"
                }
                normalized_timeframe = timeframe_mapping.get(timeframe, timeframe.lower())
                
                output_file = data_utils.get_data_file_path(
                    symbol=symbols,
                    timeframe=normalized_timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    is_raw=True,
                    data_source='alpaca'
                )
                
                # Save the data
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                convert_alpaca_bars_to_ohlcv_csv(bars_data, output_file, symbol=symbols)
                logger.info(f"Basic plan data saved to {output_file}")
                
                if save_raw_data:
                    raw_file = output_file.replace('.csv', '_raw.json')
                    with open(raw_file, 'w') as f:
                        json.dump(bars_data, f, indent=2)
                
                return output_file
            else:
                raise ValueError("No data found to save")
                
        else:
            # Multiple symbols
            saved_files = []
            for symbol in symbols:
                try:
                    file_path = self.fetch_and_save_basic_plan_data(symbol, timeframe, days_back, save_raw_data)
                    saved_files.append(file_path)
                except Exception as e:
                    logger.error(f"Failed to save data for {symbol}: {str(e)}")
                    
            return saved_files