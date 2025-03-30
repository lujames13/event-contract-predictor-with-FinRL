"""
Binance Client Module

This module provides functionality to interact with the Binance API for 
fetching historical price data and current prices.

Classes:
    BinanceClient: Main client for Binance API interaction
"""

import os
import time
import random
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from binance.client import Client
from binance.exceptions import BinanceAPIException

from utils.logger import get_logger
from utils.time_utils import get_timestamp_ms, get_datetime_from_ms, get_binance_interval
from config.config_manager import ConfigManager

from dotenv import load_dotenv
load_dotenv()

# Setup logger
logger = get_logger(__name__)


class BinanceClient:
    """
    Client for interacting with the Binance API.
    
    This class provides methods for fetching historical price data and current prices,
    with error handling and retry mechanisms.
    """
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """
        Initialize the Binance client.
        
        Args:
            api_key (str, optional): Binance API key. If None, tries to load from environment.
            api_secret (str, optional): Binance API secret. If None, tries to load from environment.
        """
        self.logger = get_logger("BinanceClient")
        
        # Load API credentials from environment if not provided
        self.api_key = api_key or os.getenv("BINANCE_API_KEY", "")
        self.api_secret = api_secret or os.getenv("BINANCE_API_SECRET", "")
        
        # Get configuration
        config_manager = ConfigManager()
        self.data_config = config_manager.get_data_config()
        
        # Initialize client
        self._initialize_client()
        
        # Track API request count for rate limiting
        self.request_count = 0
        self.last_request_time = time.time()
    
    def _initialize_client(self):
        """Initialize the Binance client with API credentials."""
        try:
            if self.api_key and self.api_secret:
                self.client = Client(self.api_key, self.api_secret)
                self.logger.info("Initialized Binance client with API credentials")
            else:
                self.client = Client("", "")  # Testnet client
                self.logger.warning("Initialized Binance client without API credentials - using limited public API")
            
            # Test connection
            self.client.ping()
            self.logger.info("Successfully connected to Binance API")
        except BinanceAPIException as e:
            self.logger.error(f"Failed to initialize Binance client: {e}")
            raise
    
    def _handle_request(self, func, *args, **kwargs):
        """
        Handle API request with retry mechanism and rate limiting.
        
        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            The result of the function call
        """
        max_retries = self.data_config.max_retries
        retry_delay = self.data_config.retry_delay_seconds
        
        # Implement rate limiting
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        # If making requests too quickly, add a small delay
        if elapsed < 0.2:  # No more than 5 requests per second
            sleep_time = 0.2 - elapsed + random.uniform(0, 0.1)  # Add jitter
            self.logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        # Update request tracking
        self.request_count += 1
        self.last_request_time = time.time()
        
        # Make the request with retries
        for retry in range(max_retries):
            try:
                result = func(*args, **kwargs)
                return result
            except BinanceAPIException as e:
                if e.code == -1003:  # Too many requests
                    sleep_time = (retry + 1) * retry_delay * 2  # Exponential backoff
                    self.logger.warning(f"Too many requests, retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                elif e.code == -1021:  # Timestamp for this request was outside the recvWindow
                    self.logger.warning("Timestamp error, retrying...")
                    time.sleep(retry_delay)
                elif retry < max_retries - 1:
                    self.logger.warning(f"API error {e.code}: {e.message}, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    self.logger.error(f"API error after {max_retries} retries: {e}")
                    raise
            except Exception as e:
                if retry < max_retries - 1:
                    self.logger.warning(f"Unexpected error: {e}, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    self.logger.error(f"Unexpected error after {max_retries} retries: {e}")
                    raise
    
    def get_historical_klines(self, symbol: str, interval: str, 
                             start_time: Optional[datetime] = None,
                             end_time: Optional[datetime] = None,
                             limit: int = 1000) -> List[List]:
        """
        Get historical klines (candlestick data) from Binance.
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
            interval (str): Kline interval (e.g., '1m', '1h')
            start_time (datetime, optional): Start time. If None, uses end_time - 1000 intervals.
            end_time (datetime, optional): End time. If None, uses current time.
            limit (int): Maximum number of klines to return (1-1000)
            
        Returns:
            List[List]: List of klines with format:
                [
                    [open_time, open, high, low, close, volume, close_time, quote_volume, 
                     trades_count, taker_buy_base_volume, taker_buy_quote_volume, ignored]
                ]
        """
        # Convert interval to Binance format
        binance_interval = get_binance_interval(interval)
        
        # Prepare request arguments
        kwargs = {
            "symbol": symbol,
            "interval": binance_interval,
            "limit": limit
        }
        
        # Add start and end times if provided
        if start_time:
            kwargs["startTime"] = get_timestamp_ms(start_time)
        
        if end_time:
            kwargs["endTime"] = get_timestamp_ms(end_time)
        
        # Make the request
        self.logger.info(f"Fetching historical klines for {symbol} {interval} "
                        f"from {start_time or 'N/A'} to {end_time or 'now'}")
        
        return self._handle_request(self.client.get_klines, **kwargs)
    
    def get_historical_klines_dataframe(self, symbol: str, interval: str,
                                       start_time: Optional[datetime] = None,
                                       end_time: Optional[datetime] = None,
                                       limit: int = 1000) -> pd.DataFrame:
        """
        Get historical klines as a pandas DataFrame.
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
            interval (str): Kline interval (e.g., '1m', '1h')
            start_time (datetime, optional): Start time
            end_time (datetime, optional): End time
            limit (int): Maximum number of klines
            
        Returns:
            pd.DataFrame: DataFrame with columns:
                date, open, high, low, close, volume, close_time, quote_volume, 
                trades, taker_buy_base, taker_buy_quote, symbol
        """
        # Get raw klines
        klines = self.get_historical_klines(symbol, interval, start_time, end_time, limit)
        
        if not klines:
            self.logger.warning(f"No klines returned for {symbol} {interval}")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                'date', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'symbol'
            ])
        
        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignored'
        ])
        
        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume',
                    'taker_buy_base', 'taker_buy_quote']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Add symbol column
        df['symbol'] = symbol
        
        # Rename timestamp to date
        df = df.rename(columns={'timestamp': 'date'})
        
        # Drop ignored column
        df = df.drop(columns=['ignored'])
        
        return df
    
    def get_historical_data(self, symbol: str, interval: str, lookback_days: int) -> pd.DataFrame:
        """
        Get historical data for the specified lookback period.
        
        This method handles pagination to fetch more than 1000 klines if needed.
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
            interval (str): Kline interval (e.g., '1m', '1h')
            lookback_days (int): Number of days to look back
            
        Returns:
            pd.DataFrame: DataFrame with historical data
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=lookback_days)
        
        self.logger.info(f"Fetching {lookback_days} days of historical data for {symbol} {interval}")
        
        # For large time ranges, we need to paginate
        interval_seconds = self._interval_to_seconds(interval)
        max_records = 1000
        
        # Calculate the time range in seconds
        time_range_seconds = lookback_days * 24 * 60 * 60
        
        # Calculate number of intervals in the time range
        intervals_count = time_range_seconds // interval_seconds
        
        # If we need more than 1000 records, paginate
        if intervals_count > max_records:
            all_klines = []
            current_end = end_time
            
            while current_end > start_time:
                # Calculate batch size
                batch_seconds = interval_seconds * max_records
                current_start = max(start_time, current_end - timedelta(seconds=batch_seconds))
                
                self.logger.debug(f"Fetching batch from {current_start} to {current_end}")
                
                # Get batch of klines
                klines = self.get_historical_klines(
                    symbol=symbol,
                    interval=interval,
                    start_time=current_start,
                    end_time=current_end,
                    limit=max_records
                )
                
                if not klines:
                    break
                
                all_klines = klines + all_klines
                
                # Update end time for next batch (use the start time of the first kline in this batch)
                if klines:
                    # Subtract a small amount to avoid overlap
                    current_end = get_datetime_from_ms(klines[0][0]) - timedelta(seconds=1)
                else:
                    break
            
            # Convert to DataFrame
            if all_klines:
                df = pd.DataFrame(all_klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignored'
                ])
                
                # Convert types
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
                
                for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume',
                            'taker_buy_base', 'taker_buy_quote']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Add symbol column
                df['symbol'] = symbol
                
                # Rename timestamp to date
                df = df.rename(columns={'timestamp': 'date'})
                
                # Drop ignored column
                df = df.drop(columns=['ignored'])
                
                # Sort by date
                df = df.sort_values('date')
                
                return df
            else:
                self.logger.warning(f"No data returned for {symbol} {interval}")
                return pd.DataFrame()
        else:
            # For smaller time ranges, get all at once
            return self.get_historical_klines_dataframe(
                symbol=symbol,
                interval=interval,
                start_time=start_time,
                end_time=end_time,
                limit=max_records
            )
    
    def get_current_price(self, symbol: str) -> float:
        """
        Get the current price for a symbol.
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
            
        Returns:
            float: Current price
        """
        try:
            ticker = self._handle_request(self.client.get_symbol_ticker, symbol=symbol)
            price = float(ticker['price'])
            self.logger.debug(f"Current price for {symbol}: {price}")
            return price
        except Exception as e:
            self.logger.error(f"Failed to get current price for {symbol}: {e}")
            raise
    
    def get_multiple_prices(self, symbols: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Get current prices for multiple symbols.
        
        Args:
            symbols (List[str], optional): List of symbols. If None, get all prices.
            
        Returns:
            Dict[str, float]: Dictionary of symbol -> price
        """
        try:
            if symbols:
                tickers = self._handle_request(self.client.get_symbol_ticker, symbols=symbols)
            else:
                tickers = self._handle_request(self.client.get_symbol_ticker)
            
            prices = {ticker['symbol']: float(ticker['price']) for ticker in tickers}
            return prices
        except Exception as e:
            self.logger.error(f"Failed to get multiple prices: {e}")
            raise
    
    def get_exchange_info(self, symbol: Optional[str] = None) -> Dict:
        """
        Get exchange information.
        
        Args:
            symbol (str, optional): Trading pair symbol. If None, get all symbols.
            
        Returns:
            Dict: Exchange information
        """
        try:
            if symbol:
                return self._handle_request(self.client.get_exchange_info, symbol=symbol)
            else:
                return self._handle_request(self.client.get_exchange_info)
        except Exception as e:
            self.logger.error(f"Failed to get exchange info: {e}")
            raise
    
    def get_server_time(self) -> datetime:
        """
        Get Binance server time.
        
        Returns:
            datetime: Server time
        """
        try:
            server_time = self._handle_request(self.client.get_server_time)
            return get_datetime_from_ms(server_time['serverTime'])
        except Exception as e:
            self.logger.error(f"Failed to get server time: {e}")
            raise
    
    def _interval_to_seconds(self, interval: str) -> int:
        """
        Convert an interval string to seconds.
        
        Args:
            interval (str): Interval string (e.g., '1m', '1h')
            
        Returns:
            int: Interval in seconds
        """
        unit = interval[-1]
        value = int(interval[:-1])
        
        if unit == 'm':
            return value * 60
        elif unit == 'h':
            return value * 60 * 60
        elif unit == 'd':
            return value * 24 * 60 * 60
        elif unit == 'w':
            return value * 7 * 24 * 60 * 60
        else:
            raise ValueError(f"Invalid interval unit: {unit}")
    
    def get_symbol_info(self, symbol: str) -> Dict:
        """
        Get detailed information for a symbol.
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
            
        Returns:
            Dict: Symbol information
        """
        try:
            exchange_info = self._handle_request(self.client.get_exchange_info, symbol=symbol)
            symbol_info = None
            
            for s in exchange_info['symbols']:
                if s['symbol'] == symbol:
                    symbol_info = s
                    break
            
            if not symbol_info:
                raise ValueError(f"Symbol {symbol} not found in exchange info")
            
            return symbol_info
        except Exception as e:
            self.logger.error(f"Failed to get symbol info for {symbol}: {e}")
            raise
    
    def get_all_tickers(self) -> List[Dict]:
        """
        Get all price tickers.
        
        Returns:
            List[Dict]: List of all price tickers
        """
        try:
            return self._handle_request(self.client.get_all_tickers)
        except Exception as e:
            self.logger.error(f"Failed to get all tickers: {e}")
            raise