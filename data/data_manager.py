"""
Data Manager Module

This module handles data acquisition, caching, and processing for the 
BTC price prediction system. It interfaces with the BinanceClient to 
fetch data and implements multi-level caching.

Classes:
    DataCacheManager: Manages data caching operations
    DataManager: Main class for data acquisition and management
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

from utils.logger import get_logger
from utils.file_utils import (
    ensure_directory, get_cache_filename, save_to_json, load_from_json,
    is_cache_valid, clean_old_cache, save_dataframe, load_dataframe
)
from utils.time_utils import (
    get_timestamp_ms, get_datetime_from_ms, get_date_range,
    time_windows_overlap, calculate_missing_periods
)
from config.config_manager import ConfigManager
from data.binance_client import BinanceClient

# Setup logger
logger = get_logger(__name__)


class DataCacheManager:
    """
    Manages data caching operations for different levels of data processing.
    
    This class implements a multi-level caching strategy:
    - Level 1: Raw data cache (JSON format)
    - Level 2: Processed data cache (Parquet format)
    - Level 3: Feature data cache (Parquet format)
    """
    
    def __init__(self, cache_dir: str = "cache"):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir (str): Base cache directory
        """
        self.logger = get_logger("DataCacheManager")
        self.cache_dir = cache_dir
        
        # Get configuration
        config_manager = ConfigManager()
        self.data_config = config_manager.get_data_config()
        
        # Define cache subdirectories
        self.raw_cache_dir = os.path.join(cache_dir, "raw")
        self.processed_cache_dir = os.path.join(cache_dir, "processed")
        self.feature_cache_dir = os.path.join(cache_dir, "features")
        
        # Create cache directories
        self._create_cache_directories()
        
        # Cache hit statistics
        self.cache_hits = {
            "raw": 0,
            "processed": 0,
            "features": 0
        }
        self.cache_misses = {
            "raw": 0,
            "processed": 0,
            "features": 0
        }
    
    def _create_cache_directories(self):
        """Create the necessary cache directories."""
        ensure_directory(self.raw_cache_dir)
        ensure_directory(self.processed_cache_dir)
        ensure_directory(self.feature_cache_dir)
        self.logger.debug("Cache directories created")
    
    def get_raw_cache_path(self, symbol: str, timeframe: str, 
                         start_date: datetime, end_date: datetime) -> str:
        """
        Get the path for raw data cache.
        
        Args:
            symbol (str): Trading symbol
            timeframe (str): Timeframe
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            str: Cache file path
        """
        return get_cache_filename(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            cache_dir=self.raw_cache_dir,
            extension="json"
        )
    
    def get_processed_cache_path(self, symbol: str, timeframe: str, 
                               start_date: datetime, end_date: datetime) -> str:
        """
        Get the path for processed data cache.
        
        Args:
            symbol (str): Trading symbol
            timeframe (str): Timeframe
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            str: Cache file path
        """
        return get_cache_filename(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            cache_dir=self.processed_cache_dir,
            extension="parquet"
        )
    
    def get_feature_cache_path(self, symbol: str, timeframe: str, 
                             start_date: datetime, end_date: datetime,
                             feature_set: str = "default") -> str:
        """
        Get the path for feature data cache.
        
        Args:
            symbol (str): Trading symbol
            timeframe (str): Timeframe
            start_date (datetime): Start date
            end_date (datetime): End date
            feature_set (str): Feature set name
            
        Returns:
            str: Cache file path
        """
        # Include feature set in the filename
        feature_dir = os.path.join(self.feature_cache_dir, feature_set)
        ensure_directory(feature_dir)
        
        return get_cache_filename(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            cache_dir=feature_dir,
            extension="parquet"
        )
    
    def save_raw_data(self, data: List[Dict], symbol: str, timeframe: str,
                    start_date: datetime, end_date: datetime) -> bool:
        """
        Save raw data to cache.
        
        Args:
            data (List[Dict]): Raw data
            symbol (str): Trading symbol
            timeframe (str): Timeframe
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            bool: Success status
        """
        cache_path = self.get_raw_cache_path(symbol, timeframe, start_date, end_date)
        
        # Create metadata
        metadata = {
            "symbol": symbol,
            "timeframe": timeframe,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "cached_at": datetime.now().isoformat(),
            "records_count": len(data)
        }
        
        # Save data with metadata
        cache_data = {
            "metadata": metadata,
            "data": data
        }
        
        result = save_to_json(cache_data, cache_path)
        
        if result:
            self.logger.debug(f"Saved raw data to cache: {cache_path}")
        
        return result
    
    def load_raw_data(self, symbol: str, timeframe: str,
                    start_date: datetime, end_date: datetime) -> Optional[List[Dict]]:
        """
        Load raw data from cache.
        
        Args:
            symbol (str): Trading symbol
            timeframe (str): Timeframe
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            Optional[List[Dict]]: Raw data or None if not found/expired
        """
        cache_path = self.get_raw_cache_path(symbol, timeframe, start_date, end_date)
        
        # Check if cache exists and is valid
        if self.data_config.use_cache and is_cache_valid(cache_path, self.data_config.cache_expiry_hours):
            cache_data = load_from_json(cache_path)
            
            if cache_data and "data" in cache_data:
                self.cache_hits["raw"] += 1
                self.logger.debug(f"Cache hit (raw): {cache_path}")
                return cache_data["data"]
        
        self.cache_misses["raw"] += 1
        self.logger.debug(f"Cache miss (raw): {cache_path}")
        return None
    
    def save_processed_data(self, df: pd.DataFrame, symbol: str, timeframe: str,
                          start_date: datetime, end_date: datetime) -> bool:
        """
        Save processed data to cache.
        
        Args:
            df (pd.DataFrame): Processed data
            symbol (str): Trading symbol
            timeframe (str): Timeframe
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            bool: Success status
        """
        cache_path = self.get_processed_cache_path(symbol, timeframe, start_date, end_date)
        
        # Add metadata columns
        df = df.copy()
        df['_cached_at'] = datetime.now().isoformat()
        
        result = save_dataframe(df, cache_path, format="parquet")
        
        if result:
            self.logger.debug(f"Saved processed data to cache: {cache_path}")
        
        return result
    
    def load_processed_data(self, symbol: str, timeframe: str,
                          start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Load processed data from cache.
        
        Args:
            symbol (str): Trading symbol
            timeframe (str): Timeframe
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            Optional[pd.DataFrame]: Processed data or None if not found/expired
        """
        cache_path = self.get_processed_cache_path(symbol, timeframe, start_date, end_date)
        
        # Check if cache exists and is valid
        if self.data_config.use_cache and is_cache_valid(cache_path, self.data_config.cache_expiry_hours):
            df = load_dataframe(cache_path)
            
            if df is not None and not df.empty:
                self.cache_hits["processed"] += 1
                self.logger.debug(f"Cache hit (processed): {cache_path}")
                
                # Remove metadata columns
                if '_cached_at' in df.columns:
                    df = df.drop(columns=['_cached_at'])
                
                return df
        
        self.cache_misses["processed"] += 1
        self.logger.debug(f"Cache miss (processed): {cache_path}")
        return None
    
    def save_feature_data(self, df: pd.DataFrame, symbol: str, timeframe: str,
                        start_date: datetime, end_date: datetime,
                        feature_set: str = "default") -> bool:
        """
        Save feature data to cache.
        
        Args:
            df (pd.DataFrame): Feature data
            symbol (str): Trading symbol
            timeframe (str): Timeframe
            start_date (datetime): Start date
            end_date (datetime): End date
            feature_set (str): Feature set name
            
        Returns:
            bool: Success status
        """
        cache_path = self.get_feature_cache_path(
            symbol, timeframe, start_date, end_date, feature_set
        )
        
        # Add metadata columns
        df = df.copy()
        df['_cached_at'] = datetime.now().isoformat()
        df['_feature_set'] = feature_set
        
        result = save_dataframe(df, cache_path, format="parquet")
        
        if result:
            self.logger.debug(f"Saved feature data to cache: {cache_path}")
        
        return result
    
    def load_feature_data(self, symbol: str, timeframe: str,
                        start_date: datetime, end_date: datetime,
                        feature_set: str = "default") -> Optional[pd.DataFrame]:
        """
        Load feature data from cache.
        
        Args:
            symbol (str): Trading symbol
            timeframe (str): Timeframe
            start_date (datetime): Start date
            end_date (datetime): End date
            feature_set (str): Feature set name
            
        Returns:
            Optional[pd.DataFrame]: Feature data or None if not found/expired
        """
        cache_path = self.get_feature_cache_path(
            symbol, timeframe, start_date, end_date, feature_set
        )
        
        # Check if cache exists and is valid
        if self.data_config.use_cache and is_cache_valid(cache_path, self.data_config.cache_expiry_hours):
            df = load_dataframe(cache_path)
            
            if df is not None and not df.empty:
                self.cache_hits["features"] += 1
                self.logger.debug(f"Cache hit (features): {cache_path}")
                
                # Remove metadata columns
                if '_cached_at' in df.columns:
                    df = df.drop(columns=['_cached_at'])
                if '_feature_set' in df.columns:
                    df = df.drop(columns=['_feature_set'])
                
                return df
        
        self.cache_misses["features"] += 1
        self.logger.debug(f"Cache miss (features): {cache_path}")
        return None
    
    def get_cache_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Get cache hit/miss statistics.
        
        Returns:
            Dict[str, Dict[str, int]]: Cache statistics
        """
        stats = {
            "hits": self.cache_hits.copy(),
            "misses": self.cache_misses.copy(),
            "hit_rates": {}
        }
        
        # Calculate hit rates
        for cache_type in self.cache_hits:
            total = self.cache_hits[cache_type] + self.cache_misses[cache_type]
            if total > 0:
                hit_rate = (self.cache_hits[cache_type] / total) * 100
            else:
                hit_rate = 0
            stats["hit_rates"][cache_type] = hit_rate
        
        return stats
    
    def clean_cache(self, max_age_days: int = 30) -> Dict[str, int]:
        """
        Clean old cache files.
        
        Args:
            max_age_days (int): Maximum age of cache files in days
            
        Returns:
            Dict[str, int]: Number of files removed from each cache
        """
        raw_cleaned = clean_old_cache(self.raw_cache_dir, max_age_days)
        processed_cleaned = clean_old_cache(self.processed_cache_dir, max_age_days)
        
        # Clean feature cache (all feature sets)
        feature_cleaned = 0
        for feature_dir in os.listdir(self.feature_cache_dir):
            full_path = os.path.join(self.feature_cache_dir, feature_dir)
            if os.path.isdir(full_path):
                feature_cleaned += clean_old_cache(full_path, max_age_days)
        
        results = {
            "raw": raw_cleaned,
            "processed": processed_cleaned,
            "features": feature_cleaned,
            "total": raw_cleaned + processed_cleaned + feature_cleaned
        }
        
        self.logger.info(f"Cleaned {results['total']} old cache files")
        return results


class DataManager:
    """
    Main class for data acquisition and management.
    
    This class handles fetching data from Binance, caching, and
    preprocessing for use in models.
    """
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """
        Initialize the data manager.
        
        Args:
            api_key (str, optional): Binance API key
            api_secret (str, optional): Binance API secret
        """
        self.logger = get_logger("DataManager")
        
        # Get configuration
        config_manager = ConfigManager()
        self.data_config = config_manager.get_data_config()
        
        # Initialize Binance client
        self.binance_client = BinanceClient(api_key, api_secret)
        
        # Initialize cache manager
        self.cache_manager = DataCacheManager(self.data_config.cache_dir)
    
    def get_historical_data(self, symbol: str, timeframe: str, 
                           lookback_days: Optional[int] = None,
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None,
                           use_cache: Optional[bool] = None) -> pd.DataFrame:
        """
        Get historical market data with intelligent caching.
        
        Args:
            symbol (str): Trading symbol
            timeframe (str): Timeframe
            lookback_days (int, optional): Days to look back. If None, use config default.
            start_date (datetime, optional): Specific start date
            end_date (datetime, optional): Specific end date
            use_cache (bool, optional): Whether to use cache. If None, use config default.
            
        Returns:
            pd.DataFrame: Historical market data
        """
        # Configure cache usage
        use_cache = self.data_config.use_cache if use_cache is None else use_cache
        
        # Determine date range
        if end_date is None:
            end_date = datetime.now()
        
        if start_date is None:
            if lookback_days is None:
                # Use default lookback days from config
                if timeframe in self.data_config.lookback_days:
                    lookback_days = self.data_config.lookback_days[timeframe]
                else:
                    lookback_days = 30  # Default fallback
            
            start_date = end_date - timedelta(days=lookback_days)
        
        self.logger.info(f"Getting historical data for {symbol} {timeframe} "
                        f"from {start_date} to {end_date}")
        
        # Try to load from cache first
        if use_cache:
            df = self.cache_manager.load_processed_data(symbol, timeframe, start_date, end_date)
            if df is not None and not df.empty:
                return df
        
        # Fetch data from Binance
        df = self.binance_client.get_historical_data(symbol, timeframe, (end_date - start_date).days)
        
        if df.empty:
            self.logger.warning(f"No data returned from Binance for {symbol} {timeframe}")
            return df
        
        # Filter by date range
        df = df[(df['date'] >= pd.Timestamp(start_date)) & 
                (df['date'] <= pd.Timestamp(end_date))]
        
        # Save to cache
        if use_cache:
            self.cache_manager.save_processed_data(df, symbol, timeframe, start_date, end_date)
        
        return df
    
    def get_current_price(self, symbol: str) -> float:
        """
        Get current price for a symbol.
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            float: Current price
        """
        return self.binance_client.get_current_price(symbol)
    
    def get_multiple_timeframes(self, symbol: str, timeframes: List[str],
                               lookback_days: Optional[int] = None,
                               start_date: Optional[datetime] = None,
                               end_date: Optional[datetime] = None) -> Dict[str, pd.DataFrame]:
        """
        Get historical data for multiple timeframes.
        
        Args:
            symbol (str): Trading symbol
            timeframes (List[str]): List of timeframes
            lookback_days (int, optional): Days to look back
            start_date (datetime, optional): Specific start date
            end_date (datetime, optional): Specific end date
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of timeframe -> data
        """
        result = {}
        
        for timeframe in timeframes:
            df = self.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                lookback_days=lookback_days,
                start_date=start_date,
                end_date=end_date
            )
            result[timeframe] = df
        
        return result
    
    def clean_cache(self, max_age_days: int = 30) -> Dict[str, int]:
        """
        Clean old cache files.
        
        Args:
            max_age_days (int): Maximum age of cache files in days
            
        Returns:
            Dict[str, int]: Number of files removed from each cache
        """
        return self.cache_manager.clean_cache(max_age_days)
    
    def get_cache_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Get cache hit/miss statistics.
        
        Returns:
            Dict[str, Dict[str, int]]: Cache statistics
        """
        return self.cache_manager.get_cache_stats()


# Example usage
if __name__ == "__main__":
    # Initialize data manager
    data_manager = DataManager()
    
    # Get historical data
    symbol = "BTCUSDT"
    timeframe = "1h"
    lookback_days = 7
    
    df = data_manager.get_historical_data(symbol, timeframe, lookback_days)
    print(f"Retrieved {len(df)} records for {symbol} {timeframe}")
    
    if not df.empty:
        print("\nFirst 5 records:")
        print(df.head())
        
        print("\nLast 5 records:")
        print(df.tail())
        
        print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
    
    # Get cache stats
    stats = data_manager.get_cache_stats()
    print("\nCache stats:")
    print(f"Hit rates: {stats['hit_rates']}")