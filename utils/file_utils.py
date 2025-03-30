"""
File Utilities Module

This module provides utility functions for file operations,
including cache management, file naming, and data serialization.

Functions:
    ensure_directory: Ensure a directory exists
    get_cache_filename: Generate standardized cache filenames
    save_to_json: Save data to a JSON file
    load_from_json: Load data from a JSON file
    is_cache_valid: Check if a cache file is still valid
    get_latest_file: Get the latest file in a directory
"""

import os
import json
import glob
import time
import shutil
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta

from utils.logger import get_logger

# Setup logger
logger = get_logger(__name__)


def ensure_directory(directory: str) -> bool:
    """
    Ensure a directory exists, creating it if needed.
    
    Args:
        directory (str): Directory path
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        os.makedirs(directory, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {directory}: {e}")
        return False


def get_cache_filename(symbol: str, timeframe: str, start_date: datetime, 
                      end_date: datetime, cache_dir: str = "cache", 
                      extension: str = "json") -> str:
    """
    Generate a standardized cache filename.
    
    Args:
        symbol (str): Trading symbol (e.g., 'BTCUSDT')
        timeframe (str): Timeframe (e.g., '1m', '1h')
        start_date (datetime): Start date
        end_date (datetime): End date
        cache_dir (str): Cache directory
        extension (str): File extension
        
    Returns:
        str: Full path to the cache file
    """
    # Format dates for filename
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    
    # Create filename
    filename = f"{symbol}_{timeframe}_{start_str}_{end_str}.{extension}"
    
    # Ensure cache directory exists
    ensure_directory(cache_dir)
    
    # Return full path
    return os.path.join(cache_dir, filename)


def save_to_json(data: Any, filepath: str, pretty: bool = True) -> bool:
    """
    Save data to a JSON file.
    
    Args:
        data (Any): Data to save
        filepath (str): Path to save the file
        pretty (bool): Whether to format the JSON prettily
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        directory = os.path.dirname(filepath)
        ensure_directory(directory)
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(data, f, indent=4, sort_keys=True)
            else:
                json.dump(data, f)
        
        logger.debug(f"Saved data to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to save data to {filepath}: {e}")
        return False


def load_from_json(filepath: str, default: Any = None) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        filepath (str): Path to the file
        default (Any): Default value to return if loading fails
        
    Returns:
        Any: Loaded data or default value
    """
    if not os.path.exists(filepath):
        logger.debug(f"File {filepath} does not exist")
        return default
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.debug(f"Loaded data from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Failed to load data from {filepath}: {e}")
        return default


def is_cache_valid(filepath: str, expiry_hours: int = 24) -> bool:
    """
    Check if a cache file is still valid based on its age.
    
    Args:
        filepath (str): Path to the cache file
        expiry_hours (int): Cache expiry time in hours
        
    Returns:
        bool: True if cache is valid, False otherwise
    """
    if not os.path.exists(filepath):
        return False
    
    # Get file modification time
    file_mtime = os.path.getmtime(filepath)
    file_datetime = datetime.fromtimestamp(file_mtime)
    
    # Check if file is older than expiry time
    age = datetime.now() - file_datetime
    return age.total_seconds() < expiry_hours * 3600


def get_latest_file(directory: str, pattern: str = "*") -> Optional[str]:
    """
    Get the latest (most recently modified) file in a directory.
    
    Args:
        directory (str): Directory to search
        pattern (str): Glob pattern for filtering files
        
    Returns:
        Optional[str]: Path to the latest file, or None if no files found
    """
    if not os.path.exists(directory):
        return None
    
    # Get all files matching the pattern
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        return None
    
    # Find the most recently modified file
    latest_file = max(files, key=os.path.getmtime)
    return latest_file


def clean_old_cache(cache_dir: str, max_age_days: int = 30) -> int:
    """
    Remove cache files older than a specified age.
    
    Args:
        cache_dir (str): Cache directory
        max_age_days (int): Maximum age in days
        
    Returns:
        int: Number of files removed
    """
    if not os.path.exists(cache_dir):
        return 0
    
    # Get current time
    now = time.time()
    cutoff = now - (max_age_days * 24 * 3600)
    
    # Track number of files removed
    removed_count = 0
    
    # Iterate through files in directory
    for filename in os.listdir(cache_dir):
        filepath = os.path.join(cache_dir, filename)
        
        # Skip directories
        if os.path.isdir(filepath):
            continue
        
        # Check file age
        file_mtime = os.path.getmtime(filepath)
        if file_mtime < cutoff:
            try:
                os.remove(filepath)
                removed_count += 1
                logger.debug(f"Removed old cache file: {filepath}")
            except Exception as e:
                logger.error(f"Failed to remove cache file {filepath}: {e}")
    
    logger.info(f"Cleaned {removed_count} old cache files from {cache_dir}")
    return removed_count


def save_dataframe(df, filepath: str, format: str = "csv") -> bool:
    """
    Save a pandas DataFrame to file.
    
    Args:
        df: Pandas DataFrame
        filepath (str): Path to save the file
        format (str): File format ('csv', 'parquet', or 'pickle')
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        directory = os.path.dirname(filepath)
        ensure_directory(directory)
        
        # Save based on format
        if format.lower() == "csv":
            df.to_csv(filepath, index=False)
        elif format.lower() == "parquet":
            df.to_parquet(filepath, index=False)
        elif format.lower() == "pickle":
            df.to_pickle(filepath)
        else:
            logger.error(f"Unsupported format: {format}")
            return False
        
        logger.debug(f"Saved DataFrame to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to save DataFrame to {filepath}: {e}")
        return False


def load_dataframe(filepath: str, format: str = None) -> Optional[Any]:
    """
    Load a pandas DataFrame from file.
    
    Args:
        filepath (str): Path to the file
        format (str, optional): File format. If None, inferred from extension.
        
    Returns:
        Optional[DataFrame]: Loaded DataFrame or None if loading fails
    """
    import pandas as pd
    
    if not os.path.exists(filepath):
        logger.debug(f"File {filepath} does not exist")
        return None
    
    try:
        # Determine format from extension if not provided
        if format is None:
            _, ext = os.path.splitext(filepath)
            ext = ext.lower().lstrip('.')
            if ext in ["csv"]:
                format = "csv"
            elif ext in ["parquet", "pq"]:
                format = "parquet"
            elif ext in ["pkl", "pickle"]:
                format = "pickle"
            else:
                logger.warning(f"Could not determine format from extension: {ext}")
                return None
        
        # Load based on format
        if format.lower() == "csv":
            df = pd.read_csv(filepath)
        elif format.lower() == "parquet":
            df = pd.read_parquet(filepath)
        elif format.lower() == "pickle":
            df = pd.read_pickle(filepath)
        else:
            logger.error(f"Unsupported format: {format}")
            return None
        
        logger.debug(f"Loaded DataFrame from {filepath}")
        return df
    except Exception as e:
        logger.error(f"Failed to load DataFrame from {filepath}: {e}")
        return None


# Example usage
if __name__ == "__main__":
    # Test ensure_directory
    test_dir = "test_cache"
    print(f"Creating directory: {ensure_directory(test_dir)}")
    
    # Test get_cache_filename
    start_date = datetime.now() - timedelta(days=7)
    end_date = datetime.now()
    cache_file = get_cache_filename("BTCUSDT", "1h", start_date, end_date, test_dir)
    print(f"Cache filename: {cache_file}")
    
    # Test save_to_json and load_from_json
    test_data = {
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "data": [{"price": 50000, "time": "2023-01-01T00:00:00"}]
    }
    save_result = save_to_json(test_data, cache_file)
    print(f"Save result: {save_result}")
    
    loaded_data = load_from_json(cache_file)
    print(f"Loaded data matches: {loaded_data['symbol'] == test_data['symbol']}")
    
    # Test is_cache_valid
    is_valid = is_cache_valid(cache_file)
    print(f"Cache is valid: {is_valid}")
    
    # Clean up
    clean_result = clean_old_cache(test_dir, max_age_days=0)  # Clean immediately
    print(f"Cleaned {clean_result} files")
    
    # Remove test directory
    shutil.rmtree(test_dir, ignore_errors=True)