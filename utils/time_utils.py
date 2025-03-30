"""
Time Utilities Module

This module provides utility functions for handling time-related operations,
particularly useful for working with market data and timeframes.

Functions:
    get_timestamp_ms: Convert a datetime to timestamp in milliseconds
    get_datetime_from_ms: Convert a millisecond timestamp to datetime
    timeframe_to_seconds: Convert a timeframe string to seconds
    get_previous_timeframe: Get the start of the previous timeframe
    align_time_to_timeframe: Align a datetime to the start of a timeframe
    get_timeframe_divisions: Get the number of divisions of a timeframe in a day
"""

import datetime
from typing import Union, Optional, Dict, Tuple, List


def get_timestamp_ms(dt: Optional[datetime.datetime] = None) -> int:
    """
    Convert a datetime to timestamp in milliseconds.
    
    Args:
        dt (datetime.datetime, optional): Datetime to convert. If None, use current time.
        
    Returns:
        int: Timestamp in milliseconds
    """
    if dt is None:
        dt = datetime.datetime.now()
    return int(dt.timestamp() * 1000)


def get_datetime_from_ms(timestamp_ms: int) -> datetime.datetime:
    """
    Convert a millisecond timestamp to datetime.
    
    Args:
        timestamp_ms (int): Timestamp in milliseconds
        
    Returns:
        datetime.datetime: Datetime object
    """
    return datetime.datetime.fromtimestamp(timestamp_ms / 1000)


def timeframe_to_seconds(timeframe: str) -> int:
    """
    Convert a timeframe string to seconds.
    
    Args:
        timeframe (str): Timeframe string (e.g., '1m', '1h', '1d')
        
    Returns:
        int: Timeframe in seconds
        
    Raises:
        ValueError: If the timeframe is invalid
    """
    # Extract the numeric part and unit
    if not timeframe:
        raise ValueError("Timeframe cannot be empty")
    
    # Handle special case for timeframes like "30m"
    if timeframe[0].isdigit():
        for i, char in enumerate(timeframe):
            if not char.isdigit():
                value = int(timeframe[:i])
                unit = timeframe[i:]
                break
        else:
            raise ValueError(f"Invalid timeframe format: {timeframe}")
    else:
        value = 1
        unit = timeframe
    
    # Convert unit to seconds
    if unit == 'm':
        return value * 60
    elif unit == 'h':
        return value * 60 * 60
    elif unit == 'd':
        return value * 24 * 60 * 60
    elif unit == 'w':
        return value * 7 * 24 * 60 * 60
    else:
        raise ValueError(f"Invalid timeframe unit: {unit}")


def get_previous_timeframe(dt: Optional[datetime.datetime] = None, 
                           timeframe: str = '1h') -> datetime.datetime:
    """
    Get the start of the previous timeframe.
    
    Args:
        dt (datetime.datetime, optional): Reference datetime. If None, use current time.
        timeframe (str): Timeframe string (e.g., '1m', '1h', '1d')
        
    Returns:
        datetime.datetime: Start of the previous timeframe
    """
    if dt is None:
        dt = datetime.datetime.now()
    
    # Convert timeframe to seconds
    seconds = timeframe_to_seconds(timeframe)
    
    # Align to current timeframe
    current_timeframe = align_time_to_timeframe(dt, timeframe)
    
    # Go back one timeframe
    previous_timeframe = current_timeframe - datetime.timedelta(seconds=seconds)
    
    return previous_timeframe


def align_time_to_timeframe(dt: datetime.datetime, timeframe: str) -> datetime.datetime:
    """
    Align a datetime to the start of a timeframe.
    
    Args:
        dt (datetime.datetime): Datetime to align
        timeframe (str): Timeframe string (e.g., '1m', '1h', '1d')
        
    Returns:
        datetime.datetime: Aligned datetime
    """
    seconds = timeframe_to_seconds(timeframe)
    timestamp = int(dt.timestamp())
    aligned_timestamp = timestamp - (timestamp % seconds)
    return datetime.datetime.fromtimestamp(aligned_timestamp)


def get_timeframe_divisions(timeframe: str) -> int:
    """
    Get the number of divisions of a timeframe in a day.
    
    Args:
        timeframe (str): Timeframe string (e.g., '1m', '1h', '1d')
        
    Returns:
        int: Number of divisions in a day
    """
    seconds = timeframe_to_seconds(timeframe)
    day_seconds = 24 * 60 * 60
    return day_seconds // seconds


def parse_timeframe_and_convert(source_tf: str, 
                                target_tf: Optional[str] = None) -> Tuple[int, Optional[int]]:
    """
    Parse timeframe string and convert to numeric values.
    
    Args:
        source_tf (str): Source timeframe (e.g., '1m', '1h')
        target_tf (str, optional): Target timeframe
    
    Returns:
        Tuple[int, Optional[int]]: Source value in minutes, target value in minutes
    """
    def parse_tf(tf):
        if tf.endswith('m'):
            return int(tf[:-1])
        elif tf.endswith('h'):
            return int(tf[:-1]) * 60
        elif tf.endswith('d'):
            return int(tf[:-1]) * 60 * 24
        else:
            raise ValueError(f"Invalid timeframe format: {tf}")
    
    source_minutes = parse_tf(source_tf)
    target_minutes = parse_tf(target_tf) if target_tf else None
    
    return source_minutes, target_minutes


def is_higher_timeframe(higher_tf: str, lower_tf: str) -> bool:
    """
    Check if one timeframe is higher than another.
    
    Args:
        higher_tf (str): The supposedly higher timeframe
        lower_tf (str): The supposedly lower timeframe
        
    Returns:
        bool: True if higher_tf is higher than lower_tf
    """
    higher_seconds = timeframe_to_seconds(higher_tf)
    lower_seconds = timeframe_to_seconds(lower_tf)
    return higher_seconds > lower_seconds


def get_common_timeframe_start(dt: datetime.datetime, timeframe: str) -> datetime.datetime:
    """
    Get the start time of the current timeframe that contains the given datetime.
    
    Args:
        dt (datetime.datetime): The datetime to check
        timeframe (str): Timeframe string
        
    Returns:
        datetime.datetime: Start time of the timeframe
    """
    return align_time_to_timeframe(dt, timeframe)


def get_binance_interval(timeframe: str) -> str:
    """
    Convert timeframe string to Binance API interval string.
    
    Args:
        timeframe (str): Timeframe string (e.g., '1m', '1h', '1d')
        
    Returns:
        str: Binance interval string
        
    Raises:
        ValueError: If the timeframe is not supported by Binance
    """
    # Binance supported intervals
    # https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md#klinecandlestick-data
    supported_intervals = {
        '1m': '1m', '3m': '3m', '5m': '5m', '15m': '15m', '30m': '30m',
        '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '8h': '8h', '12h': '12h',
        '1d': '1d', '3d': '3d', '1w': '1w', '1M': '1M'
    }
    
    if timeframe not in supported_intervals:
        raise ValueError(f"Timeframe {timeframe} is not supported by Binance API. "
                         f"Supported intervals: {list(supported_intervals.keys())}")
    
    return supported_intervals[timeframe]


def get_date_range(start_date: Optional[datetime.datetime] = None,
                  end_date: Optional[datetime.datetime] = None,
                  days_back: int = 30) -> Tuple[datetime.datetime, datetime.datetime]:
    """
    Get a date range with sensible defaults.
    
    Args:
        start_date (datetime.datetime, optional): Start date. If None, calculated from end_date and days_back.
        end_date (datetime.datetime, optional): End date. If None, uses current time.
        days_back (int): Days to look back from end_date if start_date is None.
        
    Returns:
        Tuple[datetime.datetime, datetime.datetime]: Start and end dates
    """
    if end_date is None:
        end_date = datetime.datetime.now()
    
    if start_date is None:
        start_date = end_date - datetime.timedelta(days=days_back)
    
    return start_date, end_date


def time_windows_overlap(start1: datetime.datetime, end1: datetime.datetime,
                        start2: datetime.datetime, end2: datetime.datetime) -> bool:
    """
    Check if two time windows overlap.
    
    Args:
        start1 (datetime.datetime): Start of first window
        end1 (datetime.datetime): End of first window
        start2 (datetime.datetime): Start of second window
        end2 (datetime.datetime): End of second window
        
    Returns:
        bool: True if windows overlap
    """
    return max(start1, start2) < min(end1, end2)


def calculate_missing_periods(available_start: datetime.datetime, available_end: datetime.datetime,
                             requested_start: datetime.datetime, requested_end: datetime.datetime,
                             timeframe: str) -> List[Tuple[datetime.datetime, datetime.datetime]]:
    """
    Calculate missing periods between available and requested time ranges.
    
    Args:
        available_start (datetime.datetime): Start of available data
        available_end (datetime.datetime): End of available data
        requested_start (datetime.datetime): Start of requested data
        requested_end (datetime.datetime): End of requested data
        timeframe (str): Timeframe for the data
        
    Returns:
        List[Tuple[datetime.datetime, datetime.datetime]]: List of (start, end) tuples for missing periods
    """
    missing_periods = []
    
    # Check if there's missing data at the beginning
    if requested_start < available_start:
        missing_periods.append((requested_start, available_start))
    
    # Check if there's missing data at the end
    if requested_end > available_end:
        missing_periods.append((available_end, requested_end))
    
    return missing_periods


# Example usage
if __name__ == "__main__":
    # Test timeframe_to_seconds
    print(f"1m in seconds: {timeframe_to_seconds('1m')}")
    print(f"5m in seconds: {timeframe_to_seconds('5m')}")
    print(f"1h in seconds: {timeframe_to_seconds('1h')}")
    print(f"1d in seconds: {timeframe_to_seconds('1d')}")
    
    # Test get_previous_timeframe
    now = datetime.datetime.now()
    print(f"Current time: {now}")
    print(f"Previous 1m timeframe: {get_previous_timeframe(now, '1m')}")
    print(f"Previous 1h timeframe: {get_previous_timeframe(now, '1h')}")
    
    # Test align_time_to_timeframe
    print(f"Aligned to 1h: {align_time_to_timeframe(now, '1h')}")
    
    # Test get_timeframe_divisions
    print(f"1m divisions in a day: {get_timeframe_divisions('1m')}")
    print(f"1h divisions in a day: {get_timeframe_divisions('1h')}")
    print(f"1d divisions in a day: {get_timeframe_divisions('1d')}")