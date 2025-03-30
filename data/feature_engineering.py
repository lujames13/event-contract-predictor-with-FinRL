"""
Feature Engineering Module

This module calculates technical indicators and transforms raw market data
into features suitable for machine learning models.

Classes:
    TechnicalIndicators: Class for calculating technical indicators
    FeatureGenerator: Class for generating feature sets
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Any
from datetime import datetime

from utils.logger import get_logger

# Setup logger
logger = get_logger(__name__)


class TechnicalIndicators:
    """
    Class for calculating technical indicators on price data.
    
    This class provides methods for calculating various technical indicators
    commonly used in trading, such as moving averages, RSI, MACD, etc.
    """
    
    @staticmethod
    def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all standard technical indicators to the dataframe.
        
        This method acts as a facade to apply multiple technical indicators at once.
        
        Args:
            df (pd.DataFrame): Price dataframe
            
        Returns:
            pd.DataFrame: DataFrame with all standard technical indicators added
        """
        logger.info("Adding all standard technical indicators")
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Add moving averages
        df = TechnicalIndicators.add_ma(df, periods=[7, 25, 99])
        
        # Add RSI
        df = TechnicalIndicators.add_rsi(df)
        
        # Add MACD
        df = TechnicalIndicators.add_macd(df)
        
        # Add Bollinger Bands
        df = TechnicalIndicators.add_bollinger_bands(df)
        
        # Add ATR
        df = TechnicalIndicators.add_atr(df)
        
        # Add Stochastic Oscillator
        df = TechnicalIndicators.add_stochastic(df)
        
        # Add OBV
        df = TechnicalIndicators.add_obv(df)
        
        logger.info(f"Added {len(df.columns)} columns with technical indicators")
        
        return df
    
    @staticmethod
    def add_ma(df: pd.DataFrame, periods: List[int] = [7, 25, 99], 
              column: str = 'close') -> pd.DataFrame:
        """
        Add Moving Average indicators to the dataframe.
        
        Args:
            df (pd.DataFrame): Price dataframe
            periods (List[int]): List of periods for moving averages
            column (str): Column to calculate MA on
            
        Returns:
            pd.DataFrame: DataFrame with added MA columns
        """
        df = df.copy()
        
        for period in periods:
            df[f'ma{period}'] = df[column].rolling(window=period, min_periods=1).mean()
        
        return df
    
    @staticmethod
    def add_ema(df: pd.DataFrame, periods: List[int] = [12, 26], 
               column: str = 'close') -> pd.DataFrame:
        """
        Add Exponential Moving Average indicators to the dataframe.
        
        Args:
            df (pd.DataFrame): Price dataframe
            periods (List[int]): List of periods for exponential moving averages
            column (str): Column to calculate EMA on
            
        Returns:
            pd.DataFrame: DataFrame with added EMA columns
        """
        df = df.copy()
        
        for period in periods:
            df[f'ema{period}'] = df[column].ewm(span=period, adjust=False).mean()
        
        return df
    
    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = 14, 
               column: str = 'close') -> pd.DataFrame:
        """
        Add Relative Strength Index (RSI) indicator to the dataframe.
        
        Args:
            df (pd.DataFrame): Price dataframe
            period (int): Period for RSI calculation
            column (str): Column to calculate RSI on
            
        Returns:
            pd.DataFrame: DataFrame with added RSI column
        """
        df = df.copy()
        
        # Calculate price changes
        delta = df[column].diff()
        
        # Calculate gain and loss
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        
        # Calculate RS
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        df[f'rsi{period}'] = 100 - (100 / (1 + rs))
        
        # Handle divide by zero
        df[f'rsi{period}'] = df[f'rsi{period}'].replace([np.inf, -np.inf], np.nan)
        df[f'rsi{period}'] = df[f'rsi{period}'].fillna(50)  # Fill NaN with neutral value
        
        return df

    @staticmethod
    def add_macd(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26,
                signal_period: int = 9, column: str = 'close') -> pd.DataFrame:
        """
        Add Moving Average Convergence Divergence (MACD) indicator to the dataframe.
        
        Args:
            df (pd.DataFrame): Price dataframe
            fast_period (int): Fast EMA period
            slow_period (int): Slow EMA period
            signal_period (int): Signal line period
            column (str): Column to calculate MACD on
            
        Returns:
            pd.DataFrame: DataFrame with added MACD columns
        """
        df = df.copy()
        
        # Calculate fast and slow EMAs
        fast_ema = df[column].ewm(span=fast_period, adjust=False).mean()
        slow_ema = df[column].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        df['macd'] = fast_ema - slow_ema
        
        # Calculate signal line
        df['macd_signal'] = df['macd'].ewm(span=signal_period, adjust=False).mean()
        
        # Calculate histogram
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        return df

    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame, period: int = 20,
                        std_dev: float = 2.0, column: str = 'close') -> pd.DataFrame:
        """
        Add Bollinger Bands to the dataframe.
        
        Args:
            df (pd.DataFrame): Price dataframe
            period (int): Period for moving average
            std_dev (float): Number of standard deviations
            column (str): Column to calculate Bollinger Bands on
            
        Returns:
            pd.DataFrame: DataFrame with added Bollinger Bands columns
        """
        df = df.copy()
        
        # Calculate middle band (SMA)
        df['bb_middle'] = df[column].rolling(window=period, min_periods=1).mean()
        
        # Calculate standard deviation
        rolling_std = df[column].rolling(window=period, min_periods=1).std()
        
        # Calculate upper and lower bands
        df['bb_upper'] = df['bb_middle'] + (rolling_std * std_dev)
        df['bb_lower'] = df['bb_middle'] - (rolling_std * std_dev)
        
        # Calculate bandwidth and %B
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_pctb'] = (df[column] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Handle divide by zero
        df['bb_width'] = df['bb_width'].replace([np.inf, -np.inf], np.nan)
        df['bb_pctb'] = df['bb_pctb'].replace([np.inf, -np.inf], np.nan)
        df['bb_width'] = df['bb_width'].fillna(0)
        df['bb_pctb'] = df['bb_pctb'].fillna(0.5)  # Fill NaN with middle value
        
        return df

    @staticmethod
    def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Add Average True Range (ATR) indicator to the dataframe.
        
        Args:
            df (pd.DataFrame): Price dataframe
            period (int): Period for ATR calculation
            
        Returns:
            pd.DataFrame: DataFrame with added ATR column
        """
        df = df.copy()
        
        # Calculate true range
        df['tr1'] = abs(df['high'] - df['low'])
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate ATR
        df[f'atr{period}'] = df['tr'].rolling(window=period, min_periods=1).mean()
        
        # Drop temporary columns
        df = df.drop(['tr1', 'tr2', 'tr3', 'tr'], axis=1)
        
        return df

    @staticmethod
    def add_stochastic(df: pd.DataFrame, k_period: int = 14,
                    d_period: int = 3) -> pd.DataFrame:
        """
        Add Stochastic Oscillator to the dataframe.
        
        Args:
            df (pd.DataFrame): Price dataframe
            k_period (int): Period for %K calculation
            d_period (int): Period for %D calculation
            
        Returns:
            pd.DataFrame: DataFrame with added %K and %D columns
        """
        df = df.copy()
        
        # Calculate %K
        low_min = df['low'].rolling(window=k_period, min_periods=1).min()
        high_max = df['high'].rolling(window=k_period, min_periods=1).max()
        
        df['stoch_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
        
        # Handle divide by zero
        df['stoch_k'] = df['stoch_k'].replace([np.inf, -np.inf], np.nan)
        df['stoch_k'] = df['stoch_k'].fillna(50)  # Fill NaN with neutral value
        
        # Calculate %D
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period, min_periods=1).mean()
        
        return df

    @staticmethod
    def add_cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Add Commodity Channel Index (CCI) indicator to the dataframe.
        
        Args:
            df (pd.DataFrame): Price dataframe
            period (int): Period for CCI calculation
            
        Returns:
            pd.DataFrame: DataFrame with added CCI column
        """
        df = df.copy()
        
        # Calculate typical price
        df['tp'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate SMA of typical price
        df['tp_sma'] = df['tp'].rolling(window=period, min_periods=1).mean()
        
        # Calculate mean deviation
        df['md'] = df['tp'].rolling(window=period, min_periods=1).apply(
            lambda x: np.mean(np.abs(x - np.mean(x)))
        )
        
        # Calculate CCI
        df[f'cci{period}'] = (df['tp'] - df['tp_sma']) / (0.015 * df['md'])
        
        # Handle divide by zero
        df[f'cci{period}'] = df[f'cci{period}'].replace([np.inf, -np.inf], np.nan)
        df[f'cci{period}'] = df[f'cci{period}'].fillna(0)  # Fill NaN with neutral value
        
        # Drop temporary columns
        df = df.drop(['tp', 'tp_sma', 'md'], axis=1)
        
        return df

    @staticmethod
    def add_obv(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add On-Balance Volume (OBV) indicator to the dataframe.
        
        Args:
            df (pd.DataFrame): Price dataframe
            
        Returns:
            pd.DataFrame: DataFrame with added OBV column
        """
        df = df.copy()
        
        # Calculate price change direction
        price_change = df['close'].diff()
        
        # Initialize OBV
        df['obv'] = 0.0
        
        # First row of OBV is same as volume
        if len(df) > 0:
            df.loc[0, 'obv'] = df.loc[0, 'volume']
        
        # Calculate OBV
        for i in range(1, len(df)):
            if price_change.iloc[i] > 0:
                df.loc[i, 'obv'] = df.loc[i-1, 'obv'] + df.loc[i, 'volume']
            elif price_change.iloc[i] < 0:
                df.loc[i, 'obv'] = df.loc[i-1, 'obv'] - df.loc[i, 'volume']
            else:
                df.loc[i, 'obv'] = df.loc[i-1, 'obv']
        
        return df

    @staticmethod
    def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Add Average Directional Index (ADX) indicator to the dataframe.
        
        Args:
            df (pd.DataFrame): Price dataframe
            period (int): Period for ADX calculation
            
        Returns:
            pd.DataFrame: DataFrame with added ADX, +DI, and -DI columns
        """
        df = df.copy()
        
        # Calculate +DM and -DM
        df['up_move'] = df['high'].diff()
        df['down_move'] = df['low'].shift(1) - df['low']
        
        # Calculate +DM
        df['plus_dm'] = 0.0
        df.loc[(df['up_move'] > df['down_move']) & (df['up_move'] > 0), 'plus_dm'] = df['up_move']
        
        # Calculate -DM
        df['minus_dm'] = 0.0
        df.loc[(df['down_move'] > df['up_move']) & (df['down_move'] > 0), 'minus_dm'] = df['down_move']
        
        # Calculate ATR
        df = TechnicalIndicators.add_atr(df, period)
        atr_col = f'atr{period}'
        
        # Calculate smoothed +DM and -DM
        df['plus_di'] = 100 * (df['plus_dm'].ewm(alpha=1/period, adjust=False).mean() / df[atr_col])
        df['minus_di'] = 100 * (df['minus_dm'].ewm(alpha=1/period, adjust=False).mean() / df[atr_col])
        
        # Calculate DX
        df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        
        # Handle divide by zero
        df['dx'] = df['dx'].replace([np.inf, -np.inf], np.nan)
        df['dx'] = df['dx'].fillna(0)
        
        # Calculate ADX
        df[f'adx{period}'] = df['dx'].ewm(alpha=1/period, adjust=False).mean()
        
        # Drop temporary columns
        df = df.drop(['up_move', 'down_move', 'plus_dm', 'minus_dm', 'dx'], axis=1)
        
        return df

    @staticmethod
    def add_roc(df: pd.DataFrame, period: int = 12, column: str = 'close') -> pd.DataFrame:
        """
        Add Rate of Change (ROC) indicator to the dataframe.
        
        Args:
            df (pd.DataFrame): Price dataframe
            period (int): Period for ROC calculation
            column (str): Column to calculate ROC on
            
        Returns:
            pd.DataFrame: DataFrame with added ROC column
        """
        df = df.copy()
        df[f'roc{period}'] = ((df[column] - df[column].shift(period)) / df[column].shift(period)) * 100
        
        # Handle divide by zero
        df[f'roc{period}'] = df[f'roc{period}'].replace([np.inf, -np.inf], np.nan)
        df[f'roc{period}'] = df[f'roc{period}'].fillna(0)  # Fill NaN with neutral value
        
        return df

    @staticmethod
    def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Volume Weighted Average Price (VWAP) indicator to the dataframe.
        
        Args:
            df (pd.DataFrame): Price dataframe with date column
            
        Returns:
            pd.DataFrame: DataFrame with added VWAP column
        """
        df = df.copy()
        
        # Calculate typical price
        df['tp'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate cumulative values
        df['cumulative_tp_vol'] = (df['tp'] * df['volume']).cumsum()
        df['cumulative_vol'] = df['volume'].cumsum()
        
        # Calculate VWAP
        df['vwap'] = df['cumulative_tp_vol'] / df['cumulative_vol']
        
        # Handle divide by zero
        df['vwap'] = df['vwap'].replace([np.inf, -np.inf], np.nan)
        df['vwap'] = df['vwap'].fillna(df['close'])  # Fill NaN with close price
        
        # Drop temporary columns
        df = df.drop(['tp', 'cumulative_tp_vol', 'cumulative_vol'], axis=1)
        
        return df

    @staticmethod
    def add_ichimoku(df: pd.DataFrame, tenkan_period: int = 9,
                    kijun_period: int = 26, senkou_b_period: int = 52,
                    displacement: int = 26) -> pd.DataFrame:
        """
        Add Ichimoku Cloud indicator to the dataframe.
        
        Args:
            df (pd.DataFrame): Price dataframe
            tenkan_period (int): Tenkan-sen (Conversion Line) period
            kijun_period (int): Kijun-sen (Base Line) period
            senkou_b_period (int): Senkou Span B period
            displacement (int): Displacement period
            
        Returns:
            pd.DataFrame: DataFrame with added Ichimoku Cloud columns
        """
        df = df.copy()
        
        # Calculate Tenkan-sen (Conversion Line)
        high_tenkan = df['high'].rolling(window=tenkan_period, min_periods=1).max()
        low_tenkan = df['low'].rolling(window=tenkan_period, min_periods=1).min()
        df['tenkan_sen'] = (high_tenkan + low_tenkan) / 2
        
        # Calculate Kijun-sen (Base Line)
        high_kijun = df['high'].rolling(window=kijun_period, min_periods=1).max()
        low_kijun = df['low'].rolling(window=kijun_period, min_periods=1).min()
        df['kijun_sen'] = (high_kijun + low_kijun) / 2
        
        # Calculate Senkou Span A (Leading Span A)
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(displacement)
        
        # Calculate Senkou Span B (Leading Span B)
        high_senkou = df['high'].rolling(window=senkou_b_period, min_periods=1).max()
        low_senkou = df['low'].rolling(window=senkou_b_period, min_periods=1).min()
        df['senkou_span_b'] = ((high_senkou + low_senkou) / 2).shift(displacement)
        
        # Calculate Chikou Span (Lagging Span)
        df['chikou_span'] = df['close'].shift(-displacement)
        
        return df


class FeatureGenerator:
    """
    Class for generating feature sets from raw price data.
    
    This class applies technical indicators and transforms raw data
    into feature sets suitable for machine learning models.
    """
    
    def __init__(self):
        """Initialize the feature generator."""
        self.logger = get_logger("FeatureGenerator")
        self.ti = TechnicalIndicators()
    
    def generate_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a basic set of features.
        
        Args:
            df (pd.DataFrame): Raw price dataframe
            
        Returns:
            pd.DataFrame: DataFrame with basic features
        """
        if df.empty:
            self.logger.warning("Empty dataframe provided to generate_basic_features")
            return df
        
        self.logger.info(f"Generating basic features for {len(df)} records")
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Ensure we have required columns
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                self.logger.error(f"Required column {col} not found in dataframe")
                raise ValueError(f"Required column {col} not found in dataframe")
        
        # Add moving averages
        df = TechnicalIndicators.add_ma(df, periods=[7, 25, 99])
        
        # Add RSI
        df = TechnicalIndicators.add_rsi(df, period=14)
        
        # Add MACD
        df = TechnicalIndicators.add_macd(df)
        
        # Add Bollinger Bands
        df = TechnicalIndicators.add_bollinger_bands(df)
        
        # Add ATR
        df = TechnicalIndicators.add_atr(df, period=14)
        
        self.logger.info(f"Generated {len(df.columns) - len(required_columns)} basic features")
        
        return df
    
    def generate_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate an advanced set of features.
        
        Args:
            df (pd.DataFrame): Raw price dataframe
            
        Returns:
            pd.DataFrame: DataFrame with advanced features
        """
        if df.empty:
            self.logger.warning("Empty dataframe provided to generate_advanced_features")
            return df
        
        self.logger.info(f"Generating advanced features for {len(df)} records")
        
        # First apply basic features
        df = self.generate_basic_features(df)
        
        # Add Stochastic Oscillator
        df = TechnicalIndicators.add_stochastic(df)
        
        # Add CCI
        df = TechnicalIndicators.add_cci(df)
        
        # Add OBV
        df = TechnicalIndicators.add_obv(df)
        
        # Add ADX
        df = TechnicalIndicators.add_adx(df)
        
        # Add ROC
        df = TechnicalIndicators.add_roc(df, period=12)
        
        # Add VWAP
        df = TechnicalIndicators.add_vwap(df)
        
        # Add Ichimoku Cloud
        df = TechnicalIndicators.add_ichimoku(df)
        
        return df
    
    def generate_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate price-based features.
        
        Args:
            df (pd.DataFrame): Raw price dataframe
            
        Returns:
            pd.DataFrame: DataFrame with price-based features
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Calculate price changes
        df['price_change'] = df['close'].diff()
        df['price_change_pct'] = df['close'].pct_change() * 100
        
        # Calculate returns
        df['daily_return'] = df['close'].pct_change()
        
        # Calculate price volatility (standard deviation of returns)
        df['volatility_5'] = df['daily_return'].rolling(window=5).std()
        df['volatility_15'] = df['daily_return'].rolling(window=15).std()
        
        # Calculate normalized price
        min_price = df['close'].rolling(window=20).min()
        max_price = df['close'].rolling(window=20).max()
        df['normalized_price'] = (df['close'] - min_price) / (max_price - min_price)
        
        # Handle divide by zero
        df['normalized_price'] = df['normalized_price'].replace([np.inf, -np.inf], np.nan)
        df['normalized_price'] = df['normalized_price'].fillna(0.5)
        
        # Calculate candle features
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        df['body_to_range'] = df['body_size'] / (df['high'] - df['low'])
        
        # Handle divide by zero
        df['body_to_range'] = df['body_to_range'].replace([np.inf, -np.inf], np.nan)
        df['body_to_range'] = df['body_to_range'].fillna(0)
        
        # Identify doji candles (open and close are almost the same)
        df['is_doji'] = df['body_size'] < (0.1 * (df['high'] - df['low']))
        
        # Identify bullish and bearish candles
        df['is_bullish'] = df['close'] > df['open']
        df['is_bearish'] = df['close'] < df['open']
        
        return df
    
    def generate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate volume-based features.
        
        Args:
            df (pd.DataFrame): Raw price dataframe
            
        Returns:
            pd.DataFrame: DataFrame with volume-based features
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Calculate volume changes
        df['volume_change'] = df['volume'].diff()
        df['volume_change_pct'] = df['volume'].pct_change() * 100
        
        # Calculate volume moving averages
        df['volume_ma5'] = df['volume'].rolling(window=5).mean()
        df['volume_ma20'] = df['volume'].rolling(window=20).mean()
        
        # Calculate volume relative to moving average
        df['volume_ratio_5'] = df['volume'] / df['volume_ma5']
        df['volume_ratio_20'] = df['volume'] / df['volume_ma20']
        
        # Handle divide by zero
        df['volume_ratio_5'] = df['volume_ratio_5'].replace([np.inf, -np.inf], np.nan)
        df['volume_ratio_20'] = df['volume_ratio_20'].replace([np.inf, -np.inf], np.nan)
        df['volume_ratio_5'] = df['volume_ratio_5'].fillna(1)
        df['volume_ratio_20'] = df['volume_ratio_20'].fillna(1)
        
        # Calculate volume standard deviation
        df['volume_std5'] = df['volume'].rolling(window=5).std()
        df['volume_std20'] = df['volume'].rolling(window=20).std()
        
        # Identify high volume bars
        df['is_high_volume'] = df['volume'] > (df['volume_ma20'] + 2 * df['volume_std20'])
        
        return df
    
    def generate_target_columns(self, df: pd.DataFrame, 
                               predict_periods: List[int] = [1, 3, 5]) -> pd.DataFrame:
        """
        Generate target columns for supervised learning.
        
        Args:
            df (pd.DataFrame): Price dataframe
            predict_periods (List[int]): Periods to predict ahead
            
        Returns:
            pd.DataFrame: DataFrame with target columns
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Generate target columns for close price
        for period in predict_periods:
            # Future close price
            df[f'future_close_{period}'] = df['close'].shift(-period)
            
            # Direction (up/down) - binary classification target
            df[f'direction_{period}'] = (df[f'future_close_{period}'] > df['close']).astype(int)
            
            # Future returns
            df[f'future_return_{period}'] = df[f'future_close_{period}'] / df['close'] - 1
        
        return df
    
    def normalize_features(self, df: pd.DataFrame, exclude_columns: List[str] = None) -> pd.DataFrame:
        """
        Normalize features to [0, 1] range.
        
        Args:
            df (pd.DataFrame): Feature dataframe
            exclude_columns (List[str]): Columns to exclude from normalization
            
        Returns:
            pd.DataFrame: DataFrame with normalized features
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Default exclude columns if not provided
        if exclude_columns is None:
            exclude_columns = ['date', 'symbol', 'is_bullish', 'is_bearish', 'is_doji', 'is_high_volume']
            # Also exclude target columns
            for col in df.columns:
                if col.startswith('direction_') or col.startswith('future_close_') or col.startswith('future_return_'):
                    exclude_columns.append(col)
        
        # Identify columns to normalize
        cols_to_normalize = [col for col in df.columns if col not in exclude_columns and df[col].dtype != 'object']
        
        # Apply min-max normalization
        for col in cols_to_normalize:
            min_val = df[col].min()
            max_val = df[col].max()
            
            # Avoid divide by zero
            if max_val > min_val:
                df[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                df[col] = 0.5  # Default to middle value if all values are the same
        
        return df
    
    def generate_all_features(self, df: pd.DataFrame, include_targets: bool = True,
                             predict_periods: List[int] = [1, 3, 5],
                             normalize: bool = True) -> pd.DataFrame:
        """
        Generate a complete set of features.
        
        Args:
            df (pd.DataFrame): Raw price dataframe
            include_targets (bool): Whether to include target columns
            predict_periods (List[int]): Periods to predict ahead
            normalize (bool): Whether to normalize features
            
        Returns:
            pd.DataFrame: DataFrame with all features
        """
        if df.empty:
            self.logger.warning("Empty dataframe provided to generate_all_features")
            return df
        
        self.logger.info(f"Generating all features for {len(df)} records")
        
        # Generate each feature set
        df = self.generate_advanced_features(df)
        df = self.generate_price_features(df)
        df = self.generate_volume_features(df)
        
        # Generate target columns if requested
        if include_targets:
            df = self.generate_target_columns(df, predict_periods)
        
        # Normalize features if requested
        if normalize:
            df = self.normalize_features(df)
        
        # Ensure all NaN values are filled
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        self.logger.info(f"Generated feature dataframe with {len(df.columns)} columns")
        
        return df


# Example usage
if __name__ == "__main__":
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Create a sample dataframe
    data = {
        'date': pd.date_range(start=datetime.now() - timedelta(days=100), periods=100, freq='D'),
        'open': np.random.normal(100, 5, 100),
        'high': np.random.normal(105, 5, 100),
        'low': np.random.normal(95, 5, 100),
        'close': np.random.normal(102, 5, 100),
        'volume': np.random.normal(1000000, 200000, 100)
    }
    
    # Ensure high >= open, close, low
    for i in range(len(data['high'])):
        data['high'][i] = max(data['high'][i], data['open'][i], data['close'][i], data['low'][i])
    
    # Ensure low <= open, close, high
    for i in range(len(data['low'])):
        data['low'][i] = min(data['low'][i], data['open'][i], data['close'][i], data['high'][i])
    
    df = pd.DataFrame(data)
    
    # Initialize the feature generator
    feature_generator = FeatureGenerator()
    
    # Generate all features
    features_df = feature_generator.generate_all_features(df)
    
    # Print feature names
    print(f"Generated {len(features_df.columns)} features:")
    for col in features_df.columns:
        print(f"- {col}")
    
    # Print first few rows
    print("\nFirst 5 rows:")
    print(features_df.head())
    
    # Print some statistics
    print("\nFeature statistics:")
    print(features_df.describe())