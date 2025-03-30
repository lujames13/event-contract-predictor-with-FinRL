"""
Market Analyzer Module

This module analyzes market data to identify market states, trends,
and other characteristics useful for prediction models.

Classes:
    MarketAnalyzer: Main class for market analysis functionality
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime

from utils.logger import get_logger
from data.feature_engineering import TechnicalIndicators

# Setup logger
logger = get_logger(__name__)


class MarketAnalyzer:
    """
    Analyzes market data to identify market states and characteristics.
    
    This class provides methods for detecting trends, volatility levels,
    market regimes, and other characteristics to help inform prediction models.
    """
    
    # Market state constants
    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    SIDEWAYS = "sideways"
    WEAK_DOWNTREND = "weak_downtrend"
    STRONG_DOWNTREND = "strong_downtrend"
    
    # Volatility state constants
    HIGH_VOLATILITY = "high_volatility"
    MEDIUM_VOLATILITY = "medium_volatility"
    LOW_VOLATILITY = "low_volatility"
    
    def __init__(self):
        """Initialize the market analyzer."""
        self.logger = get_logger("MarketAnalyzer")
    
    def analyze_trend(self, df: pd.DataFrame, 
                     short_period: int = 20, 
                     medium_period: int = 50,
                     long_period: int = 200) -> Dict[str, Any]:
        """
        Analyze market trend based on moving averages.
        
        Args:
            df (pd.DataFrame): Price dataframe
            short_period (int): Short-term moving average period
            medium_period (int): Medium-term moving average period
            long_period (int): Long-term moving average period
            
        Returns:
            Dict: Trend analysis results
        """
        if df.empty:
            self.logger.warning("Empty dataframe provided to analyze_trend")
            return {"trend": "unknown", "strength": 0, "duration": 0}
        
        self.logger.debug(f"Analyzing trend for {len(df)} price records")
        
        # Create a copy of the dataframe for analysis
        df_analysis = df.copy()
        
        # Ensure we have the required columns
        if 'close' not in df_analysis.columns:
            self.logger.error("Close price column not found in dataframe")
            return {"trend": "unknown", "strength": 0, "duration": 0}
        
        # Calculate moving averages if not already present
        if f'ma{short_period}' not in df_analysis.columns:
            df_analysis = TechnicalIndicators.add_ma(df_analysis, periods=[short_period, medium_period, long_period])
        
        # Get the most recent values
        recent = df_analysis.iloc[-1]
        
        # Get moving averages
        ma_short = recent[f'ma{short_period}']
        ma_medium = recent[f'ma{medium_period}']
        ma_long = recent[f'ma{long_period}']
        
        # Determine trend based on moving average alignment
        if ma_short > ma_medium > ma_long:
            # Strong uptrend (all MAs aligned upward)
            trend = self.STRONG_UPTREND
            strength = 1.0
        elif ma_short > ma_medium and ma_medium <= ma_long:
            # Weak uptrend (short-term bullish but long-term bearish/neutral)
            trend = self.WEAK_UPTREND
            strength = 0.5
        elif ma_short < ma_medium < ma_long:
            # Strong downtrend (all MAs aligned downward)
            trend = self.STRONG_DOWNTREND
            strength = -1.0
        elif ma_short < ma_medium and ma_medium >= ma_long:
            # Weak downtrend (short-term bearish but long-term bullish/neutral)
            trend = self.WEAK_DOWNTREND
            strength = -0.5
        else:
            # Sideways/consolidation
            trend = self.SIDEWAYS
            strength = 0.0
        
        # Calculate trend duration
        duration = self._calculate_trend_duration(df_analysis, trend)
        
        # Calculate trend momentum
        momentum = self._calculate_trend_momentum(df_analysis)
        
        # Calculate trend consistency
        consistency = self._calculate_trend_consistency(df_analysis)
        
        return {
            "trend": trend,
            "strength": strength,
            "duration": duration,
            "momentum": momentum,
            "consistency": consistency,
            "short_ma": ma_short,
            "medium_ma": ma_medium,
            "long_ma": ma_long
        }
    
    def _calculate_trend_duration(self, df: pd.DataFrame, current_trend: str) -> int:
        """
        Calculate how many periods the current trend has been in effect.
        
        Args:
            df (pd.DataFrame): Price dataframe with moving averages
            current_trend (str): Current trend state
            
        Returns:
            int: Trend duration in periods
        """
        # This is a simplified version - a real implementation would track the
        # precise point where the trend changed
        if 'ma20' not in df.columns or 'ma50' not in df.columns or 'ma200' not in df.columns:
            return 0
        
        if current_trend == self.STRONG_UPTREND:
            # Count periods where ma20 > ma50 > ma200
            condition = (df['ma20'] > df['ma50']) & (df['ma50'] > df['ma200'])
        elif current_trend == self.WEAK_UPTREND:
            # Count periods where ma20 > ma50 and ma50 <= ma200
            condition = (df['ma20'] > df['ma50']) & (df['ma50'] <= df['ma200'])
        elif current_trend == self.STRONG_DOWNTREND:
            # Count periods where ma20 < ma50 < ma200
            condition = (df['ma20'] < df['ma50']) & (df['ma50'] < df['ma200'])
        elif current_trend == self.WEAK_DOWNTREND:
            # Count periods where ma20 < ma50 and ma50 >= ma200
            condition = (df['ma20'] < df['ma50']) & (df['ma50'] >= df['ma200'])
        else:  # SIDEWAYS
            # Count periods where the ma's are close to each other
            close_threshold = 0.01  # 1% difference
            rel_diff_20_50 = abs(df['ma20'] - df['ma50']) / df['ma50']
            rel_diff_50_200 = abs(df['ma50'] - df['ma200']) / df['ma200']
            condition = (rel_diff_20_50 < close_threshold) & (rel_diff_50_200 < close_threshold)
        
        # Find the longest consecutive run of True values at the end
        if condition.iloc[-1]:
            duration = 1
            for i in range(2, len(condition) + 1):
                if i > len(condition) or not condition.iloc[-i]:
                    break
                duration += 1
            return duration
        else:
            return 0
    
    def _calculate_trend_momentum(self, df: pd.DataFrame) -> float:
        """
        Calculate the momentum of the current trend.
        
        Args:
            df (pd.DataFrame): Price dataframe
            
        Returns:
            float: Trend momentum [-1, 1]
        """
        # We can use MACD or ROC for momentum
        if 'macd' in df.columns and len(df) > 1:
            # Calculate momentum from MACD
            recent_macd = df['macd'].iloc[-1]
            prev_macd = df['macd'].iloc[-2]
            macd_momentum = recent_macd - prev_macd
            
            # Normalize to [-1, 1] range
            if 'close' in df.columns:
                close_price = df['close'].iloc[-1]
                normalized_momentum = macd_momentum / (close_price * 0.01)  # Normalize to 1% of price
                return max(min(normalized_momentum, 1), -1)  # Clamp to [-1, 1]
            else:
                return 0
        elif 'roc12' in df.columns:
            # Use Rate of Change as momentum
            roc = df['roc12'].iloc[-1]
            # Normalize ROC to [-1, 1]
            return max(min(roc / 10, 1), -1)  # Assuming ROC of 10% is strong
        else:
            # Calculate simple momentum from price changes
            if 'close' in df.columns and len(df) > 5:
                recent_close = df['close'].iloc[-1]
                prev_close = df['close'].iloc[-6]  # 5 periods ago
                price_change = (recent_close - prev_close) / prev_close
                return max(min(price_change * 5, 1), -1)  # Scale to [-1, 1]
            else:
                return 0
    
    def _calculate_trend_consistency(self, df: pd.DataFrame) -> float:
        """
        Calculate the consistency of the current trend.
        
        Args:
            df (pd.DataFrame): Price dataframe
            
        Returns:
            float: Trend consistency [0, 1]
        """
        # Use price direction consistency over the past N periods
        if 'close' not in df.columns or len(df) < 10:
            return 0.5
        
        # Get the last 10 price changes
        price_changes = df['close'].diff().iloc[-10:]
        
        # Count positive and negative changes
        positive_changes = (price_changes > 0).sum()
        negative_changes = (price_changes < 0).sum()
        
        # Calculate directional consistency
        total_changes = positive_changes + negative_changes
        if total_changes == 0:
            return 0.5
        
        # If mostly positive changes, consistency is high for uptrend
        if positive_changes > negative_changes:
            return positive_changes / total_changes
        # If mostly negative changes, consistency is high for downtrend
        else:
            return negative_changes / total_changes
    
    def analyze_volatility(self, df: pd.DataFrame, 
                          short_period: int = 20,
                          long_period: int = 100) -> Dict[str, Any]:
        """
        Analyze market volatility.
        
        Args:
            df (pd.DataFrame): Price dataframe
            short_period (int): Short-term volatility period
            long_period (int): Long-term volatility period
            
        Returns:
            Dict: Volatility analysis results
        """
        if df.empty:
            self.logger.warning("Empty dataframe provided to analyze_volatility")
            return {"volatility": "unknown", "level": 0}
        
        self.logger.debug(f"Analyzing volatility for {len(df)} price records")
        
        # Create a copy of the dataframe for analysis
        df_analysis = df.copy()
        
        # Ensure we have the required columns
        if 'close' not in df_analysis.columns:
            self.logger.error("Close price column not found in dataframe")
            return {"volatility": "unknown", "level": 0}
        
        # Calculate daily returns if not already present
        if 'daily_return' not in df_analysis.columns:
            df_analysis['daily_return'] = df_analysis['close'].pct_change()
        
        # Calculate volatility (standard deviation of returns)
        short_volatility = df_analysis['daily_return'].rolling(window=short_period).std().iloc[-1]
        long_volatility = df_analysis['daily_return'].rolling(window=long_period).std().iloc[-1]
        
        # Calculate ATR if not already present
        if 'atr14' not in df_analysis.columns and all(col in df_analysis.columns for col in ['high', 'low', 'close']):
            df_analysis = TechnicalIndicators.add_atr(df_analysis, period=14)
        
        # Get ATR relative to price
        if 'atr14' in df_analysis.columns:
            atr = df_analysis['atr14'].iloc[-1]
            close = df_analysis['close'].iloc[-1]
            atr_pct = atr / close * 100  # ATR as percentage of close price
        else:
            atr_pct = None
        
        # Volatility classification
        # These thresholds would normally be calibrated to specific market conditions
        if short_volatility is not None:
            annual_vol = short_volatility * (252 ** 0.5)  # Convert to annualized volatility
            
            if annual_vol > 0.4:  # Over 40% annually
                volatility_state = self.HIGH_VOLATILITY
                level = 1.0
            elif annual_vol > 0.2:  # 20-40% annually
                volatility_state = self.MEDIUM_VOLATILITY
                level = 0.5
            else:  # Under 20% annually
                volatility_state = self.LOW_VOLATILITY
                level = 0.0
        else:
            volatility_state = "unknown"
            level = 0
        
        # Calculate relative volatility
        relative_volatility = short_volatility / long_volatility if long_volatility > 0 else 1
        
        # Calculate volatility trend
        if len(df_analysis) > 20:
            vol_20_periods_ago = df_analysis['daily_return'].rolling(window=short_period).std().iloc[-21]
            volatility_change = (short_volatility - vol_20_periods_ago) / vol_20_periods_ago if vol_20_periods_ago > 0 else 0
        else:
            volatility_change = 0
        
        return {
            "volatility": volatility_state,
            "level": level,
            "short_term": short_volatility,
            "long_term": long_volatility,
            "relative": relative_volatility,
            "trend": volatility_change,
            "atr_pct": atr_pct
        }
    
    def analyze_market_state(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive market state analysis.
        
        Args:
            df (pd.DataFrame): Price dataframe
            
        Returns:
            Dict: Comprehensive market state analysis
        """
        if df.empty:
            self.logger.warning("Empty dataframe provided to analyze_market_state")
            return {"state": "unknown"}
        
        self.logger.info(f"Analyzing market state for {len(df)} price records")
        
        # Get trend analysis
        trend_analysis = self.analyze_trend(df)
        
        # Get volatility analysis
        volatility_analysis = self.analyze_volatility(df)
        
        # Analyze support/resistance
        support_resistance = self.analyze_support_resistance(df)
        
        # Analyze market regime
        market_regime = self.determine_market_regime(
            trend=trend_analysis.get("trend", "unknown"),
            volatility=volatility_analysis.get("volatility", "unknown")
        )
        
        # Calculate predicted reversal probability
        reversal_probability = self.calculate_reversal_probability(df, trend_analysis, volatility_analysis)
        
        # Combined analysis
        analysis = {
            "state": market_regime,
            "trend": trend_analysis,
            "volatility": volatility_analysis,
            "support_resistance": support_resistance,
            "reversal_probability": reversal_probability
        }
        
        self.logger.info(f"Market state determined as '{market_regime}' with "
                        f"trend '{trend_analysis.get('trend')}' and "
                        f"volatility '{volatility_analysis.get('volatility')}'")
        
        return analysis
    
    def determine_market_regime(self, trend: str, volatility: str) -> str:
        """
        Determine overall market regime based on trend and volatility.
        
        Args:
            trend (str): Trend state
            volatility (str): Volatility state
            
        Returns:
            str: Market regime
        """
        # Define market regimes based on trend and volatility combinations
        if trend == self.STRONG_UPTREND:
            if volatility == self.HIGH_VOLATILITY:
                return "volatile_bull"
            elif volatility == self.LOW_VOLATILITY:
                return "steady_bull"
            else:
                return "moderate_bull"
        
        elif trend == self.WEAK_UPTREND:
            if volatility == self.HIGH_VOLATILITY:
                return "volatile_recovery"
            elif volatility == self.LOW_VOLATILITY:
                return "weak_recovery"
            else:
                return "moderate_recovery"
        
        elif trend == self.STRONG_DOWNTREND:
            if volatility == self.HIGH_VOLATILITY:
                return "volatile_bear"
            elif volatility == self.LOW_VOLATILITY:
                return "steady_bear"
            else:
                return "moderate_bear"
        
        elif trend == self.WEAK_DOWNTREND:
            if volatility == self.HIGH_VOLATILITY:
                return "volatile_correction"
            elif volatility == self.LOW_VOLATILITY:
                return "weak_correction"
            else:
                return "moderate_correction"
        
        else:  # SIDEWAYS
            if volatility == self.HIGH_VOLATILITY:
                return "volatile_consolidation"
            elif volatility == self.LOW_VOLATILITY:
                return "tight_range"
            else:
                return "consolidation"
    
    def analyze_support_resistance(self, df: pd.DataFrame, 
                                  lookback_periods: int = 100,
                                  price_threshold: float = 0.02) -> Dict[str, Any]:
        """
        Analyze key support and resistance levels.
        
        Args:
            df (pd.DataFrame): Price dataframe
            lookback_periods (int): Number of periods to look back
            price_threshold (float): Price proximity threshold (as percentage)
            
        Returns:
            Dict: Support and resistance analysis
        """
        if df.empty or 'close' not in df.columns:
            return {"support": [], "resistance": []}
        
        # Use only a subset of the data for analysis
        analysis_df = df.iloc[-lookback_periods:].copy() if len(df) > lookback_periods else df.copy()
        
        # Get the most recent close
        recent_close = analysis_df['close'].iloc[-1]
        
        # Find local highs and lows
        analysis_df['local_high'] = analysis_df['high'].rolling(window=11, center=True).apply(
            lambda x: x[5] == max(x), raw=True
        ).fillna(0).astype(bool)
        
        analysis_df['local_low'] = analysis_df['low'].rolling(window=11, center=True).apply(
            lambda x: x[5] == min(x), raw=True
        ).fillna(0).astype(bool)
        
        # Get high and low points
        highs = analysis_df[analysis_df['local_high']]['high'].tolist()
        lows = analysis_df[analysis_df['local_low']]['low'].tolist()
        
        # Cluster similar price levels
        resistance_levels = self._cluster_price_levels(highs, price_threshold)
        support_levels = self._cluster_price_levels(lows, price_threshold)
        
        # Filter levels by relevance (proximity to current price)
        resistance_levels = [level for level in resistance_levels if level > recent_close]
        support_levels = [level for level in support_levels if level < recent_close]
        
        # Sort by proximity to current price
        resistance_levels.sort()
        support_levels.sort(reverse=True)
        
        # Calculate proximity percentage
        closest_resistance = min(resistance_levels) if resistance_levels else None
        closest_support = max(support_levels) if support_levels else None
        
        if closest_resistance:
            resistance_proximity = (closest_resistance - recent_close) / recent_close
        else:
            resistance_proximity = None
        
        if closest_support:
            support_proximity = (recent_close - closest_support) / recent_close
        else:
            support_proximity = None
        
        return {
            "support": support_levels,
            "resistance": resistance_levels,
            "closest_support": closest_support,
            "closest_resistance": closest_resistance,
            "support_proximity": support_proximity,
            "resistance_proximity": resistance_proximity
        }
    
    def _cluster_price_levels(self, price_points: List[float], threshold: float) -> List[float]:
        """
        Cluster similar price levels.
        
        Args:
            price_points (List[float]): List of price points
            threshold (float): Proximity threshold as percentage
            
        Returns:
            List[float]: Clustered price levels
        """
        if not price_points:
            return []
        
        # Sort price points
        sorted_prices = sorted(price_points)
        clusters = []
        current_cluster = [sorted_prices[0]]
        
        # Group prices within threshold
        for price in sorted_prices[1:]:
            if price <= current_cluster[0] * (1 + threshold) and price >= current_cluster[0] * (1 - threshold):
                # Price is within threshold of first price in cluster
                current_cluster.append(price)
            else:
                # Start a new cluster
                clusters.append(sum(current_cluster) / len(current_cluster))  # Average price of cluster
                current_cluster = [price]
        
        # Add the last cluster
        if current_cluster:
            clusters.append(sum(current_cluster) / len(current_cluster))
        
        return clusters
    
    def calculate_reversal_probability(self, df: pd.DataFrame, 
                                      trend_analysis: Dict[str, Any],
                                      volatility_analysis: Dict[str, Any]) -> float:
        """
        Calculate the probability of a trend reversal.
        
        Args:
            df (pd.DataFrame): Price dataframe
            trend_analysis (Dict): Trend analysis results
            volatility_analysis (Dict): Volatility analysis results
            
        Returns:
            float: Reversal probability [0, 1]
        """
        # This is a simplified reversal probability calculation
        # A real implementation would incorporate many more factors
        
        reversal_probability = 0.5  # Start with neutral probability
        
        # Factor 1: Trend duration
        trend_duration = trend_analysis.get('duration', 0)
        # Longer trends are more likely to reverse
        if trend_duration > 50:
            reversal_probability += 0.15
        elif trend_duration > 20:
            reversal_probability += 0.05
        
        # Factor 2: Trend strength
        trend_strength = abs(trend_analysis.get('strength', 0))
        # Stronger trends are less likely to reverse immediately
        reversal_probability -= trend_strength * 0.1
        
        # Factor 3: Volatility
        volatility_level = volatility_analysis.get('level', 0)
        # Higher volatility increases reversal probability
        reversal_probability += volatility_level * 0.1
        
        # Factor 4: RSI overbought/oversold
        if 'rsi14' in df.columns:
            rsi = df['rsi14'].iloc[-1]
            if rsi > 70:  # Overbought
                reversal_probability += 0.15 + (rsi - 70) * 0.005  # More overbought = higher probability
            elif rsi < 30:  # Oversold
                reversal_probability += 0.15 + (30 - rsi) * 0.005  # More oversold = higher probability
        
        # Factor 5: Bollinger Band proximity
        if all(col in df.columns for col in ['close', 'bb_upper', 'bb_lower']):
            close = df['close'].iloc[-1]
            upper = df['bb_upper'].iloc[-1]
            lower = df['bb_lower'].iloc[-1]
            
            if close > upper:  # Price above upper band
                reversal_probability += 0.1
            elif close < lower:  # Price below lower band
                reversal_probability += 0.1
        
        # Factor 6: Price near support/resistance
        sr_analysis = self.analyze_support_resistance(df)
        support_proximity = sr_analysis.get('support_proximity')
        resistance_proximity = sr_analysis.get('resistance_proximity')
        
        if support_proximity is not None and support_proximity < 0.02:  # Within 2% of support
            reversal_probability -= 0.1  # Support reduces downward reversal probability
        
        if resistance_proximity is not None and resistance_proximity < 0.02:  # Within 2% of resistance
            reversal_probability += 0.1  # Resistance increases upward reversal probability
        
        # Ensure probability is in [0, 1] range
        return max(0, min(1, reversal_probability))
    
    def analyze_market_cycles(self, df: pd.DataFrame, cycle_period: int = 20) -> Dict[str, Any]:
        """
        Analyze market cycles and patterns.
        
        Args:
            df (pd.DataFrame): Price dataframe
            cycle_period (int): Cycle detection period
            
        Returns:
            Dict: Market cycle analysis
        """
        if df.empty or 'close' not in df.columns:
            return {"cycle_position": "unknown"}
        
        # This is a simplified cycle detection algorithm
        # Real implementation would use more sophisticated methods like Fourier analysis
        
        # Calculate cycles using rate of change
        if 'roc12' not in df.columns:
            df = TechnicalIndicators.add_roc(df, period=cycle_period)
        
        # Look for zero crossings of ROC to detect cycle transitions
        roc_col = f'roc{cycle_period}'
        if roc_col in df.columns and len(df) > cycle_period:
            # Detect sign changes (zero crossings)
            sign_changes = np.diff(np.signbit(df[roc_col].values))
            change_indices = np.where(sign_changes)[0]
            
            if len(change_indices) > 0:
                # Calculate average cycle length
                if len(change_indices) > 1:
                    avg_cycle_length = np.mean(np.diff(change_indices))
                else:
                    avg_cycle_length = None
                
                # Determine current cycle position
                last_change_idx = change_indices[-1] if len(change_indices) > 0 else 0
                periods_since_change = len(df) - 1 - last_change_idx
                
                if avg_cycle_length is not None:
                    cycle_position = (periods_since_change / avg_cycle_length) % 1
                else:
                    cycle_position = None
                
                # Determine cycle phase
                roc_sign = np.sign(df[roc_col].iloc[-1])
                roc_trend = df[roc_col].iloc[-1] - df[roc_col].iloc[-2] if len(df) > 1 else 0
                
                if roc_sign > 0 and roc_trend > 0:
                    cycle_phase = "acceleration"
                elif roc_sign > 0 and roc_trend <= 0:
                    cycle_phase = "deceleration"
                elif roc_sign <= 0 and roc_trend <= 0:
                    cycle_phase = "contraction"
                else:  # roc_sign <= 0 and roc_trend > 0
                    cycle_phase = "recovery"
                
                return {
                    "cycle_position": cycle_position,
                    "cycle_phase": cycle_phase,
                    "avg_cycle_length": avg_cycle_length,
                    "periods_since_change": periods_since_change
                }
        
        return {"cycle_position": "unknown"}
    
    def detect_divergences(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Detect divergences between price and indicators.
        
        Args:
            df (pd.DataFrame): Price dataframe with indicators
            
        Returns:
            Dict[str, bool]: Detected divergences
        """
        if df.empty or 'close' not in df.columns:
            return {"rsi_divergence": False, "macd_divergence": False}
        
        divergences = {}
        
        # Check for RSI divergence
        if 'rsi14' in df.columns and len(df) > 14:
            divergences["rsi_bullish_divergence"] = self._check_bullish_divergence(
                df['close'], df['rsi14'], periods=14
            )
            divergences["rsi_bearish_divergence"] = self._check_bearish_divergence(
                df['close'], df['rsi14'], periods=14
            )
        
        # Check for MACD divergence
        if 'macd' in df.columns and len(df) > 14:
            divergences["macd_bullish_divergence"] = self._check_bullish_divergence(
                df['close'], df['macd'], periods=14
            )
            divergences["macd_bearish_divergence"] = self._check_bearish_divergence(
                df['close'], df['macd'], periods=14
            )
        
        return divergences
    
    def _check_bullish_divergence(self, price: pd.Series, indicator: pd.Series, 
                                 periods: int = 14) -> bool:
        """
        Check for bullish divergence (price making lower lows, indicator making higher lows).
        
        Args:
            price (pd.Series): Price series
            indicator (pd.Series): Indicator series
            periods (int): Lookback periods
            
        Returns:
            bool: True if bullish divergence detected
        """
        if len(price) < periods or len(indicator) < periods:
            return False
        
        # Get recent data
        recent_price = price.iloc[-periods:].values
        recent_indicator = indicator.iloc[-periods:].values
        
        # Find price lows
        price_lows = []
        for i in range(1, len(recent_price) - 1):
            if recent_price[i] < recent_price[i-1] and recent_price[i] < recent_price[i+1]:
                price_lows.append((i, recent_price[i]))
        
        # Find indicator lows
        indicator_lows = []
        for i in range(1, len(recent_indicator) - 1):
            if recent_indicator[i] < recent_indicator[i-1] and recent_indicator[i] < recent_indicator[i+1]:
                indicator_lows.append((i, recent_indicator[i]))
        
        # Need at least two lows to compare
        if len(price_lows) < 2 or len(indicator_lows) < 2:
            return False
        
        # Sort by index (time)
        price_lows.sort(key=lambda x: x[0])
        indicator_lows.sort(key=lambda x: x[0])
        
        # Check for divergence (price lower low, indicator higher low)
        price_making_lower_lows = price_lows[-1][1] < price_lows[-2][1]
        indicator_making_higher_lows = indicator_lows[-1][1] > indicator_lows[-2][1]
        
        return price_making_lower_lows and indicator_making_higher_lows
    
    def _check_bearish_divergence(self, price: pd.Series, indicator: pd.Series, 
                                 periods: int = 14) -> bool:
        """
        Check for bearish divergence (price making higher highs, indicator making lower highs).
        
        Args:
            price (pd.Series): Price series
            indicator (pd.Series): Indicator series
            periods (int): Lookback periods
            
        Returns:
            bool: True if bearish divergence detected
        """
        if len(price) < periods or len(indicator) < periods:
            return False
        
        # Get recent data
        recent_price = price.iloc[-periods:].values
        recent_indicator = indicator.iloc[-periods:].values
        
        # Find price highs
        price_highs = []
        for i in range(1, len(recent_price) - 1):
            if recent_price[i] > recent_price[i-1] and recent_price[i] > recent_price[i+1]:
                price_highs.append((i, recent_price[i]))
        
        # Find indicator highs
        indicator_highs = []
        for i in range(1, len(recent_indicator) - 1):
            if recent_indicator[i] > recent_indicator[i-1] and recent_indicator[i] > recent_indicator[i+1]:
                indicator_highs.append((i, recent_indicator[i]))
        
        # Need at least two highs to compare
        if len(price_highs) < 2 or len(indicator_highs) < 2:
            return False
        
        # Sort by index (time)
        price_highs.sort(key=lambda x: x[0])
        indicator_highs.sort(key=lambda x: x[0])
        
        # Check for divergence (price higher high, indicator lower high)
        price_making_higher_highs = price_highs[-1][1] > price_highs[-2][1]
        indicator_making_lower_highs = indicator_highs[-1][1] < indicator_highs[-2][1]
        
        return price_making_higher_highs and indicator_making_lower_highs
    
    def analyze_multi_timeframe(self, timeframes_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze market across multiple timeframes.
        
        Args:
            timeframes_data (Dict[str, pd.DataFrame]): Dictionary of timeframe -> price dataframe
            
        Returns:
            Dict[str, Any]: Multi-timeframe analysis results
        """
        if not timeframes_data:
            return {"alignment": "unknown", "strength": 0}
        
        self.logger.info(f"Analyzing {len(timeframes_data)} timeframes")
        
        # Analyze each timeframe
        timeframe_analyses = {}
        for timeframe, df in timeframes_data.items():
            timeframe_analyses[timeframe] = self.analyze_market_state(df)
        
        # Check trend alignment across timeframes
        trends = [analysis['trend']['trend'] for tf, analysis in timeframe_analyses.items()]
        
        # Determine overall alignment
        if all(trend == self.STRONG_UPTREND for trend in trends):
            alignment = "strongly_bullish"
            strength = 1.0
        elif all(trend == self.STRONG_DOWNTREND for trend in trends):
            alignment = "strongly_bearish"
            strength = -1.0
        elif all(trend in [self.STRONG_UPTREND, self.WEAK_UPTREND] for trend in trends):
            alignment = "bullish"
            strength = 0.7
        elif all(trend in [self.STRONG_DOWNTREND, self.WEAK_DOWNTREND] for trend in trends):
            alignment = "bearish"
            strength = -0.7
        elif any(trend in [self.STRONG_UPTREND, self.WEAK_UPTREND] for trend in trends) and \
             any(trend in [self.STRONG_DOWNTREND, self.WEAK_DOWNTREND] for trend in trends):
            alignment = "conflicted"
            strength = 0.0
        elif all(trend == self.SIDEWAYS for trend in trends):
            alignment = "neutral"
            strength = 0.0
        else:
            alignment = "mixed"
            
            # Calculate strength as weighted average
            uptrend_count = sum(1 for trend in trends if trend in [self.STRONG_UPTREND, self.WEAK_UPTREND])
            downtrend_count = sum(1 for trend in trends if trend in [self.STRONG_DOWNTREND, self.WEAK_DOWNTREND])
            strength = (uptrend_count - downtrend_count) / len(trends)
        
        # Check for divergences between timeframes
        has_divergence = False
        divergence_details = {}
        
        for i, (tf1, analysis1) in enumerate(timeframe_analyses.items()):
            for tf2, analysis2 in list(timeframe_analyses.items())[i+1:]:
                # Check if one timeframe is trending up and the other down
                trend1 = analysis1['trend']['trend']
                trend2 = analysis2['trend']['trend']
                
                if ((trend1.startswith('strong_up') or trend1.startswith('weak_up')) and
                    (trend2.startswith('strong_down') or trend2.startswith('weak_down'))) or \
                   ((trend2.startswith('strong_up') or trend2.startswith('weak_up')) and
                    (trend1.startswith('strong_down') or trend1.startswith('weak_down'))):
                    has_divergence = True
                    divergence_details[f"{tf1}_vs_{tf2}"] = [trend1, trend2]
        
        return {
            "alignment": alignment,
            "strength": strength,
            "has_divergence": has_divergence,
            "divergence_details": divergence_details,
            "timeframe_analyses": timeframe_analyses
        }
    
    def predict_optimal_timeframe(self, timeframes_data: Dict[str, pd.DataFrame]) -> str:
        """
        Predict which timeframe might be optimal for trading given current conditions.
        
        Args:
            timeframes_data (Dict[str, pd.DataFrame]): Dictionary of timeframe -> price dataframe
            
        Returns:
            str: Optimal timeframe
        """
        if not timeframes_data:
            return None
        
        # Get multi-timeframe analysis
        mtf_analysis = self.analyze_multi_timeframe(timeframes_data)
        
        # Calculate scores for each timeframe
        scores = {}
        
        for tf, df in timeframes_data.items():
            if df.empty:
                continue
            
            # Get analysis for this timeframe
            analysis = mtf_analysis['timeframe_analyses'].get(tf, {})
            
            # Initialize score
            scores[tf] = 0
            
            # Factor 1: Trend strength
            trend_strength = abs(analysis.get('trend', {}).get('strength', 0))
            scores[tf] += trend_strength * 30  # Strong trends get higher score
            
            # Factor 2: Trend consistency
            trend_consistency = analysis.get('trend', {}).get('consistency', 0.5)
            scores[tf] += trend_consistency * 20
            
            # Factor 3: Volatility
            volatility_level = analysis.get('volatility', {}).get('level', 0)
            # For shorter timeframes, higher volatility is better
            # For longer timeframes, lower volatility is better
            if tf in ['1m', '5m', '15m']:
                scores[tf] += volatility_level * 20
            else:
                scores[tf] += (1 - volatility_level) * 20
            
            # Factor 4: Support/Resistance proximity
            sr_analysis = analysis.get('support_resistance', {})
            support_proximity = sr_analysis.get('support_proximity')
            resistance_proximity = sr_analysis.get('resistance_proximity')
            
            # If price is near support or resistance, this is a good timeframe
            if support_proximity is not None and support_proximity < 0.03:  # Within 3% of support
                scores[tf] += 15
            
            if resistance_proximity is not None and resistance_proximity < 0.03:  # Within 3% of resistance
                scores[tf] += 15
            
            # Factor 5: Divergence presence
            divergences = self.detect_divergences(df)
            if any(divergences.values()):
                scores[tf] += 20  # Divergences provide good trading opportunities
        
        # Choose highest scoring timeframe
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        else:
            return None


# Example usage
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    from data.feature_engineering import FeatureGenerator, TechnicalIndicators
    
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
    
    # Add technical indicators
    df = TechnicalIndicators.add_ma(df, periods=[7, 20, 50, 200])
    df = TechnicalIndicators.add_rsi(df)
    df = TechnicalIndicators.add_macd(df)
    df = TechnicalIndicators.add_bollinger_bands(df)
    df = TechnicalIndicators.add_atr(df)
    
    # Initialize market analyzer
    market_analyzer = MarketAnalyzer()
    
    # Analyze market
    analysis = market_analyzer.analyze_market_state(df)
    
    print("Market Analysis:")
    print(f"State: {analysis['state']}")
    print(f"Trend: {analysis['trend']['trend']}")
    print(f"Volatility: {analysis['volatility']['volatility']}")
    print(f"Reversal Probability: {analysis['reversal_probability']:.2f}")
    
    # Check for divergences
    divergences = market_analyzer.detect_divergences(df)
    print("\nDivergences:")
    for name, detected in divergences.items():
        print(f"{name}: {'Detected' if detected else 'Not detected'}")
    
    # Analyze multiple timeframes
    timeframes = {
        '1h': df,  # Same data for example purposes
        '4h': df,
        '1d': df
    }
    mtf_analysis = market_analyzer.analyze_multi_timeframe(timeframes)
    
    print("\nMulti-timeframe Analysis:")
    print(f"Alignment: {mtf_analysis['alignment']}")
    print(f"Strength: {mtf_analysis['strength']}")
    print(f"Has divergence: {mtf_analysis['has_divergence']}")
    
    # Predict optimal timeframe
    optimal_tf = market_analyzer.predict_optimal_timeframe(timeframes)
    print(f"\nOptimal timeframe: {optimal_tf}")