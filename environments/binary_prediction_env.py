"""
Binary Prediction Environment Module

This module implements a custom Gymnasium environment for binary prediction
(up/down) of cryptocurrency prices.

Classes:
    BinaryPredictionEnv: Gymnasium-compatible environment for directional price prediction
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime

from utils.logger import get_logger

# Setup logger
logger = get_logger(__name__)


class BinaryPredictionEnv(gym.Env):
    """
    A Gymnasium environment for binary (up/down) price prediction.
    
    This environment is designed for training agents to predict the direction
    of price movements over a specified time horizon.
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, df: pd.DataFrame, 
                window_size: int = 20,
                prediction_horizon: int = 1,
                features: Optional[List[str]] = None,
                reward_scaling: float = 1.0,
                render_mode: Optional[str] = None):
        """
        Initialize the Binary Prediction Environment.
        
        Args:
            df (pd.DataFrame): DataFrame with price and feature data
            window_size (int): Number of time steps to use as observation
            prediction_horizon (int): Number of steps ahead to predict
            features (List[str], optional): List of feature columns to use
            reward_scaling (float): Scaling factor for rewards
            render_mode (str, optional): Rendering mode
        """
        super(BinaryPredictionEnv, self).__init__()
        
        self.logger = get_logger("BinaryPredictionEnv")
        self.df = df.copy()  # Create a copy to avoid modifying the original
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.reward_scaling = reward_scaling
        self.render_mode = render_mode
        
        # Validate and process DataFrame
        self._validate_and_process_df()
        
        # Set up features
        self.features = features or self._default_features()
        self._validate_features()
        
        # Set up action and observation spaces
        self.action_space = spaces.Discrete(2)  # 0: Down, 1: Up
        
        # Observation space: window_size x num_features
        self.num_features = len(self.features)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.window_size, self.num_features),
            dtype=np.float32
        )
        
        # Initialize other state variables
        self.current_step = 0
        self.max_steps = len(self.df) - self.window_size - self.prediction_horizon
        self.current_action = None
        self.current_price = None
        self.future_price = None
        self.episode_returns = 0
        self.episode_correct = 0
        self.episode_count = 0
        
        self.logger.info(f"Initialized BinaryPredictionEnv with {self.num_features} features "
                        f"and {self.max_steps} max steps")
    
    def _validate_and_process_df(self):
        """Validate and process the input DataFrame."""
        if self.df.empty:
            raise ValueError("Empty DataFrame provided")
        
        # Ensure DataFrame is sorted by date
        if 'date' in self.df.columns:
            self.df = self.df.sort_values('date')
        
        # Check for required columns
        required_columns = ['close']
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"Required column '{col}' not found in DataFrame")
        
        # Check if DataFrame has enough data
        min_length = self.window_size + self.prediction_horizon + 1
        if len(self.df) < min_length:
            raise ValueError(f"DataFrame has {len(self.df)} rows, but at least {min_length} are required")
        
        # Fill NaN values - using newer pandas methods to avoid deprecation warning
        self.df = self.df.ffill().bfill().fillna(0)
        
        # Create target column if needed
        target_col = f'direction_{self.prediction_horizon}'
        if target_col not in self.df.columns:
            self.df[target_col] = (self.df['close'].shift(-self.prediction_horizon) > self.df['close']).astype(int)
        
        self.logger.debug(f"DataFrame processed with shape {self.df.shape}")
    
    def _default_features(self) -> List[str]:
        """Get a default list of features based on available columns."""
        # Basic set of technical indicators if available
        default_features = []
        
        # Price-based features
        for col in ['close', 'open', 'high', 'low']:
            if col in self.df.columns:
                default_features.append(col)
        
        # Volume
        if 'volume' in self.df.columns:
            default_features.append('volume')
        
        # Technical indicators
        indicator_prefixes = ['ma', 'ema', 'rsi', 'macd', 'bb_', 'atr', 'adx', 'stoch_', 'cci', 'obv']
        for prefix in indicator_prefixes:
            for col in self.df.columns:
                if col.startswith(prefix) and col not in default_features:
                    default_features.append(col)
        
        if not default_features:
            # If no features were found, at least use 'close'
            if 'close' in self.df.columns:
                default_features.append('close')
            else:
                raise ValueError("No usable features found in DataFrame")
                
        self.logger.debug(f"Selected {len(default_features)} default features")
        return default_features
    
    def _validate_features(self):
        """Validate that all specified features exist in the DataFrame."""
        missing_features = [f for f in self.features if f not in self.df.columns]
        if missing_features:
            self.logger.warning(f"Missing features: {missing_features}")
            # Remove missing features from the list
            self.features = [f for f in self.features if f in self.df.columns]
            
            if not self.features:
                raise ValueError("No valid features found in DataFrame")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state and returns initial observation.
        
        Args:
            seed (int, optional): Random seed
            options (Dict, optional): Additional options
            
        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Reset state
        self.current_step = 0
        self.current_action = None
        self.current_price = None
        self.future_price = None
        self.episode_returns = 0
        self.episode_correct = 0
        self.episode_count += 1
        
        # Random start position (if training)
        if options and options.get('random_start', False) and self.max_steps > 0:
            self.current_step = self.np_random.integers(0, max(1, self.max_steps - 1))
            self.logger.debug(f"Random start at step {self.current_step}")
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action (int): Action to take (0: Down, 1: Up)
            
        Returns:
            observation: New observation
            reward: Reward from the action
            terminated: Whether the episode is done
            truncated: Whether the episode was truncated
            info: Additional information
        """
        if self.current_step >= self.max_steps:
            self.logger.warning(f"Step called after episode end at step {self.current_step}")
            # Return last observation with zero reward and done=True
            observation = self._get_observation()
            return observation, 0, True, False, self._get_info()
        
        # Validate action
        if action not in [0, 1]:
            self.logger.warning(f"Invalid action {action}, using random action instead")
            action = self.action_space.sample()
        
        # Store current step information
        self.current_action = action
        self.current_price = self.df.iloc[self.current_step + self.window_size]['close']
        
        # Get future price (after prediction horizon)
        future_idx = self.current_step + self.window_size + self.prediction_horizon
        self.future_price = self.df.iloc[future_idx]['close'] if future_idx < len(self.df) else None
        
        # Get actual direction
        actual_direction = self.df.iloc[self.current_step + self.window_size][f'direction_{self.prediction_horizon}']
        
        # Calculate reward
        reward = self._calculate_reward(action, actual_direction)
        self.episode_returns += reward
        
        # Update step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        # Get new observation
        observation = self._get_observation()
        info = self._get_info()
        
        if done:
            self.logger.debug(f"Episode {self.episode_count} finished with "
                             f"return {self.episode_returns:.2f} and "
                             f"accuracy {self.episode_correct / max(1, self.current_step):.2f}")
        
        return observation, reward, done, False, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get the current observation.
        
        Returns:
            np.ndarray: Observation array
        """
        try:
            # Get data for the current window
            start_idx = self.current_step
            end_idx = start_idx + self.window_size
            
            # Safety check
            if end_idx > len(self.df):
                self.logger.warning(f"Observation window exceeds DataFrame length. Adjusting.")
                end_idx = len(self.df)
                
            window_data = self.df.iloc[start_idx:end_idx][self.features].values
            
            # Ensure correct shape
            if window_data.shape[0] < self.window_size:
                # Pad with zeros if needed (shouldn't happen with proper indexing)
                pad_rows = self.window_size - window_data.shape[0]
                window_data = np.vstack([np.zeros((pad_rows, len(self.features))), window_data])
                self.logger.warning(f"Observation needed padding: {pad_rows} rows")
            
            # Handle NaN values
            window_data = np.nan_to_num(window_data, nan=0.0)
            
            return window_data.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Error getting observation: {e}")
            # Return a zero array as a fallback
            return np.zeros((self.window_size, len(self.features)), dtype=np.float32)
    
    def _calculate_reward(self, action: int, actual_direction: int) -> float:
        """
        Calculate reward for the current action.
        
        Args:
            action (int): Action taken (0: Down, 1: Up)
            actual_direction (int): Actual price direction (0: Down, 1: Up)
            
        Returns:
            float: Reward value
        """
        # Binary reward based on correct prediction
        correct_prediction = (action == actual_direction)
        
        if correct_prediction:
            reward = 1.0
            self.episode_correct += 1
        else:
            reward = -1.0
        
        # Apply reward scaling
        reward *= self.reward_scaling
        
        return reward
    
    def _get_info(self) -> Dict:
        """
        Get additional information about the current state.
        
        Returns:
            Dict: Information dictionary
        """
        info = {
            'step': self.current_step,
            'episode_returns': self.episode_returns,
            'accuracy': self.episode_correct / max(1, self.current_step)
        }
        
        # Add price information if available
        if self.current_price is not None:
            info['current_price'] = self.current_price
        
        if self.future_price is not None:
            info['future_price'] = self.future_price
            
        if self.current_action is not None:
            info['action'] = self.current_action
        
        # Add additional market state info if available
        try:
            if 'rsi14' in self.df.columns and self.current_step + self.window_size - 1 < len(self.df):
                info['rsi'] = self.df.iloc[self.current_step + self.window_size - 1]['rsi14']
        except Exception as e:
            self.logger.warning(f"Error adding market info: {e}")
        
        return info
    
    def render(self):
        """Render the current state of the environment."""
        if self.render_mode == 'human':
            if self.current_step > 0:
                action_name = "UP" if self.current_action == 1 else "DOWN"
                
                # Safely access the actual direction
                try:
                    idx = self.current_step + self.window_size - 1
                    if idx < len(self.df):
                        actual_direction = self.df.iloc[idx][f'direction_{self.prediction_horizon}']
                        actual_name = "UP" if actual_direction == 1 else "DOWN"
                        correct = self.current_action == actual_direction
                    else:
                        actual_name = "UNKNOWN"
                        correct = False
                except Exception as e:
                    self.logger.warning(f"Error rendering actual direction: {e}")
                    actual_name = "ERROR"
                    correct = False
                
                print(f"Step {self.current_step}: Predicted {action_name}, Actual {actual_name}, "
                      f"{'CORRECT' if correct else 'WRONG'}")
                
                # Fixed format specifier issue
                price_str = f"Current Price: {self.current_price:.2f}" if self.current_price is not None else "Current Price: N/A"
                future_price_str = f"Future Price: {self.future_price:.2f}" if self.future_price is not None else "Future Price: N/A"
                print(f"{price_str}, {future_price_str}")
                
                print(f"Episode Return: {self.episode_returns:.2f}, "
                      f"Accuracy: {self.episode_correct / max(1, self.current_step):.2f}")
    
    def close(self):
        """Clean up environment resources."""
        pass


# Environment Factory Function
def create_binary_prediction_env(df: pd.DataFrame, 
                               window_size: int = 20,
                               prediction_horizon: int = 1,
                               features: Optional[List[str]] = None,
                               reward_scaling: float = 1.0,
                               random_start: bool = True,
                               render_mode: Optional[str] = None) -> BinaryPredictionEnv:
    """
    Create a binary prediction environment with the given parameters.
    
    Args:
        df (pd.DataFrame): DataFrame with price and feature data
        window_size (int): Number of time steps to use as observation
        prediction_horizon (int): Number of steps ahead to predict
        features (List[str], optional): List of feature columns to use
        reward_scaling (float): Scaling factor for rewards
        random_start (bool): Whether to start at a random position
        render_mode (str, optional): Rendering mode
        
    Returns:
        BinaryPredictionEnv: The configured environment
    """
    # Validate inputs
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    
    if prediction_horizon <= 0:
        raise ValueError(f"prediction_horizon must be positive, got {prediction_horizon}")
    
    if reward_scaling <= 0:
        raise ValueError(f"reward_scaling must be positive, got {reward_scaling}")
    
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    
    try:
        # Create the environment
        env = BinaryPredictionEnv(
            df=df_copy,
            window_size=window_size,
            prediction_horizon=prediction_horizon,
            features=features,
            reward_scaling=reward_scaling,
            render_mode=render_mode
        )
        
        # Store random_start setting
        env.random_start = random_start
        
        # Monkey patch the reset function to include random_start option
        original_reset = env.reset
        
        def reset_with_random_start(seed=None, options=None):
            options = options or {}
            if 'random_start' not in options:
                options['random_start'] = env.random_start
            return original_reset(seed=seed, options=options)
        
        env.reset = reset_with_random_start
        
        # Test the environment to ensure it works
        try:
            obs, info = env.reset()
            _, _, _, _, _ = env.step(env.action_space.sample())
        except Exception as e:
            logger = get_logger(__name__)
            logger.error(f"Environment test failed: {e}")
            raise RuntimeError(f"Environment creation failed: {e}") from e
        
        return env
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Failed to create environment: {e}")
        raise


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
    df = TechnicalIndicators.add_ma(df, periods=[7, 20, 50])
    df = TechnicalIndicators.add_rsi(df)
    df = TechnicalIndicators.add_macd(df)
    df = TechnicalIndicators.add_bollinger_bands(df)
    
    # Create target columns
    for horizon in [1, 3, 5]:
        df[f'direction_{horizon}'] = (df['close'].shift(-horizon) > df['close']).astype(int)
    
    # Create the environment
    env = create_binary_prediction_env(
        df=df,
        window_size=10,
        prediction_horizon=1,
        render_mode='human'
    )
    
    # Reset the environment
    obs, info = env.reset()
    
    print(f"Observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    # Run a few random steps
    for i in range(5):
        action = env.action_space.sample()  # Random action
        obs, reward, done, truncated, info = env.step(action)
        
        print(f"\nStep {i+1}")
        print(f"Action: {action}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        env.render()
        
        if done:
            break
    
    env.close()