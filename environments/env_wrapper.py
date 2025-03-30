"""
Environment Wrapper Module

This module provides wrapper classes to ensure compatibility between our custom
environments and external libraries like FinRL and Stable-Baselines3.

Classes:
    FinRLCompatWrapper: Wrapper for compatibility with FinRL
    SB3CompatWrapper: Wrapper for compatibility with Stable-Baselines3
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any

from utils.logger import get_logger
from environments.binary_prediction_env import BinaryPredictionEnv

# Setup logger
logger = get_logger(__name__)


class FinRLCompatWrapper(gym.Wrapper):
    """
    Wrapper to make our environment compatible with FinRL.
    
    FinRL has specific expectations for environment interfaces, 
    such as a certain format for observations and actions.
    """
    
    def __init__(self, env: gym.Env):
        """
        Initialize the FinRL compatibility wrapper.
        
        Args:
            env (gym.Env): The environment to wrap
        """
        super(FinRLCompatWrapper, self).__init__(env)
        self.logger = get_logger("FinRLCompatWrapper")
        
        # FinRL expects a different observation space format
        self._setup_observation_space()
        
        # Track state for FinRL compatibility
        self.state = None
        self.terminal = False
        self.turbulence = 0.0  # FinRL uses this for market turbulence
        
        self.logger.info(f"FinRL compatibility wrapper initialized with "
                        f"observation space: {self.observation_space}")
    
    def _setup_observation_space(self):
        """
        Set up the observation space for FinRL compatibility.
        
        FinRL typically expects a flat observation vector.
        """
        if isinstance(self.env.observation_space, gym.spaces.Box):
            # Get original shape
            orig_shape = self.env.observation_space.shape
            
            # For a window of observations, we flatten to 1D
            if len(orig_shape) == 2:
                flat_dim = orig_shape[0] * orig_shape[1]
                self.observation_space = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(flat_dim,),
                    dtype=np.float32
                )
            else:
                # Keep the original space if it's already 1D
                pass
        
        # Define state dimension for FinRL
        self.state_dim = self.observation_space.shape[0]
        
        # Action dimension remains the same
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.action_dim = self.env.action_space.n
        else:
            self.action_dim = self.env.action_space.shape[0]
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment and return an initial observation.
        
        Args:
            seed (int, optional): Random seed
            options (Dict, optional): Additional options
            
        Returns:
            observation: Initial observation
            info: Additional information
        """
        observation, info = self.env.reset(seed=seed, options=options)
        
        # Convert observation to FinRL format
        self.state = self._convert_observation(observation)
        self.terminal = False
        
        return self.state, info
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            observation: New observation
            reward: Reward from the action
            terminated: Whether the episode is done
            truncated: Whether the episode was truncated
            info: Additional information
        """
        # Convert action if needed
        if isinstance(self.env.action_space, gym.spaces.Discrete) and not isinstance(action, (int, np.integer)):
            # FinRL might send action as array, convert it to int
            action = int(action)
        
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Convert observation to FinRL format
        self.state = self._convert_observation(observation)
        self.terminal = terminated
        
        # Update turbulence from info if available
        if 'turbulence' in info:
            self.turbulence = info['turbulence']
        elif 'volatility' in info:
            # Use volatility as a proxy for turbulence
            self.turbulence = info['volatility']
        
        return self.state, reward, terminated, truncated, info
    
    def _convert_observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Convert observation to FinRL format.
        
        Args:
            observation (np.ndarray): Original observation
            
        Returns:
            np.ndarray: Converted observation
        """
        # If observation is 2D (window x features), flatten it
        if len(observation.shape) == 2:
            return observation.flatten().astype(np.float32)
        return observation


class SB3CompatWrapper(gym.Wrapper):
    """
    Wrapper to ensure compatibility with Stable-Baselines3.
    
    Stable-Baselines3 expects certain formats for observations and actions,
    and this wrapper ensures compatibility.
    """
    
    def __init__(self, env: gym.Env):
        """
        Initialize the Stable-Baselines3 compatibility wrapper.
        
        Args:
            env (gym.Env): The environment to wrap
        """
        super(SB3CompatWrapper, self).__init__(env)
        self.logger = get_logger("SB3CompatWrapper")
        
        # Make sure observation space is compatible with SB3
        self._check_and_adapt_spaces()
        
        self.logger.info(f"Stable-Baselines3 compatibility wrapper initialized with "
                        f"observation space: {self.observation_space}")
    
    def _check_and_adapt_spaces(self):
        """Check and adapt spaces for Stable-Baselines3 compatibility."""
        # SB3 expects observation space to have a specific shape
        if isinstance(self.env.observation_space, gym.spaces.Box):
            obs_shape = self.env.observation_space.shape
            
            # If observation is 2D (e.g., window x features), but SB3 policy expects 1D
            # For CNN policies, keep it 2D and add a channel dimension if needed
            if len(obs_shape) == 2:
                # Keep it 2D for CNN policies
                pass
            elif len(obs_shape) == 1:
                # Make sure it's compatible with MLP policies
                pass
            
            # Ensure dtype is float32 for SB3
            if self.env.observation_space.dtype != np.float32:
                self.observation_space = gym.spaces.Box(
                    low=self.env.observation_space.low,
                    high=self.env.observation_space.high,
                    shape=self.env.observation_space.shape,
                    dtype=np.float32
                )
        
        # Check action space for SB3 compatibility
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            # Discrete action space is fine for SB3
            pass
        elif isinstance(self.env.action_space, gym.spaces.Box):
            # Make sure it's float32 for SB3
            if self.env.action_space.dtype != np.float32:
                self.action_space = gym.spaces.Box(
                    low=self.env.action_space.low,
                    high=self.env.action_space.high,
                    shape=self.env.action_space.shape,
                    dtype=np.float32
                )
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment and return an initial observation.
        
        Args:
            seed (int, optional): Random seed
            options (Dict, optional): Additional options
            
        Returns:
            observation: Initial observation
            info: Additional information
        """
        observation, info = self.env.reset(seed=seed, options=options)
        
        # Convert observation to correct format
        observation = self._ensure_format(observation)
        
        return observation, info
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            observation: New observation
            reward: Reward from the action
            terminated: Whether the episode is done
            truncated: Whether the episode was truncated
            info: Additional information
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Convert observation to correct format
        observation = self._ensure_format(observation)
        
        return observation, reward, terminated, truncated, info
    
    def _ensure_format(self, observation: np.ndarray) -> np.ndarray:
        """
        Ensure the observation is in the correct format for Stable-Baselines3.
        
        Args:
            observation (np.ndarray): Original observation
            
        Returns:
            np.ndarray: Formatted observation
        """
        # Make sure observation has the right dtype
        return observation.astype(np.float32)


# Create a wrapped environment for FinRL
def create_finrl_env(df: pd.DataFrame, **kwargs) -> gym.Env:
    """
    Create a FinRL-compatible environment.
    
    Args:
        df (pd.DataFrame): DataFrame with price and feature data
        **kwargs: Additional arguments for the BinaryPredictionEnv
        
    Returns:
        gym.Env: FinRL-compatible environment
    """
    from environments.binary_prediction_env import create_binary_prediction_env
    
    # Create the base environment
    base_env = create_binary_prediction_env(df=df, **kwargs)
    
    # Wrap for FinRL compatibility
    return FinRLCompatWrapper(base_env)


# Create a wrapped environment for Stable-Baselines3
def create_sb3_env(df: pd.DataFrame, **kwargs) -> gym.Env:
    """
    Create a Stable-Baselines3-compatible environment.
    
    Args:
        df (pd.DataFrame): DataFrame with price and feature data
        **kwargs: Additional arguments for the BinaryPredictionEnv
        
    Returns:
        gym.Env: Stable-Baselines3-compatible environment
    """
    from environments.binary_prediction_env import create_binary_prediction_env
    
    # Create the base environment
    base_env = create_binary_prediction_env(df=df, **kwargs)
    
    # Wrap for SB3 compatibility
    return SB3CompatWrapper(base_env)


# Example usage
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    from data.feature_engineering import TechnicalIndicators
    
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
    
    # Create target columns
    for horizon in [1, 3, 5]:
        df[f'direction_{horizon}'] = (df['close'].shift(-horizon) > df['close']).astype(int)
    
    print("Testing FinRL compatibility wrapper...")
    env_finrl = create_finrl_env(
        df=df,
        window_size=10,
        prediction_horizon=1
    )
    
    obs_finrl, info_finrl = env_finrl.reset()
    print(f"FinRL observation shape: {obs_finrl.shape}")
    print(f"FinRL observation type: {obs_finrl.dtype}")
    
    action_finrl = env_finrl.action_space.sample()
    obs_finrl, reward_finrl, done_finrl, truncated_finrl, info_finrl = env_finrl.step(action_finrl)
    print(f"FinRL step output - reward: {reward_finrl}, done: {done_finrl}")
    
    print("\nTesting Stable-Baselines3 compatibility wrapper...")
    env_sb3 = create_sb3_env(
        df=df,
        window_size=10,
        prediction_horizon=1
    )
    
    obs_sb3, info_sb3 = env_sb3.reset()
    print(f"SB3 observation shape: {obs_sb3.shape}")
    print(f"SB3 observation type: {obs_sb3.dtype}")
    
    action_sb3 = env_sb3.action_space.sample()
    obs_sb3, reward_sb3, done_sb3, truncated_sb3, info_sb3 = env_sb3.step(action_sb3)
    print(f"SB3 step output - reward: {reward_sb3}, done: {done_sb3}")