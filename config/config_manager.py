"""
Configuration Management Module

This module handles loading, validating, and accessing configuration settings
from YAML files and environment variables.

Classes:
    DataConfig: Configuration for data fetching and storage
    ModelConfig: Configuration for ML/RL models
    TradingConfig: Configuration for trading parameters
    DiscordConfig: Configuration for Discord bot
    ConfigManager: Main configuration manager
"""

import os
import yaml
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, validator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("config_manager")


@dataclass
class DataConfig:
    """Configuration for data fetching and storage"""
    
    # Trading pairs to monitor (e.g., 'BTCUSDT')
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT"])
    
    # Timeframes to fetch and analyze
    timeframes: List[str] = field(
        default_factory=lambda: ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
    )
    
    # Days to look back for historical data
    lookback_days: Dict[str, int] = field(
        default_factory=lambda: {
            "1m": 3,    # 3 days for 1-minute data
            "5m": 7,    # 7 days for 5-minute data
            "15m": 14,  # 14 days for 15-minute data
            "30m": 30,  # 30 days for 30-minute data
            "1h": 60,   # 60 days for 1-hour data
            "4h": 120,  # 120 days for 4-hour data
            "1d": 365,  # 365 days for 1-day data
        }
    )
    
    # Cache settings
    cache_dir: str = "cache"
    use_cache: bool = True
    cache_expiry_hours: int = 24
    
    # API request settings
    max_retries: int = 3
    retry_delay_seconds: int = 1
    request_timeout_seconds: int = 10


@dataclass
class ModelConfig:
    """Configuration for ML/RL models"""
    
    # Base directory for model storage
    model_dir: str = "models"
    
    # Algorithm selection
    algorithm: str = "PPO"  # Options: "PPO", "A2C", "DQN", "XGBoost", "LSTM"
    
    # Training parameters
    batch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor
    n_steps: int = 2048
    ent_coef: float = 0.01  # Entropy coefficient
    
    # Model architecture
    hidden_sizes: List[int] = field(default_factory=lambda: [64, 64])
    activation: str = "relu"
    
    # Training control
    max_train_steps: int = 50000
    eval_freq: int = 10000
    save_freq: int = 10000
    log_freq: int = 1000
    
    # Device settings
    device: str = "cpu"  # Options: "cpu", "cuda"


@dataclass
class TradingConfig:
    """Configuration for trading parameters"""
    
    # Prediction timeframes
    prediction_timeframes: List[str] = field(
        default_factory=lambda: ["10m", "30m", "1h", "1d"]
    )
    
    # Source timeframes for each prediction timeframe
    source_timeframes: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "10m": ["1m", "3m", "5m"],
            "30m": ["5m", "15m", "30m"],
            "1h": ["15m", "30m", "1h"],
            "1d": ["1h", "4h", "1d"],
        }
    )
    
    # Confidence thresholds
    min_confidence: float = 0.6
    high_confidence: float = 0.8
    
    # Trading parameters
    initial_amount: float = 10000  # Initial capital
    max_position_size: float = 0.2  # Maximum position size as fraction of capital
    transaction_fee: float = 0.001  # 0.1% fee


@dataclass
class DiscordConfig:
    """Configuration for Discord bot"""
    
    # Discord bot token
    token: str = ""
    
    # Discord channels
    channels: Dict[str, str] = field(default_factory=dict)
    
    # Notification settings
    notify_on_prediction: bool = True
    notify_on_high_confidence: bool = True
    notification_cooldown_minutes: int = 60
    
    # Command settings
    command_prefix: str = "/"
    admin_user_ids: List[str] = field(default_factory=list)


class ConfigValidationModel(BaseModel):
    """Pydantic model for config validation"""
    
    data: Dict[str, Any]
    model: Dict[str, Any]
    trading: Dict[str, Any]
    discord: Dict[str, Any]
    
    @validator("data")
    def validate_data_config(cls, data_config):
        # Validate timeframes
        valid_timeframes = ["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"]
        for tf in data_config.get("timeframes", []):
            if tf not in valid_timeframes:
                raise ValueError(f"Invalid timeframe: {tf}. Must be one of {valid_timeframes}")
        
        # Validate symbols (simple check)
        for symbol in data_config.get("symbols", []):
            if not symbol.endswith("USDT"):
                logger.warning(f"Symbol {symbol} does not end with USDT, which may be unusual")
        
        return data_config
    
    @validator("model")
    def validate_model_config(cls, model_config):
        # Validate algorithm
        valid_algorithms = ["PPO", "A2C", "DQN", "XGBoost", "LSTM", "RandomForest"]
        algorithm = model_config.get("algorithm", "")
        if algorithm not in valid_algorithms:
            raise ValueError(f"Invalid algorithm: {algorithm}. Must be one of {valid_algorithms}")
        
        # Validate learning rate
        lr = model_config.get("learning_rate", 0)
        if lr <= 0 or lr > 1:
            raise ValueError(f"Invalid learning rate: {lr}. Must be between 0 and 1")
        
        return model_config
    
    @validator("trading")
    def validate_trading_config(cls, trading_config):
        # Validate confidence thresholds
        min_conf = trading_config.get("min_confidence", 0)
        high_conf = trading_config.get("high_confidence", 0)
        
        if min_conf < 0 or min_conf > 1:
            raise ValueError(f"min_confidence must be between 0 and 1, got {min_conf}")
        
        if high_conf < 0 or high_conf > 1:
            raise ValueError(f"high_confidence must be between 0 and 1, got {high_conf}")
        
        if min_conf > high_conf:
            raise ValueError(f"min_confidence ({min_conf}) cannot be greater than high_confidence ({high_conf})")
        
        return trading_config


class ConfigManager:
    """
    Main configuration manager that handles loading, validating, and providing
    access to configuration settings.
    """
    
    _instance = None  # Singleton instance
    
    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern"""
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config_path: str = "config/default_config.yaml"):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        # Only initialize once (singleton pattern)
        if self._initialized:
            return
        
        self.config_path = config_path
        self.logger = logging.getLogger("ConfigManager")
        
        # Default configuration objects
        self.data_config = DataConfig()
        self.model_config = ModelConfig()
        self.trading_config = TradingConfig()
        self.discord_config = DiscordConfig()
        
        # Load configuration
        self._load_config()
        self._initialized = True
    
    def _load_config(self):
        """Load configuration from file and environment variables"""
        try:
            # Load from YAML file if it exists
            if os.path.exists(self.config_path):
                self.logger.info(f"Loading configuration from {self.config_path}")
                with open(self.config_path, "r") as f:
                    config_data = yaml.safe_load(f)
                
                # Validate configuration
                validation_model = ConfigValidationModel(**config_data)
                
                # Update configuration objects with validated data
                self._update_from_dict(self.data_config, validation_model.data)
                self._update_from_dict(self.model_config, validation_model.model)
                self._update_from_dict(self.trading_config, validation_model.trading)
                self._update_from_dict(self.discord_config, validation_model.discord)
            else:
                self.logger.warning(f"Configuration file {self.config_path} not found, using defaults")
                # Ensure the config directory exists
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
                # Save default configuration
                self.save_config()
            
            # Override with environment variables
            self._load_from_env()
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            self.logger.info("Using default configuration settings")
            # Reset to defaults
            self.data_config = DataConfig()
            self.model_config = ModelConfig()
            self.trading_config = TradingConfig()
            self.discord_config = DiscordConfig()
    
    def _update_from_dict(self, config_obj, config_dict):
        """Update a configuration object from a dictionary"""
        for key, value in config_dict.items():
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        # Binance API credentials (sensitive data)
        if os.getenv("BINANCE_API_KEY"):
            self.logger.info("Loading Binance API key from environment")
            # This would typically be stored elsewhere, not in config objects directly
        
        if os.getenv("BINANCE_API_SECRET"):
            self.logger.info("Loading Binance API secret from environment")
            # This would typically be stored elsewhere, not in config objects directly
        
        # Discord bot token (sensitive data)
        if os.getenv("DISCORD_BOT_TOKEN"):
            self.logger.info("Loading Discord bot token from environment")
            self.discord_config.token = os.getenv("DISCORD_BOT_TOKEN")
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            # Prepare configuration dictionary
            config_data = {
                "data": {k: v for k, v in self.data_config.__dict__.items()},
                "model": {k: v for k, v in self.model_config.__dict__.items()},
                "trading": {k: v for k, v in self.trading_config.__dict__.items()},
                "discord": {k: v for k, v in self.discord_config.__dict__.items()},
            }
            
            # Remove sensitive data if present
            if "token" in config_data["discord"]:
                config_data["discord"]["token"] = "" if not self.discord_config.token else "[SET]"
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            # Save to file
            with open(self.config_path, "w") as f:
                yaml.dump(config_data, f, default_flow_style=False)
            
            self.logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
    
    def get_data_config(self) -> DataConfig:
        """Get data configuration"""
        return self.data_config
    
    def get_model_config(self) -> ModelConfig:
        """Get model configuration"""
        return self.model_config
    
    def get_trading_config(self) -> TradingConfig:
        """Get trading configuration"""
        return self.trading_config
    
    def get_discord_config(self) -> DiscordConfig:
        """Get Discord bot configuration"""
        return self.discord_config
    
    def validate(self) -> bool:
        """Validate the current configuration"""
        try:
            config_data = {
                "data": {k: v for k, v in self.data_config.__dict__.items()},
                "model": {k: v for k, v in self.model_config.__dict__.items()},
                "trading": {k: v for k, v in self.trading_config.__dict__.items()},
                "discord": {k: v for k, v in self.discord_config.__dict__.items()},
            }
            
            # Use pydantic for validation
            ConfigValidationModel(**config_data)
            return True
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # Get configuration
    config_manager = ConfigManager()
    
    # Access configuration
    data_config = config_manager.get_data_config()
    model_config = config_manager.get_model_config()
    trading_config = config_manager.get_trading_config()
    discord_config = config_manager.get_discord_config()
    
    # Print configuration
    print(f"Data config: {data_config}")
    print(f"Model config: {model_config}")
    print(f"Trading config: {trading_config}")
    print(f"Discord config: {discord_config}")
    
    # Validate configuration
    is_valid = config_manager.validate()
    print(f"Configuration is valid: {is_valid}")