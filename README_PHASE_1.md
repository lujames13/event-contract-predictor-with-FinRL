# Phase 1: BTC Price Prediction System - Foundation Components

## System Overview

The first phase of the BTC Price Prediction System establishes the core foundation for cryptocurrency price prediction using deep reinforcement learning. This document provides a comprehensive guide to the implemented modules, their functionalities, and how to interact with them.

## Directory Structure

```
event-contract-predictor-with-FinRL/
├── config/                     # Configuration management
│   ├── __init__.py
│   ├── config_manager.py       # Configuration loading and management
│   └── default_config.yaml     # Default configuration
│
├── data/                       # Data acquisition and processing
│   ├── __init__.py
│   ├── binance_client.py       # Binance API wrapper
│   ├── data_manager.py         # Data management and caching
│   ├── feature_engineering.py  # Technical indicators and features
│   └── market_analyzer.py      # Market state analysis
│
├── environments/               # RL environments
│   ├── __init__.py
│   ├── binary_prediction_env.py  # Binary prediction environment
│   └── env_wrapper.py          # Environment wrappers for FinRL/SB3
│
├── utils/                      # Utility functions
│   ├── __init__.py
│   ├── logger.py               # Logging utilities
│   ├── time_utils.py           # Time-related utilities
│   └── file_utils.py           # File handling utilities
│
├── main.py                     # Main entry point
└── requirements.txt            # Dependencies
```

## Module Guide

### Configuration System (config/)

The configuration system manages application settings from multiple sources (YAML files and environment variables).

#### ConfigManager

```python
from config.config_manager import ConfigManager

# Load configuration with default path
config_manager = ConfigManager()  

# Access configuration categories
data_config = config_manager.get_data_config()
model_config = config_manager.get_model_config()
trading_config = config_manager.get_trading_config()
discord_config = config_manager.get_discord_config()

# Use configuration values
symbols = data_config.symbols
timeframes = data_config.timeframes
use_cache = data_config.use_cache

# Validate configuration
is_valid = config_manager.validate()
```

#### Configuration Classes

* `DataConfig`: Data-related settings (symbols, timeframes, caching)
* `ModelConfig`: Model parameters (algorithm, hyperparameters)
* `TradingConfig`: Trading settings (prediction timeframes, confidence thresholds)
* `DiscordConfig`: Discord bot settings (future implementation)

### Data Acquisition (data/)

#### BinanceClient

Handles API communication with Binance, including error handling and rate limiting.

```python
from data.binance_client import BinanceClient

# Initialize client (uses env vars for API credentials if not provided)
client = BinanceClient(api_key=None, api_secret=None)

# Get historical klines data
klines = client.get_historical_klines(
    symbol="BTCUSDT",
    interval="1h",
    start_time=datetime(2023, 1, 1),
    end_time=datetime(2023, 1, 31)
)

# Get as DataFrame with preprocessing
df = client.get_historical_klines_dataframe(
    symbol="BTCUSDT",
    interval="1h",
    start_time=datetime(2023, 1, 1),
    end_time=datetime(2023, 1, 31)
)

# Fetch multiple days (handles pagination automatically)
df = client.get_historical_data(
    symbol="BTCUSDT",
    interval="1h",
    lookback_days=30
)

# Get current price
price = client.get_current_price("BTCUSDT")

# Get multiple prices
prices = client.get_multiple_prices(["BTCUSDT", "ETHUSDT"])
```

#### DataManager

Manages data acquisition, caching, and retrieval with a multi-level cache system.

```python
from data.data_manager import DataManager

# Initialize data manager
data_manager = DataManager()

# Get historical data with caching
df = data_manager.get_historical_data(
    symbol="BTCUSDT",
    timeframe="1h",
    lookback_days=30,
    use_cache=True  # Uses config default if None
)

# Get data for multiple timeframes
timeframes_data = data_manager.get_multiple_timeframes(
    symbol="BTCUSDT",
    timeframes=["1m", "5m", "15m", "1h"],
    lookback_days=30
)

# Get current price
price = data_manager.get_current_price("BTCUSDT")

# Get cache statistics
stats = data_manager.get_cache_stats()
print(f"Cache hit rates: {stats['hit_rates']}")

# Clean old cache files
cleaned = data_manager.clean_cache(max_age_days=30)
```

### Feature Engineering (data/)

#### TechnicalIndicators

Static methods for calculating various technical indicators.

```python
from data.feature_engineering import TechnicalIndicators

# Add moving averages
df = TechnicalIndicators.add_ma(df, periods=[7, 20, 50, 200])

# Add RSI
df = TechnicalIndicators.add_rsi(df, period=14)

# Add MACD
df = TechnicalIndicators.add_macd(df)

# Add Bollinger Bands
df = TechnicalIndicators.add_bollinger_bands(df)

# Add ADX
df = TechnicalIndicators.add_adx(df)

# Add ATR
df = TechnicalIndicators.add_atr(df)

# Add Stochastic Oscillator
df = TechnicalIndicators.add_stochastic(df)

# Add On-Balance Volume
df = TechnicalIndicators.add_obv(df)

# Add Ichimoku Cloud
df = TechnicalIndicators.add_ichimoku(df)
```

#### FeatureGenerator

Generates feature sets for machine learning models.

```python
from data.feature_engineering import FeatureGenerator

# Initialize feature generator
feature_generator = FeatureGenerator()

# Generate basic features
df_basic = feature_generator.generate_basic_features(df)

# Generate advanced features
df_advanced = feature_generator.generate_advanced_features(df)

# Generate price-based features
df_price = feature_generator.generate_price_features(df)

# Generate volume-based features
df_volume = feature_generator.generate_volume_features(df)

# Generate target columns for supervised learning
df_targets = feature_generator.generate_target_columns(
    df, predict_periods=[1, 3, 5]
)

# Generate complete feature set
df_features = feature_generator.generate_all_features(
    df,
    include_targets=True,
    predict_periods=[1, 3, 5],
    normalize=True
)
```

### Market Analysis (data/)

#### MarketAnalyzer

Analyzes market states, trends, and conditions.

```python
from data.market_analyzer import MarketAnalyzer

# Initialize market analyzer
market_analyzer = MarketAnalyzer()

# Analyze price trend
trend_analysis = market_analyzer.analyze_trend(df)
print(f"Trend: {trend_analysis['trend']}")
print(f"Strength: {trend_analysis['strength']}")

# Analyze volatility
volatility_analysis = market_analyzer.analyze_volatility(df)
print(f"Volatility: {volatility_analysis['volatility']}")

# Comprehensive market state analysis
market_state = market_analyzer.analyze_market_state(df)
print(f"Market state: {market_state['state']}")

# Analyze support and resistance levels
sr_levels = market_analyzer.analyze_support_resistance(df)
print(f"Support levels: {sr_levels['support']}")
print(f"Resistance levels: {sr_levels['resistance']}")

# Detect divergences
divergences = market_analyzer.detect_divergences(df)
for name, detected in divergences.items():
    print(f"{name}: {'Detected' if detected else 'Not detected'}")

# Analyze multiple timeframes
mtf_analysis = market_analyzer.analyze_multi_timeframe(timeframes_data)
print(f"Alignment: {mtf_analysis['alignment']}")
print(f"Strength: {mtf_analysis['strength']}")

# Predict optimal timeframe for trading
optimal_tf = market_analyzer.predict_optimal_timeframe(timeframes_data)
```

### Reinforcement Learning Environment (environments/)

#### BinaryPredictionEnv

Custom Gymnasium environment for binary (up/down) price prediction.

```python
from environments.binary_prediction_env import create_binary_prediction_env

# Create environment
env = create_binary_prediction_env(
    df=df_features,
    window_size=20,            # Number of time steps to use as observation
    prediction_horizon=1,      # Steps ahead to predict
    features=None,             # Features to use (None = default)
    reward_scaling=1.0,        # Reward scaling factor
    random_start=True,         # Start at random position during training
    render_mode='human'        # For visualization during debugging
)

# Reset environment
observation, info = env.reset(seed=42)

# Run a step with action (0=Down, 1=Up)
action = 1  # Predict price will go up
observation, reward, terminated, truncated, info = env.step(action)

# Access environment info
print(f"Current price: {info['current_price']}")
print(f"Future price: {info['future_price']}")
print(f"Accuracy: {info['accuracy']}")

# Clean up
env.close()
```

#### Environment Wrappers

Wrappers for compatibility with external RL libraries.

```python
from environments.env_wrapper import create_finrl_env, create_sb3_env

# Create FinRL-compatible environment
env_finrl = create_finrl_env(
    df=df_features,
    window_size=20,
    prediction_horizon=1
)

# Create Stable-Baselines3-compatible environment
env_sb3 = create_sb3_env(
    df=df_features,
    window_size=20,
    prediction_horizon=1
)
```

### Utilities (utils/)

#### Logger

```python
from utils.logger import get_logger, configure_logging

# Configure logging
configure_logging(log_to_console=True, log_to_file=True, log_level="DEBUG")

# Get a logger
logger = get_logger("my_module")

# Log messages
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
```

#### Time Utilities

```python
from utils.time_utils import (
    get_timestamp_ms, get_datetime_from_ms, timeframe_to_seconds,
    get_previous_timeframe, align_time_to_timeframe
)

# Convert datetime to ms timestamp
timestamp = get_timestamp_ms(datetime.now())

# Convert ms timestamp to datetime
dt = get_datetime_from_ms(timestamp)

# Convert timeframe to seconds
seconds = timeframe_to_seconds("1h")  # 3600

# Get previous timeframe start
prev_tf = get_previous_timeframe(datetime.now(), "1h")

# Align time to timeframe start
aligned = align_time_to_timeframe(datetime.now(), "1h")
```

#### File Utilities

```python
from utils.file_utils import (
    ensure_directory, get_cache_filename, save_to_json, load_from_json,
    is_cache_valid, clean_old_cache, save_dataframe, load_dataframe
)

# Create directory if it doesn't exist
ensure_directory("cache/data")

# Generate standardized cache filename
cache_path = get_cache_filename(
    symbol="BTCUSDT",
    timeframe="1h",
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 1, 31),
    cache_dir="cache/data"
)

# Save data to JSON
save_to_json(data, cache_path)

# Load data from JSON
data = load_from_json(cache_path)

# Check if cache is valid (not expired)
if is_cache_valid(cache_path, expiry_hours=24):
    # Use cached data
    pass

# Save DataFrame to file
save_dataframe(df, "data.parquet", format="parquet")

# Load DataFrame from file
df = load_dataframe("data.parquet")
```

## Main Application Flow

The main application demonstrates the integrated components:

```python
# main.py - Simplified flow
from utils.logger import get_logger, configure_logging
from config.config_manager import ConfigManager
from data.data_manager import DataManager
from data.feature_engineering import FeatureGenerator
from data.market_analyzer import MarketAnalyzer
from environments.binary_prediction_env import create_binary_prediction_env

def run_demo(args):
    # Initialize data manager
    data_manager = DataManager()
    
    # Fetch data
    df = data_manager.get_historical_data(args.symbol, args.timeframe, args.days)
    
    # Generate features
    feature_generator = FeatureGenerator()
    df_features = feature_generator.generate_basic_features(df)
    df_features = feature_generator.generate_target_columns(df_features)
    
    # Analyze market
    market_analyzer = MarketAnalyzer()
    analysis = market_analyzer.analyze_market_state(df_features)
    
    # Create and demonstrate environment
    env = create_binary_prediction_env(df_features, window_size=20)
    observation, info = env.reset()
    
    # Run a few random steps
    for i in range(5):
        action = env.action_space.sample()
        observation, reward, done, truncated, info = env.step(action)
        if done:
            break
    env.close()
```

## Running the System

To run the system:

```bash
# Ensure directories exist
python ensure_directories.py

# Run the demo
python main.py --mode demo --symbol BTCUSDT --timeframe 1h --days 30 --cache
```

## Next Steps for Phase 2

Phase 2 should focus on:

1. Implementing DRL models (PPO, A2C, DQN)
2. Creating model training workflows
3. Building a prediction engine
4. Developing a backtesting system
5. Implementing performance evaluation metrics

## Common Usage Patterns

### Full Data Pipeline

```python
# 1. Initialize components
data_manager = DataManager()
feature_generator = FeatureGenerator()
market_analyzer = MarketAnalyzer()

# 2. Fetch and process data
df = data_manager.get_historical_data("BTCUSDT", "1h", 30)
df_features = feature_generator.generate_all_features(df)

# 3. Analyze market
analysis = market_analyzer.analyze_market_state(df_features)

# 4. Create environment for training
env = create_sb3_env(
    df=df_features,
    window_size=20,
    prediction_horizon=1
)

# 5. (Future) Train model
# model = PPO('MlpPolicy', env, verbose=1)
# model.learn(total_timesteps=10000)

# 6. (Future) Make predictions
# obs, _ = env.reset()
# action, _ = model.predict(obs)
```

### Multi-timeframe Analysis

```python
# Get data for multiple timeframes
timeframes = ["15m", "1h", "4h", "1d"]
timeframes_data = {}

for tf in timeframes:
    df = data_manager.get_historical_data("BTCUSDT", tf, 30)
    df_features = feature_generator.generate_basic_features(df)
    timeframes_data[tf] = df_features

# Analyze multiple timeframes
mtf_analysis = market_analyzer.analyze_multi_timeframe(timeframes_data)
```

## Troubleshooting

### Common Issues

1. **Missing API Credentials**: Set `BINANCE_API_KEY` and `BINANCE_API_SECRET` in environment variables or `.env` file
2. **API Rate Limits**: The system implements automatic rate limiting, but extensive use might hit Binance limits
3. **Missing Directories**: Run `ensure_directories.py` before first execution
4. **Import Errors**: Ensure all packages are installed with `pip install -r requirements.txt`

### Debugging

Enable debug logging for more information:

```bash
python main.py --mode demo --debug
```

Check the logs in the `logs/` directory for detailed information about the execution flow.