# BTC Price Prediction System with FinRL

A deep reinforcement learning system for predicting BTC/USDT price movements across multiple timeframes using the FinRL framework.

## Project Overview

This system predicts cryptocurrency price movements (up/down) for different time horizons (10m, 30m, 1h, 1d) using deep reinforcement learning. It provides the following capabilities:

- Data acquisition from Binance with multi-level caching
- Technical indicator calculation and feature engineering
- Market state analysis and classification
- Binary prediction environment for reinforcement learning
- (Future) Discord bot for notifications and user interaction

## Project Structure

```
event-contract-predictor-with-FinRL/
├── config/                     # Configuration management
│   ├── config_manager.py       # Configuration loading and management
│   └── default_config.yaml     # Default configuration
│
├── data/                       # Data acquisition and processing
│   ├── binance_client.py       # Binance API wrapper
│   ├── data_manager.py         # Data management and caching
│   ├── feature_engineering.py  # Technical indicators and features
│   └── market_analyzer.py      # Market state analysis
│
├── environments/               # RL environments
│   ├── binary_prediction_env.py  # Binary prediction environment
│   └── env_wrapper.py          # Environment wrappers for FinRL/SB3
│
├── utils/                      # Utility functions
│   ├── logger.py               # Logging utilities
│   ├── time_utils.py           # Time-related utilities
│   └── file_utils.py           # File handling utilities
│
├── main.py                     # Main entry point
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/event-contract-predictor-with-FinRL.git
   cd event-contract-predictor-with-FinRL
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your Binance API credentials:
   ```
   BINANCE_API_KEY=your_api_key
   BINANCE_API_SECRET=your_api_secret
   ```

## Usage

### Run the demo

The demo mode demonstrates the key features of the system:

```bash
python main.py --mode demo --symbol BTCUSDT --timeframe 1h --days 30 --cache
```

### Command-line Options

- `--mode`: Operating mode (demo, train, predict, backtest)
- `--symbol`: Trading symbol (e.g., BTCUSDT, ETHUSDT)
- `--timeframe`: Data timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d)
- `--days`: Number of days to look back
- `--cache`: Use cache for data
- `--config`: Path to configuration file
- `--debug`: Enable debug logging

## Feature Highlights

### Multi-level Data Caching

The system implements a three-tier caching strategy to minimize API calls:
- Raw data cache: JSON format for API responses
- Processed data cache: Parquet format for cleaned data
- Feature data cache: Parquet format for calculated features

### Technical Indicators

Over 20 technical indicators are available, including:
- Moving averages (SMA, EMA)
- Oscillators (RSI, Stochastic, CCI)
- Trend indicators (MACD, ADX)
- Volatility indicators (Bollinger Bands, ATR)
- Volume indicators (OBV, VWAP)

### Market State Analysis

The system detects market conditions including:
- Trend detection (strong uptrend, weak uptrend, sideways, weak downtrend, strong downtrend)
- Volatility assessment (high, medium, low)
- Support/resistance identification
- Divergence detection
- Multi-timeframe analysis

### Reinforcement Learning

The system includes a custom Gymnasium environment for binary prediction:
- Observation: Window of price data and technical indicators
- Action: Predict price direction (up/down)
- Reward: Based on prediction accuracy
- Compatible with FinRL and Stable-Baselines3

## Future Development

- Model training and evaluation
- Prediction engine
- Backtesting system
- Performance metrics and visualization
- Discord bot for notifications

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational purposes only. It is not financial advice, and it should not be used to make real investment decisions. Trading cryptocurrencies involves significant risk. Always do your own research before making investment decisions.