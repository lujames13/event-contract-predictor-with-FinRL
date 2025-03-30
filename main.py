"""
BTC Price Prediction System - Main Module

This is the main entry point for the BTC price prediction system.
It demonstrates the basic functionality of the implemented components.
"""

import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from utils.logger import get_logger, configure_logging
from config.config_manager import ConfigManager
from data.binance_client import BinanceClient
from data.data_manager import DataManager
from data.feature_engineering import FeatureGenerator, TechnicalIndicators
from data.market_analyzer import MarketAnalyzer
from environments.binary_prediction_env import create_binary_prediction_env
from environments.env_wrapper import create_sb3_env

# Setup logger
logger = get_logger("main")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="BTC Price Prediction System")
    
    parser.add_argument("--mode", type=str, default="demo",
                        choices=["demo", "train", "predict", "backtest"],
                        help="Operating mode")
    
    parser.add_argument("--symbol", type=str, default="BTCUSDT",
                        help="Trading symbol")
    
    parser.add_argument("--timeframe", type=str, default="1h",
                        choices=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
                        help="Data timeframe")
    
    parser.add_argument("--days", type=int, default=30,
                        help="Number of days to look back")
    
    parser.add_argument("--cache", action="store_true",
                        help="Use cache")
    
    parser.add_argument("--config", type=str, default="config/default_config.yaml",
                        help="Configuration file path")
    
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    
    return parser.parse_args()


def setup_environment(args):
    """Set up the environment based on arguments."""
    # Configure logging
    log_level = "DEBUG" if args.debug else "INFO"
    configure_logging(log_to_console=True, log_to_file=True, log_level=log_level)
    
    # Load configuration
    config_manager = ConfigManager(args.config)
    
    # Override config with command-line arguments
    data_config = config_manager.get_data_config()
    data_config.use_cache = args.cache
    
    logger.info(f"Running in {args.mode} mode")
    logger.info(f"Symbol: {args.symbol}, Timeframe: {args.timeframe}, "
               f"Days: {args.days}, Cache: {args.cache}")
    
    return config_manager


def demo_fetch_data(data_manager, symbol, timeframe, days):
    """Demonstrate data fetching."""
    logger.info("Demo: Fetching and analyzing data")
    
    # Fetch data
    df = data_manager.get_historical_data(symbol, timeframe, days)
    
    logger.info(f"Fetched {len(df)} records for {symbol} {timeframe}")
    
    if not df.empty:
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"Price range: {df['low'].min():.2f} to {df['high'].max():.2f}")
        
        # Show first and last records
        logger.info("\nFirst record:")
        logger.info(df.iloc[0][['date', 'open', 'high', 'low', 'close', 'volume']].to_dict())
        
        logger.info("\nLast record:")
        logger.info(df.iloc[-1][['date', 'open', 'high', 'low', 'close', 'volume']].to_dict())
    
    return df


def demo_feature_engineering(df):
    """Demonstrate feature engineering."""
    logger.info("\nDemo: Feature Engineering")
    
    # Generate features
    feature_generator = FeatureGenerator()
    
    # Add basic features
    df_with_features = feature_generator.generate_basic_features(df)
    
    logger.info(f"Generated {len(df_with_features.columns) - 6} features")
    logger.info(f"Feature columns: {', '.join(list(df_with_features.columns)[:10])}...")
    
    # Add target columns
    df_with_features = feature_generator.generate_target_columns(df_with_features)
    
    # Show prediction targets
    prediction_cols = [col for col in df_with_features.columns if col.startswith('direction_')]
    logger.info(f"Prediction target columns: {', '.join(prediction_cols)}")
    
    return df_with_features


def demo_market_analysis(df_with_features):
    """Demonstrate market analysis."""
    logger.info("\nDemo: Market Analysis")
    
    market_analyzer = MarketAnalyzer()
    
    # Analyze current market state
    analysis = market_analyzer.analyze_market_state(df_with_features)
    
    logger.info(f"Market state: {analysis['state']}")
    logger.info(f"Trend: {analysis['trend']['trend']}")
    logger.info(f"Trend strength: {analysis['trend']['strength']}")
    logger.info(f"Volatility: {analysis['volatility']['volatility']}")
    logger.info(f"Volatility level: {analysis['volatility']['level']}")
    logger.info(f"Reversal probability: {analysis['reversal_probability']:.2f}")
    
    # Check for divergences
    divergences = market_analyzer.detect_divergences(df_with_features)
    logger.info("\nDivergences:")
    for name, detected in divergences.items():
        logger.info(f"{name}: {'Detected' if detected else 'Not detected'}")
    
    return analysis


def demo_environment(df_with_features):
    """Demonstrate environment setup and basic interaction."""
    logger.info("\nDemo: Environment Setup")
    
    # Create environment
    env = create_binary_prediction_env(
        df=df_with_features,
        window_size=20,
        prediction_horizon=1,
        reward_scaling=1.0,
        render_mode='human'
    )
    
    logger.info(f"Environment created with observation space {env.observation_space}")
    logger.info(f"Action space: {env.action_space}")
    
    # Reset environment
    obs, info = env.reset()
    logger.info(f"Initial observation shape: {obs.shape}")
    
    # Run a few steps
    total_reward = 0
    num_steps = min(5, env.max_steps)
    
    logger.info(f"\nRunning {num_steps} random steps:")
    
    for i in range(num_steps):
        # Random action
        action = env.action_space.sample()
        
        # Step
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        # Display results
        logger.info(f"Step {i+1}: Action: {action}, Reward: {reward}, Done: {done}")
        env.render()
        
        if done:
            break
    
    logger.info(f"Total reward: {total_reward}")
    env.close()
    
    return env


def run_demo(args):
    """Run demonstration mode."""
    logger.info("Starting demonstration")
    
    # Initialize data manager
    data_manager = DataManager()
    
    # Fetch data
    df = demo_fetch_data(data_manager, args.symbol, args.timeframe, args.days)
    
    if df.empty:
        logger.error("Failed to fetch data, exiting demo")
        return
    
    # Generate features
    df_with_features = demo_feature_engineering(df)
    
    # Analyze market
    analysis = demo_market_analysis(df_with_features)
    
    # Demonstrate environment
    env = demo_environment(df_with_features)
    
    logger.info("Demonstration completed")


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup environment
    config_manager = setup_environment(args)
    
    # Run selected mode
    if args.mode == "demo":
        run_demo(args)
    elif args.mode == "train":
        logger.info("Training mode not implemented yet")
    elif args.mode == "predict":
        logger.info("Prediction mode not implemented yet")
    elif args.mode == "backtest":
        logger.info("Backtesting mode not implemented yet")
    else:
        logger.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()