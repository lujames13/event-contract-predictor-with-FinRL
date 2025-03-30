"""
BTC Price Prediction System with FinRL
A deep reinforcement learning system for predicting BTC/USDT price movements.

This is the main entry point for the application.
"""

import os
import sys
import logging
import argparse
import yaml
import json
import pandas as pd
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    logger.info(f"Loaded configuration from {config_path}")
    return config

def initialize_components(config: dict):
    """
    Initialize system components.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        tuple: Initialized components
    """
    # Import components
    from data.binance_client import BinanceClient  # 修改這裡，使用 BinanceClient 而不是 BinanceDataFetcher
    from data.feature_engineering import TechnicalIndicators
    from models.model_manager import ModelManager
    from prediction.predictor import PricePredictor
    from backtest.backtest_engine import BacktestEngine
    
    # Initialize components
    binance_config = config.get('binance', {})
    api_key = binance_config.get('api_key', os.getenv('BINANCE_API_KEY'))
    api_secret = binance_config.get('api_secret', os.getenv('BINANCE_API_SECRET'))
    
    # 使用 BinanceClient 並將其賦值給 data_fetcher 變量
    data_fetcher = BinanceClient(api_key, api_secret)
    model_manager = ModelManager(config.get('model_dir', './models/saved'))
    predictor = PricePredictor(config, model_manager)
    backtest_engine = BacktestEngine(config, model_manager)
    
    logger.info("System components initialized")
    
    return data_fetcher, model_manager, predictor, backtest_engine

def fetch_and_process_data(data_fetcher, symbol: str, timeframe: str, lookback_days: int):
    """
    Fetch and process data for prediction.
    
    Args:
        data_fetcher: BinanceClient instance
        symbol (str): Trading symbol
        timeframe (str): Timeframe
        lookback_days (int): Number of days to look back
        
    Returns:
        pd.DataFrame: Processed data
    """
    logger.info(f"Fetching {symbol} data for timeframe {timeframe}, lookback {lookback_days} days")
    
    # Use get_historical_data method from BinanceClient
    data = data_fetcher.get_historical_data(symbol, timeframe, lookback_days)
    
    # Add technical indicators
    from data.feature_engineering import TechnicalIndicators
    data = TechnicalIndicators.add_indicators(data)
    
    # Add target column (price direction)
    data['target'] = (data['close'].shift(-1) > data['close']).astype(int)
    
    # Remove NaN values
    data = data.dropna()
    
    logger.info(f"Processed data shape: {data.shape}")
    
    return data

def train_model(model_manager, data: pd.DataFrame, model_type: str, model_id: str, model_params: dict = None):
    """
    Train a model on the provided data.
    
    Args:
        model_manager: ModelManager instance
        data (pd.DataFrame): Training data
        model_type (str): Type of model to train
        model_id (str): Model identifier
        model_params (dict, optional): Model parameters
        
    Returns:
        BaseModel: Trained model
    """
    # Create model
    model = model_manager.create_model(model_type, model_id, model_params)
    
    # Train based on model type
    if model_type in ['PPOModel', 'A2CModel']:
        # For RL models
        from models.drl.model_trainer import ModelTrainer
        from environments.binary_prediction_env import create_binary_prediction_env
        from environments.env_wrapper import create_sb3_env
        
        trainer = ModelTrainer({'log_dir': './logs', 'model_dir': './models/saved'})
        
        # Create environment using the factory function from Phase 1
        window_size = model_params.get('window_size', 10) if model_params else 10
        env = create_sb3_env(data, window_size=window_size, prediction_horizon=1)
        
        logger.info(f"Training {model_type} {model_id}...")
        training_metrics = trainer.train_model(model, data, 'binary', total_timesteps=50000)
        
        logger.info(f"Training completed: {training_metrics}")
        
    elif model_type == 'XGBoostModel':
        # For XGBoost model
        logger.info(f"Training XGBoost model {model_id}...")
        metrics = model.train(data, target_col='target')
        
        logger.info(f"Training completed: {metrics}")
        
    elif model_type == 'LSTMModel':
        # For LSTM model
        logger.info(f"Training LSTM model {model_id}...")
        metrics = model.train(data, target_col='target', epochs=30, batch_size=32)
        
        logger.info(f"Training completed: {metrics}")
        
    else:
        logger.error(f"Unsupported model type: {model_type}")
        raise ValueError(f"Unsupported model type: {model_type}")
        
    # Save model
    model_path = model_manager.save_model(model_id)
    logger.info(f"Saved model to {model_path}")
    
    return model

def run_predictions(predictor, data_fetcher, symbol: str, timeframes: list = None):
    """
    Run predictions for specified timeframes.
    
    Args:
        predictor: PricePredictor instance
        data_fetcher: BinanceDataFetcher instance
        symbol (str): Trading symbol
        timeframes (list, optional): List of timeframes
        
    Returns:
        dict: Prediction results
    """
    if timeframes is None:
        timeframes = ['30m', '1h', '1d']
        
    logger.info(f"Running predictions for {symbol} on timeframes: {timeframes}")
    
    results = predictor.predict_multi_timeframe(symbol, timeframes)
    
    # Print prediction summary
    for tf, pred in results.items():
        if 'error' in pred:
            logger.error(f"Error in prediction for {tf}: {pred['error_message']}")
        else:
            direction = pred['prediction']['direction']
            confidence = pred['prediction']['confidence']
            price = pred['current_price']
            logger.info(f"{tf} prediction: {direction.upper()} with {confidence:.2f} confidence, current price: {price}")
            
    return results

def run_backtest(backtest_engine, model_manager, data: pd.DataFrame, model_ids: list = None):
    """
    Run backtests for specified models.
    
    Args:
        backtest_engine: BacktestEngine instance
        model_manager: ModelManager instance
        data (pd.DataFrame): Data for backtesting
        model_ids (list, optional): List of model IDs
        
    Returns:
        dict: Backtest results
    """
    if model_ids is None:
        # Use all loaded models
        model_ids = list(model_manager.models.keys())
        
    if not model_ids:
        logger.error("No models available for backtesting")
        return {}
        
    logger.info(f"Running backtests for models: {model_ids}")
    
    models = [model_manager.get_model(model_id) for model_id in model_ids]
    results = backtest_engine.backtest_multiple_models(models, data)
    
    # Print backtest summary
    for model_id, result in results.items():
        if 'error' in result:
            logger.error(f"Error in backtest for {model_id}: {result['error']}")
        else:
            metrics = result['metrics']
            accuracy = metrics['accuracy']
            f1 = metrics['f1_weighted']
            logger.info(f"Backtest results for {model_id}: Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
    return results

def create_ensemble(model_manager, model_ids: list, weights: dict = None):
    """
    Create an ensemble from specified models.
    
    Args:
        model_manager: ModelManager instance
        model_ids (list): List of model IDs
        weights (dict, optional): Weights for each model
        
    Returns:
        dict: Ensemble configuration
    """
    ensemble_id = f"ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info(f"Creating ensemble {ensemble_id} with models: {model_ids}")
    
    ensemble_config = model_manager.create_ensemble(model_ids, ensemble_id, weights)
    
    logger.info(f"Created ensemble: {ensemble_config}")
    
    return ensemble_config

def optimize_ensemble(backtest_engine, model_manager, data: pd.DataFrame, model_ids: list):
    """
    Optimize ensemble weights.
    
    Args:
        backtest_engine: BacktestEngine instance
        model_manager: ModelManager instance
        data (pd.DataFrame): Data for optimization
        model_ids (list): List of model IDs
        
    Returns:
        dict: Optimization results
    """
    logger.info(f"Optimizing ensemble weights for models: {model_ids}")
    
    models = [model_manager.get_model(model_id) for model_id in model_ids]
    results = backtest_engine.optimize_ensemble_weights(models, data)
    
    # Print optimization summary
    logger.info(f"Best weights: {results['best_weights']}")
    logger.info(f"Best accuracy: {results['best_accuracy']:.4f}")
    
    return results

def main():
    """
    Main entry point.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='BTC Price Prediction System with FinRL')
    parser.add_argument('--mode', choices=['train', 'predict', 'backtest', 'optimize'], default='predict',
                      help='Operating mode')
    parser.add_argument('--config', default='config.yaml', help='Path to configuration file')
    parser.add_argument('--symbol', default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--timeframe', default='1h', help='Timeframe for data')
    parser.add_argument('--days', type=int, default=30, help='Number of days to look back')
    parser.add_argument('--model-type', choices=['PPOModel', 'A2CModel', 'XGBoostModel', 'LSTMModel'],
                      default='PPOModel', help='Type of model to train')
    parser.add_argument('--model-id', help='Model identifier')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Initialize components
        data_fetcher, model_manager, predictor, backtest_engine = initialize_components(config)
        
        # Fetch and process data
        data = fetch_and_process_data(data_fetcher, args.symbol, args.timeframe, args.days)
        
        # Execute specified mode
        if args.mode == 'train':
            # Train a model
            model_id = args.model_id or f"{args.model_type}_{args.symbol}_{args.timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model = train_model(model_manager, data, args.model_type, model_id)
            
        elif args.mode == 'predict':
            # Run predictions
            results = run_predictions(predictor, data_fetcher, args.symbol)
            
            # Save prediction results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            with open(f"predictions_{args.symbol}_{timestamp}.json", 'w') as f:
                json.dump(results, f, indent=2)
            
        elif args.mode == 'backtest':
            # Run backtests
            results = run_backtest(backtest_engine, model_manager, data)
            
            # Save backtest results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backtest_engine.save_results(results, f"backtest_{args.symbol}_{args.timeframe}")
            
        elif args.mode == 'optimize':
            # Optimize ensemble weights
            # Get all model IDs or use specified ones
            model_ids = model_manager.list_models()
            model_ids = [m['model_id'] for m in model_ids]
            
            results = optimize_ensemble(backtest_engine, model_manager, data, model_ids)
            
            # Create ensemble with optimized weights
            create_ensemble(model_manager, model_ids, results['best_weights'])
            
        logger.info(f"Completed {args.mode} mode")
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()