"""
Main prediction engine module.
This module provides the main prediction functionality.
"""

import os
import logging
import time
import json
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Any, Tuple, Optional

from models.base_model import BaseModel
from models.model_manager import ModelManager
from models.ensemble import EnsembleModel
from data.market_analyzer import MarketAnalyzer
from data.binance_client import BinanceClient  # Change from BinanceDataFetcher to BinanceClient

# Setup logging
logger = logging.getLogger(__name__)

class PricePredictor:
    """
    Price prediction engine.
    
    This class provides functionality for generating price movement predictions
    based on models and market data.
    """
    
    def __init__(self, config: Dict[str, Any], model_manager: ModelManager):
        """
        Initialize the price predictor.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
            model_manager (ModelManager): Model manager instance
        """
        self.config = config
        self.model_manager = model_manager
        self.data_fetcher = None
        self.market_analyzer = None
        self.prediction_history = []
        self.current_models = {}
        self.ensemble_model = None
        
        # Initialize components
        self._initialize_components()
        
        logger.info("Price predictor initialized")
    
    def _initialize_components(self) -> None:
        """
        Initialize necessary components.
        """
        # Initialize data fetcher
        if 'binance' in self.config:
            api_key = self.config['binance'].get('api_key')
            api_secret = self.config['binance'].get('api_secret')
            
            self.data_fetcher = BinanceClient(api_key, api_secret)  # Change to BinanceClient
            logger.debug("Initialized Binance data fetcher")
            
        # Initialize market analyzer
        self.market_analyzer = MarketAnalyzer()
        logger.debug("Initialized market analyzer")
        
        # Load models specified in config
        if 'models' in self.config:
            models_config = self.config['models']
            for model_config in models_config:
                try:
                    model_id = model_config.get('model_id')
                    model_type = model_config.get('model_type')
                    if model_id and model_type:
                        # Check if model exists
                        model_path = os.path.join(self.model_manager.model_dir, f"{model_id}.model")
                        if os.path.exists(model_path):
                            model = self.model_manager.load_model(model_id, model_type)
                            self.current_models[model_id] = model
                            logger.info(f"Loaded model {model_id} of type {model_type}")
                        else:
                            logger.warning(f"Model {model_id} not found, skipping")
                except Exception as e:
                    logger.error(f"Error loading model {model_config}: {str(e)}", exc_info=True)
            
            # Create ensemble if specified
            if 'ensemble' in self.config:
                ensemble_config = self.config['ensemble']
                ensemble_id = ensemble_config.get('ensemble_id', 'default_ensemble')
                model_weights = ensemble_config.get('weights', {})
                
                if self.current_models:
                    self.ensemble_model = EnsembleModel(
                        ensemble_id=ensemble_id,
                        models=self.current_models,
                        weights=model_weights
                    )
                    logger.info(f"Created ensemble model with ID {ensemble_id}")
                else:
                    logger.warning("No models available for ensemble creation")
    
    def fetch_data(self, symbol: str, timeframe: str, lookback_days: int = 30) -> pd.DataFrame:
        """
        Fetch and prepare data for prediction.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT')
            timeframe (str): Timeframe (e.g., '1h', '1d')
            lookback_days (int): Number of days to look back
            
        Returns:
            pd.DataFrame: Prepared data
        """
        if not self.data_fetcher:
            raise ValueError("Data fetcher not initialized")
            
        logger.info(f"Fetching {symbol} data for timeframe {timeframe}, lookback {lookback_days} days")
        
        # Fetch data - use get_historical_data instead of fetch_historical_data
        data = self.data_fetcher.get_historical_data(symbol, timeframe, lookback_days)
        
        # Preprocess data (should include technical indicators)
        data = self._preprocess_data(data)
        
        return data
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data for prediction.
        
        Args:
            data (pd.DataFrame): Raw data
            
        Returns:
            pd.DataFrame: Processed data
        """
        # Add technical indicators using the existing implementation from Phase 1
        from data.feature_engineering import TechnicalIndicators
        data = TechnicalIndicators.add_indicators(data)
        
        # Ensure target columns for binary prediction exist
        for horizon in [1, 3, 5]:
            target_col = f'direction_{horizon}'
            if target_col not in data.columns:
                data[target_col] = (data['close'].shift(-horizon) > data['close']).astype(int)
        
        # Analyze market state if market analyzer is available
        if self.market_analyzer:
            market_state = self.market_analyzer.detect_market_state(data)
            data['market_state'] = market_state
            logger.debug(f"Detected market state: {market_state}")
            
        return data
    
    def predict(self, data: pd.DataFrame, model_id: Optional[str] = None, 
               timeframe: str = '1h', confidence_threshold: float = 0.6) -> Dict[str, Any]:
        """
        Generate price movement prediction.
        
        Args:
            data (pd.DataFrame): Prepared data
            model_id (str, optional): Specific model ID to use
            timeframe (str): Timeframe of the prediction
            confidence_threshold (float): Minimum confidence threshold
            
        Returns:
            Dict[str, Any]: Prediction result
        """
        start_time = time.time()
        
        # Determine which model to use
        if model_id and model_id in self.current_models:
            model = self.current_models[model_id]
            logger.debug(f"Using specific model {model_id} for prediction")
            prediction_source = f"model:{model_id}"
            
            # Make prediction
            try:
                proba = model.predict(data.iloc[-1:])
                prediction_value = float(proba[0])
                direction = "up" if prediction_value > 0.5 else "down"
                confidence = abs(prediction_value - 0.5) * 2  # Scale to [0, 1]
            except Exception as e:
                logger.error(f"Error making prediction with model {model_id}: {str(e)}", exc_info=True)
                return self._create_error_prediction(str(e), timeframe)
                
        # Use ensemble if available
        elif self.ensemble_model:
            logger.debug(f"Using ensemble model for prediction")
            prediction_source = f"ensemble:{self.ensemble_model.ensemble_id}"
            
            # Make prediction with ensemble
            try:
                proba, confidence = self.ensemble_model.predict_with_confidence(data.iloc[-1:])
                prediction_value = float(proba[0])
                direction = "up" if prediction_value > 0.5 else "down"
                confidence = float(confidence[0])
                
                # Get model agreement
                agreement = float(self.ensemble_model.get_model_agreement(data.iloc[-1:]))
                
            except Exception as e:
                logger.error(f"Error making prediction with ensemble model: {str(e)}", exc_info=True)
                return self._create_error_prediction(str(e), timeframe)
                
        else:
            logger.error("No suitable model available for prediction")
            return self._create_error_prediction("No suitable model available", timeframe)
            
        # Check confidence threshold
        if confidence < confidence_threshold:
            logger.info(f"Prediction confidence {confidence:.2f} below threshold {confidence_threshold}, flagging as low confidence")
            
        # Get current price
        current_price = data['close'].iloc[-1]
        
        # Create prediction result
        prediction = {
            "symbol": data['tic'].iloc[0] if 'tic' in data.columns else "unknown",
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat(),
            "current_price": float(current_price),
            "prediction": {
                "direction": direction,
                "probability": float(prediction_value),
                "confidence": float(confidence),
                "meets_threshold": confidence >= confidence_threshold
            },
            "source": prediction_source,
            "execution_time": time.time() - start_time
        }
        
        # Add market state if available
        if 'market_state' in data.columns:
            prediction["market_state"] = data['market_state'].iloc[-1]
            
        # Add model agreement if available
        if 'agreement' in locals():
            prediction["model_agreement"] = agreement
            
        # Store prediction in history
        self.prediction_history.append(prediction)
        
        # Trim history if too long
        max_history = self.config.get('max_prediction_history', 1000)
        if len(self.prediction_history) > max_history:
            self.prediction_history = self.prediction_history[-max_history:]
            
        logger.info(f"Generated prediction: {direction} with confidence {confidence:.2f} for {timeframe}")
        
        return prediction
    
    def _create_error_prediction(self, error_message: str, timeframe: str) -> Dict[str, Any]:
        """
        Create an error prediction result.
        
        Args:
            error_message (str): Error message
            timeframe (str): Timeframe of the prediction
            
        Returns:
            Dict[str, Any]: Error prediction result
        """
        return {
            "symbol": "unknown",
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat(),
            "error": True,
            "error_message": error_message,
            "prediction": None
        }
    
    def predict_multi_timeframe(self, symbol: str, timeframes: List[str] = ['30m', '1h', '1d'], 
                               lookback_days: int = 30) -> Dict[str, Dict[str, Any]]:
        """
        Generate predictions for multiple timeframes.
        
        Args:
            symbol (str): Trading symbol
            timeframes (List[str]): List of timeframes to predict
            lookback_days (int): Number of days to look back
            
        Returns:
            Dict[str, Dict[str, Any]]: Predictions for each timeframe
        """
        results = {}
        
        for timeframe in timeframes:
            try:
                # Fetch data for timeframe
                data = self.fetch_data(symbol, timeframe, lookback_days)
                
                # Generate prediction
                prediction = self.predict(data, timeframe=timeframe)
                
                results[timeframe] = prediction
                
            except Exception as e:
                logger.error(f"Error generating prediction for {symbol} {timeframe}: {str(e)}", exc_info=True)
                results[timeframe] = self._create_error_prediction(str(e), timeframe)
                
        return results
    
    def save_prediction_history(self, filepath: str) -> None:
        """
        Save prediction history to a file.
        
        Args:
            filepath (str): Path to save the prediction history
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(self.prediction_history, f, indent=2)
                
            logger.info(f"Saved prediction history to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving prediction history: {str(e)}", exc_info=True)
    
    def load_prediction_history(self, filepath: str) -> bool:
        """
        Load prediction history from a file.
        
        Args:
            filepath (str): Path to load the prediction history from
            
        Returns:
            bool: Whether the history was successfully loaded
        """
        try:
            if not os.path.exists(filepath):
                logger.warning(f"Prediction history file not found: {filepath}")
                return False
                
            # Load from file
            with open(filepath, 'r') as f:
                self.prediction_history = json.load(f)
                
            logger.info(f"Loaded prediction history with {len(self.prediction_history)} records from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading prediction history: {str(e)}", exc_info=True)
            return False
    
    def get_prediction_performance(self, symbol: str, timeframe: str, 
                                 days_back: int = 30) -> Dict[str, float]:
        """
        Calculate performance metrics for past predictions.
        
        Args:
            symbol (str): Trading symbol
            timeframe (str): Timeframe
            days_back (int): Number of days to look back
            
        Returns:
            Dict[str, float]: Performance metrics
        """
        # Fetch historical data to verify predictions
        try:
            data = self.fetch_data(symbol, timeframe, days_back)
            
            # Filter prediction history for this symbol and timeframe
            relevant_predictions = [
                p for p in self.prediction_history 
                if p.get("symbol") == symbol and p.get("timeframe") == timeframe
            ]
            
            if not relevant_predictions:
                logger.warning(f"No prediction history for {symbol} {timeframe}")
                return {"error": "No prediction history available"}
                
            # Match predictions with outcomes
            correct_predictions = 0
            total_evaluated = 0
            confident_correct = 0
            confident_total = 0
            
            for pred in relevant_predictions:
                # Skip error predictions
                if pred.get("error", False):
                    continue
                    
                # Get prediction details
                pred_time = datetime.fromisoformat(pred["timestamp"])
                direction = pred["prediction"]["direction"]
                is_confident = pred["prediction"]["meets_threshold"]
                
                # Find closest data point after prediction
                future_data = data[pd.to_datetime(data['date']) > pred_time]
                if len(future_data) < 2:  # Need at least 2 points to evaluate
                    continue
                    
                # Get price before and after
                start_price = future_data['close'].iloc[0]
                end_price = future_data['close'].iloc[1]
                actual_direction = "up" if end_price > start_price else "down"
                
                # Check if prediction was correct
                is_correct = direction == actual_direction
                
                # Update counters
                total_evaluated += 1
                if is_correct:
                    correct_predictions += 1
                    
                if is_confident:
                    confident_total += 1
                    if is_correct:
                        confident_correct += 1
            
            # Calculate metrics
            if total_evaluated > 0:
                accuracy = correct_predictions / total_evaluated
            else:
                accuracy = 0
                
            if confident_total > 0:
                confident_accuracy = confident_correct / confident_total
            else:
                confident_accuracy = 0
                
            metrics = {
                "accuracy": accuracy,
                "confident_accuracy": confident_accuracy,
                "total_evaluated": total_evaluated,
                "confident_evaluated": confident_total,
                "evaluation_period_days": days_back
            }
            
            logger.info(f"Prediction performance for {symbol} {timeframe}: "
                       f"Accuracy {accuracy:.2f} ({correct_predictions}/{total_evaluated}), "
                       f"Confident accuracy {confident_accuracy:.2f} ({confident_correct}/{confident_total})")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating prediction performance: {str(e)}", exc_info=True)
            return {"error": str(e)}