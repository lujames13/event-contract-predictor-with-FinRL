"""
Backtesting engine for evaluating model performance.
This module provides functionality for backtesting models on historical data.
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
from data.feature_engineering import TechnicalIndicators
from prediction.predictor import PricePredictor

# Setup logging
logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    Backtesting engine for evaluating model performance.
    
    This class provides functionality for backtesting models on historical data.
    """
    
    def __init__(self, config: Dict[str, Any], model_manager: ModelManager):
        """
        Initialize the backtest engine.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
            model_manager (ModelManager): Model manager instance
        """
        self.config = config
        self.model_manager = model_manager
        self.results_dir = config.get('results_dir', './backtest_results')
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
        logger.info("Backtest engine initialized")
    
    def backtest_model(self, model: BaseModel, data: pd.DataFrame, target_col: str = 'target',
                     window_size: int = 10, start_index: Optional[int] = None, 
                     end_index: Optional[int] = None) -> Dict[str, Any]:
        """
        Backtest a model on historical data.
        
        Args:
            model (BaseModel): Model to backtest
            data (pd.DataFrame): Historical data
            target_col (str): Name of the target column
            window_size (int): Lookback window size
            start_index (int, optional): Starting index for backtest
            end_index (int, optional): Ending index for backtest
            
        Returns:
            Dict[str, Any]: Backtest results
        """
        logger.info(f"Starting backtest for model {model.model_id}")
        
        start_time = time.time()
        
        # Ensure the target column exists
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
            
        # Set backtest range
        if start_index is None:
            start_index = window_size  # Start after initial window
        if end_index is None:
            end_index = len(data)
            
        # Validate indices
        if start_index < window_size:
            logger.warning(f"Start index {start_index} less than window size {window_size}, setting to {window_size}")
            start_index = window_size
            
        if end_index > len(data):
            logger.warning(f"End index {end_index} greater than data length {len(data)}, setting to {len(data)}")
            end_index = len(data)
            
        if start_index >= end_index:
            raise ValueError(f"Start index {start_index} must be less than end index {end_index}")
            
        # Initialize results storage
        predictions = []
        targets = []
        confidences = []
        timestamps = []
        execution_times = []
        
        # Choose prediction method based on model type
        use_direction_method = hasattr(model, 'predict_direction')
        
        # Run backtest
        for i in range(start_index, end_index):
            # Get window of data for prediction
            window_data = data.iloc[i-window_size:i].copy()
            
            # Make prediction
            try:
                predict_start = time.time()
                
                if use_direction_method:
                    # Get probability and convert to binary
                    proba = model.predict(window_data.iloc[-1:])
                    if np.isnan(proba).all():
                        logger.warning(f"NaN prediction at index {i}, skipping")
                        continue
                        
                    pred = float(proba[0]) if len(proba) > 0 else np.nan
                    confidence = abs(pred - 0.5) * 2  # Scale to [0, 1]
                    direction = int(pred > 0.5)
                else:
                    # Use predict method that returns direction (0 or 1)
                    pred_array = model.predict(window_data.iloc[-1:])
                    if np.isnan(pred_array).all():
                        logger.warning(f"NaN prediction at index {i}, skipping")
                        continue
                        
                    direction = int(pred_array[0]) if len(pred_array) > 0 else np.nan
                    pred = float(direction)
                    confidence = 1.0  # Default confidence
                    
                # Get actual target
                target = data[target_col].iloc[i]
                
                # Record results
                predictions.append(pred)
                targets.append(target)
                confidences.append(confidence)
                timestamps.append(data['date'].iloc[i] if 'date' in data.columns else i)
                execution_times.append(time.time() - predict_start)
                
            except Exception as e:
                logger.error(f"Error making prediction at index {i}: {str(e)}", exc_info=True)
                continue
                
        # Calculate performance metrics
        metrics = self._calculate_metrics(predictions, targets, confidences)
        
        # Create results dictionary
        results = {
            "model_id": model.model_id,
            "model_type": model.__class__.__name__,
            "backtest_time": time.time() - start_time,
            "data_points": len(predictions),
            "start_index": start_index,
            "end_index": end_index,
            "window_size": window_size,
            "metrics": metrics,
            "avg_execution_time": np.mean(execution_times),
            "timestamp": datetime.now().isoformat()
        }
        
        # Save detailed results if configured
        if self.config.get('save_detailed_results', False):
            detailed_results = {
                "predictions": predictions,
                "targets": targets,
                "confidences": confidences,
                "timestamps": [t.isoformat() if isinstance(t, pd.Timestamp) else t for t in timestamps],
                "execution_times": execution_times
            }
            results["detailed_results"] = detailed_results
            
        logger.info(f"Backtest completed for model {model.model_id}: "
                   f"Accuracy: {metrics['accuracy']:.4f}, "
                   f"Confident accuracy: {metrics['confident_accuracy']:.4f}")
        
        return results
    
    def _calculate_metrics(self, predictions: List[float], targets: List[int], 
                         confidences: List[float]) -> Dict[str, float]:
        """
        Calculate performance metrics from backtest results.
        
        Args:
            predictions (List[float]): Predicted probabilities
            targets (List[int]): Actual target values
            confidences (List[float]): Prediction confidences
            
        Returns:
            Dict[str, float]: Performance metrics
        """
        # Convert predictions to binary directions
        binary_predictions = [1 if p > 0.5 else 0 for p in predictions]
        
        # Calculate accuracy
        correct = sum(1 for p, t in zip(binary_predictions, targets) if p == t)
        accuracy = correct / len(predictions) if predictions else 0
        
        # Calculate metrics for confident predictions
        confidence_threshold = self.config.get('confidence_threshold', 0.6)
        confident_indices = [i for i, conf in enumerate(confidences) if conf >= confidence_threshold]
        
        if confident_indices:
            confident_predictions = [binary_predictions[i] for i in confident_indices]
            confident_targets = [targets[i] for i in confident_indices]
            confident_correct = sum(1 for p, t in zip(confident_predictions, confident_targets) if p == t)
            confident_accuracy = confident_correct / len(confident_indices)
            confident_ratio = len(confident_indices) / len(predictions)
        else:
            confident_accuracy = 0
            confident_ratio = 0
            
        # Calculate precision, recall for each class
        true_positives = sum(1 for p, t in zip(binary_predictions, targets) if p == 1 and t == 1)
        false_positives = sum(1 for p, t in zip(binary_predictions, targets) if p == 1 and t == 0)
        true_negatives = sum(1 for p, t in zip(binary_predictions, targets) if p == 0 and t == 0)
        false_negatives = sum(1 for p, t in zip(binary_predictions, targets) if p == 0 and t == 1)
        
        # Precision (when model predicts up, how often is it correct)
        precision_up = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        
        # Precision (when model predicts down, how often is it correct)
        precision_down = true_negatives / (true_negatives + false_negatives) if (true_negatives + false_negatives) > 0 else 0
        
        # Recall (how many actual up movements were correctly predicted)
        recall_up = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        # Recall (how many actual down movements were correctly predicted)
        recall_down = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
        
        # F1 scores
        f1_up = 2 * (precision_up * recall_up) / (precision_up + recall_up) if (precision_up + recall_up) > 0 else 0
        f1_down = 2 * (precision_down * recall_down) / (precision_down + recall_down) if (precision_down + recall_down) > 0 else 0
        
        # Overall F1 (weighted by class frequency)
        up_ratio = sum(targets) / len(targets)
        down_ratio = 1 - up_ratio
        f1_weighted = (up_ratio * f1_up + down_ratio * f1_down)
        
        # Calculate profit potential (simplified)
        correct_direction_profit = sum(confidences[i] for i in range(len(predictions)) if binary_predictions[i] == targets[i])
        avg_direction_profit = correct_direction_profit / len(predictions) if predictions else 0
        
        metrics = {
            "accuracy": accuracy,
            "confident_accuracy": confident_accuracy,
            "confident_ratio": confident_ratio,
            "precision_up": precision_up,
            "precision_down": precision_down,
            "recall_up": recall_up,
            "recall_down": recall_down,
            "f1_up": f1_up,
            "f1_down": f1_down,
            "f1_weighted": f1_weighted,
            "avg_confidence": np.mean(confidences) if confidences else 0,
            "avg_direction_profit": avg_direction_profit,
            "up_ratio": up_ratio,
            "confident_predictions": len(confident_indices),
            "total_predictions": len(predictions)
        }
        
        return metrics