"""
Model ensemble module.
This module provides functions for ensembling multiple models.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Any, Tuple, Optional

from models.base_model import BaseModel

# Setup logging
logger = logging.getLogger(__name__)

class EnsembleModel:
    """
    Ensemble model class.
    
    This class provides functionality for ensembling multiple models.
    """
    
    def __init__(self, ensemble_id: str, models: Dict[str, BaseModel], weights: Optional[Dict[str, float]] = None):
        """
        Initialize the ensemble model.
        
        Args:
            ensemble_id (str): Unique identifier for the ensemble
            models (Dict[str, BaseModel]): Dictionary of models (model_id -> model)
            weights (Dict[str, float], optional): Weights for each model
        """
        self.ensemble_id = ensemble_id
        self.models = models
        
        # Initialize weights if not provided
        if weights is None:
            self.weights = {model_id: 1.0 / len(models) for model_id in models}
        else:
            # Ensure all models have weights
            self.weights = {}
            for model_id in models:
                self.weights[model_id] = weights.get(model_id, 0.0)
                
            # Normalize weights
            total_weight = sum(self.weights.values())
            if total_weight > 0:
                self.weights = {k: v / total_weight for k, v in self.weights.items()}
            else:
                # Equal weights if all weights are zero
                self.weights = {model_id: 1.0 / len(models) for model_id in models}
                
        # Validate models
        for model_id, model in models.items():
            if not model.is_trained:
                logger.warning(f"Model {model_id} is not trained. It will be excluded from ensemble.")
                self.weights[model_id] = 0.0
                
        # Renormalize weights after validation
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}
        else:
            raise ValueError("No trained models in ensemble")
            
        logger.info(f"Initialized ensemble {ensemble_id} with {len(models)} models and weights: {self.weights}")
    
    def predict(self, data: pd.DataFrame, prediction_method: str = 'weighted_average') -> np.ndarray:
        """
        Generate predictions using the ensemble.
        
        Args:
            data (pd.DataFrame): Data to predict on
            prediction_method (str): Method to use for ensembling predictions
                - 'weighted_average': Weighted average of probabilities
                - 'majority_vote': Majority vote of binary predictions
                - 'max_confidence': Use prediction from most confident model
            
        Returns:
            np.ndarray: Ensembled predictions
        """
        # Get individual model predictions
        predictions = {}
        confidences = {}
        
        for model_id, model in self.models.items():
            if self.weights[model_id] > 0:
                try:
                    # Different model types have different predict methods
                    if hasattr(model, 'predict_direction'):
                        # Get both probabilities and binary predictions
                        proba = model.predict(data)
                        binary = model.predict_direction(data)
                    else:
                        # Assume it returns probabilities by default
                        proba = model.predict(data)
                        binary = (proba > 0.5).astype(int)
                        
                    predictions[model_id] = {
                        'proba': proba,
                        'binary': binary
                    }
                    
                    # Calculate confidence as distance from decision boundary
                    confidence = np.abs(proba - 0.5) * 2  # Scale to [0, 1]
                    confidences[model_id] = confidence
                    
                except Exception as e:
                    logger.error(f"Error getting predictions from model {model_id}: {str(e)}", exc_info=True)
        
        if not predictions:
            raise RuntimeError("No valid predictions from any model")
            
        # Choose ensembling method
        if prediction_method == 'weighted_average':
            return self._weighted_average_ensemble(predictions)
        elif prediction_method == 'majority_vote':
            return self._majority_vote_ensemble(predictions)
        elif prediction_method == 'max_confidence':
            return self._max_confidence_ensemble(predictions, confidences)
        else:
            raise ValueError(f"Unknown prediction method: {prediction_method}")
    
    def _weighted_average_ensemble(self, predictions: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
        """
        Ensemble predictions using weighted average of probabilities.
        
        Args:
            predictions (Dict[str, Dict[str, np.ndarray]]): Predictions from each model
            
        Returns:
            np.ndarray: Ensembled predictions
        """
        # Initialize with NaNs
        result = np.full(len(next(iter(predictions.values()))['proba']), np.nan)
        weights_used = np.zeros_like(result)
        
        # Weighted sum of probabilities
        for model_id, preds in predictions.items():
            # Handle NaN values in predictions
            mask = ~np.isnan(preds['proba'])
            result[mask] = np.nansum([
                result[mask],
                preds['proba'][mask] * self.weights[model_id]
            ], axis=0)
            weights_used[mask] += self.weights[model_id]
            
        # Normalize by weights used (to handle NaNs)
        mask = weights_used > 0
        result[mask] = result[mask] / weights_used[mask]
        
        return result
    
    def predict_direction(self, data: pd.DataFrame, threshold: float = 0.5, 
                        prediction_method: str = 'weighted_average') -> np.ndarray:
        """
        Predict price direction (1 for up, 0 for down).
        
        Args:
            data (pd.DataFrame): Data to predict on
            threshold (float): Probability threshold for classification
            prediction_method (str): Method to use for ensembling
            
        Returns:
            np.ndarray: Binary predictions (0 or 1)
        """
        # Get probabilistic predictions
        proba = self.predict(data, prediction_method)
        
        # Convert to binary predictions
        result = np.full_like(proba, np.nan)
        mask = ~np.isnan(proba)
        result[mask] = (proba[mask] > threshold).astype(int)
        
        return result
    
    def predict_with_confidence(self, data: pd.DataFrame, 
                              prediction_method: str = 'weighted_average') -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions with confidence scores.
        
        Args:
            data (pd.DataFrame): Data to predict on
            prediction_method (str): Method to use for ensembling
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (predictions, confidence)
        """
        # Get probabilistic predictions
        proba = self.predict(data, prediction_method)
        
        # Calculate confidence as distance from decision boundary
        confidence = np.abs(proba - 0.5) * 2
        
        return proba, confidence
    
    def get_model_agreement(self, data: pd.DataFrame) -> np.ndarray:
        """
        Calculate the level of agreement between models.
        
        Args:
            data (pd.DataFrame): Data to predict on
            
        Returns:
            np.ndarray: Agreement scores (0 to 1)
        """
        # Get binary predictions from each model
        binary_predictions = {}
        
        for model_id, model in self.models.items():
            if self.weights[model_id] > 0:
                try:
                    if hasattr(model, 'predict_direction'):
                        binary = model.predict_direction(data)
                    else:
                        proba = model.predict(data)
                        binary = (proba > 0.5).astype(int)
                        
                    binary_predictions[model_id] = binary
                except Exception as e:
                    logger.error(f"Error getting predictions from model {model_id}: {str(e)}", exc_info=True)
        
        if not binary_predictions:
            raise RuntimeError("No valid predictions from any model")
            
        # Stack binary predictions
        n_samples = len(next(iter(binary_predictions.values())))
        n_models = len(binary_predictions)
        stacked = np.zeros((n_samples, n_models))
        model_indices = {model_id: i for i, model_id in enumerate(binary_predictions.keys())}
        
        for model_id, preds in binary_predictions.items():
            idx = model_indices[model_id]
            mask = ~np.isnan(preds)
            stacked[mask, idx] = preds[mask]
            
        # Calculate agreement for each sample
        agreement = np.zeros(n_samples)
        counts = np.zeros((n_samples, 2))  # Counts for [0, 1]
        
        for i in range(n_samples):
            valid_preds = stacked[i, ~np.isnan(stacked[i, :])]
            if len(valid_preds) > 0:
                # Count occurrences of each class
                for val in [0, 1]:
                    counts[i, val] = np.sum(valid_preds == val)
                
                # Agreement is the proportion of the most common class
                agreement[i] = np.max(counts[i, :]) / np.sum(counts[i, :])
            else:
                agreement[i] = np.nan
                
        return agreement
    
    def update_weights(self, performance_metrics: Dict[str, float], 
                      metric: str = 'accuracy', alpha: float = 0.5) -> None:
        """
        Update model weights based on performance metrics.
        
        Args:
            performance_metrics (Dict[str, float]): Performance metric for each model
            metric (str): Metric to use for weight updating
            alpha (float): Smoothing factor (0 = no change, 1 = complete update)
        """
        if not performance_metrics:
            logger.warning("No performance metrics provided for weight updating")
            return
            
        # Get metrics for each model
        model_metrics = {}
        for model_id in self.models:
            if model_id in performance_metrics:
                model_metrics[model_id] = performance_metrics[model_id]
            else:
                logger.warning(f"No performance metric for model {model_id}")
                model_metrics[model_id] = 0.0
                
        # Calculate new weights based on relative performance
        total_metric = sum(model_metrics.values())
        if total_metric > 0:
            new_weights = {model_id: score / total_metric for model_id, score in model_metrics.items()}
            
            # Apply smoothing
            for model_id in self.weights:
                self.weights[model_id] = (1 - alpha) * self.weights[model_id] + alpha * new_weights.get(model_id, 0)
                
            # Normalize weights
            total_weight = sum(self.weights.values())
            if total_weight > 0:
                self.weights = {k: v / total_weight for k, v in self.weights.items()}
                
            logger.info(f"Updated ensemble weights: {self.weights}")
        else:
            logger.warning(f"Total performance metric is zero, weights not updated")
    
    def _majority_vote_ensemble(self, predictions: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
        """
        Ensemble predictions using majority vote of binary predictions.
        
        Args:
            predictions (Dict[str, Dict[str, np.ndarray]]): Predictions from each model
            
        Returns:
            np.ndarray: Ensembled predictions (0 or 1)
        """
        # Get binary predictions
        binary_preds = [preds['binary'] for preds in predictions.values()]
        
        # Stack along new axis for counting
        if binary_preds:
            stacked = np.stack(binary_preds, axis=1)
            
            # Count votes for class 1, weighted by model weights
            votes = np.zeros(len(stacked))
            total_weights = np.zeros(len(stacked))
            
            for i, model_id in enumerate(predictions.keys()):
                # Handle NaN values
                mask = ~np.isnan(stacked[:, i])
                votes[mask] += stacked[mask, i] * self.weights[model_id]
                total_weights[mask] += self.weights[model_id]
            
            # Get majority class (threshold at 0.5 of total weight)
            mask = total_weights > 0
            result = np.full(len(stacked), np.nan)
            result[mask] = (votes[mask] / total_weights[mask] > 0.5).astype(float)
            
            return result
        else:
            return np.array([])
    
    def _max_confidence_ensemble(self, predictions: Dict[str, Dict[str, np.ndarray]],
                                confidences: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Ensemble predictions by selecting the prediction from the most confident model.
        
        Args:
            predictions (Dict[str, Dict[str, np.ndarray]]): Predictions from each model
            confidences (Dict[str, np.ndarray]): Confidence scores for each model
            
        Returns:
            np.ndarray: Ensembled predictions
        """
        n_samples = len(next(iter(predictions.values()))['proba'])
        result = np.full(n_samples, np.nan)
        
        # For each sample, find the model with highest confidence
        for i in range(n_samples):
            max_conf = -1
            best_pred = np.nan
            
            for model_id, conf_arr in confidences.items():
                if i < len(conf_arr) and not np.isnan(conf_arr[i]):
                    conf = conf_arr[i] * self.weights[model_id]  # Apply model weight
                    if conf > max_conf:
                        max_conf = conf
                        best_pred = predictions[model_id]['proba'][i]
            
            result[i] = best_pred
            
        return result