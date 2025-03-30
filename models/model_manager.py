"""
Model management and selection module.
This module provides functionality for managing and selecting models.
"""

import os
import logging
import json
import glob
from typing import Dict, List, Union, Any, Tuple, Optional, Type
import pandas as pd
import numpy as np

from models.base_model import BaseModel
from models.drl.ppo_model import PPOModel
from models.drl.a2c_model import A2CModel
from models.ml.xgboost_model import XGBoostModel
from models.ml.lstm_model import LSTMModel

# Setup logging
logger = logging.getLogger(__name__)

class ModelManager:
    """
    Model management and selection class.
    
    This class manages model registration, loading, saving, and selection.
    """
    
    def __init__(self, model_dir: str = "./models/saved"):
        """
        Initialize the model manager.
        
        Args:
            model_dir (str): Directory for storing models
        """
        self.model_dir = model_dir
        self.models = {}  # Dictionary of loaded models
        self.model_registry = {}  # Registry of model types
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Register model types
        self._register_model_types()
        
        logger.debug(f"Model manager initialized with model directory: {model_dir}")
    
    def _register_model_types(self) -> None:
        """
        Register model types for model loading.
        """
        self.model_registry = {
            "PPOModel": PPOModel,
            "A2CModel": A2CModel,
            "XGBoostModel": XGBoostModel,
            "LSTMModel": LSTMModel
        }
        
        logger.debug(f"Registered model types: {list(self.model_registry.keys())}")
    
    def create_model(self, model_type: str, model_id: str, model_params: Dict[str, Any] = None) -> BaseModel:
        """
        Create a new model instance.
        
        Args:
            model_type (str): Type of model to create
            model_id (str): Unique identifier for the model
            model_params (Dict[str, Any], optional): Model parameters
            
        Returns:
            BaseModel: Newly created model instance
        """
        if model_type not in self.model_registry:
            raise ValueError(f"Unknown model type: {model_type}. Supported types: {list(self.model_registry.keys())}")
            
        # Check if model ID already exists
        if model_id in self.models:
            logger.warning(f"Model with ID {model_id} already exists. Overwriting.")
            
        # Create model instance
        model_class = self.model_registry[model_type]
        model = model_class(model_id=model_id, model_params=model_params or {})
        
        # Store model in dictionary
        self.models[model_id] = model
        
        logger.info(f"Created new {model_type} with ID: {model_id}")
        
        return model
    
    def get_model(self, model_id: str) -> BaseModel:
        """
        Get a model by its ID.
        
        Args:
            model_id (str): Model identifier
            
        Returns:
            BaseModel: Model instance
        """
        if model_id not in self.models:
            raise ValueError(f"Model with ID {model_id} not found")
            
        return self.models[model_id]
    
    def load_model(self, model_id: str, model_type: str) -> BaseModel:
        """
        Load a model from disk.
        
        Args:
            model_id (str): Model identifier
            model_type (str): Type of model to load
            
        Returns:
            BaseModel: Loaded model instance
        """
        if model_type not in self.model_registry:
            raise ValueError(f"Unknown model type: {model_type}. Supported types: {list(self.model_registry.keys())}")
            
        # Check if model is already loaded
        if model_id in self.models:
            logger.info(f"Model {model_id} already loaded")
            return self.models[model_id]
            
        # Construct model path
        model_path = os.path.join(self.model_dir, f"{model_id}.model")
        
        # Check if model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # Create model instance and load
        model_class = self.model_registry[model_type]
        model = model_class.load(self.model_dir, model_id)
        
        # Store model in dictionary
        self.models[model_id] = model
        
        logger.info(f"Loaded {model_type} with ID: {model_id}")
        
        return model
    
    def save_model(self, model_id: str) -> str:
        """
        Save a model to disk.
        
        Args:
            model_id (str): Model identifier
            
        Returns:
            str: Path to the saved model
        """
        if model_id not in self.models:
            raise ValueError(f"Model with ID {model_id} not found")
            
        model = self.models[model_id]
        path = model.save(self.model_dir)
        
        logger.info(f"Saved model {model_id} to {path}")
        
        return path
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all loaded models.
        
        Returns:
            List[Dict[str, Any]]: Information about loaded models
        """
        return [
            {
                "model_id": model_id,
                "model_type": model.__class__.__name__,
                "is_trained": model.is_trained,
                "creation_date": model.creation_date.isoformat(),
                "last_training_date": model.last_training_date.isoformat() if model.last_training_date else None
            }
            for model_id, model in self.models.items()
        ]
    
    def list_saved_models(self) -> List[Dict[str, Any]]:
        """
        List all saved models on disk.
        
        Returns:
            List[Dict[str, Any]]: Information about saved models
        """
        model_files = glob.glob(os.path.join(self.model_dir, "*.metadata.json"))
        
        models_info = []
        for file_path in model_files:
            try:
                with open(file_path, 'r') as f:
                    metadata = json.load(f)
                    
                models_info.append({
                    "model_id": metadata.get("model_id", "unknown"),
                    "model_type": metadata.get("model_type", "unknown"),
                    "is_trained": metadata.get("is_trained", False),
                    "creation_date": metadata.get("creation_date", "unknown"),
                    "last_training_date": metadata.get("last_training_date", None)
                })
            except Exception as e:
                logger.error(f"Error reading metadata file {file_path}: {str(e)}")
        
        return models_info
    
    def delete_model(self, model_id: str, delete_files: bool = False) -> None:
        """
        Delete a model from memory and optionally from disk.
        
        Args:
            model_id (str): Model identifier
            delete_files (bool): Whether to delete model files from disk
        """
        if model_id not in self.models:
            logger.warning(f"Model with ID {model_id} not found in memory")
            
        else:
            # Remove from memory
            model = self.models.pop(model_id)
            logger.info(f"Removed model {model_id} from memory")
            
        # Delete files if requested
        if delete_files:
            files_to_delete = glob.glob(os.path.join(self.model_dir, f"{model_id}*"))
            for file_path in files_to_delete:
                try:
                    os.remove(file_path)
                    logger.debug(f"Deleted file: {file_path}")
                except Exception as e:
                    logger.error(f"Error deleting file {file_path}: {str(e)}")
            
            logger.info(f"Deleted {len(files_to_delete)} files for model {model_id}")
    
    def select_best_model(self, model_ids: List[str], 
                          performance_data: Dict[str, Dict[str, float]], 
                          metric: str = "accuracy",
                          market_state: Optional[str] = None) -> str:
        """
        Select the best model based on performance metrics.
        
        Args:
            model_ids (List[str]): List of model IDs to choose from
            performance_data (Dict[str, Dict[str, float]]): Performance metrics for each model
            metric (str): Metric to use for selection
            market_state (str, optional): Current market state for conditional selection
            
        Returns:
            str: ID of the best model
        """
        if not model_ids:
            raise ValueError("No models provided for selection")
            
        # Filter models to those that exist in performance data
        valid_models = [model_id for model_id in model_ids if model_id in performance_data]
        
        if not valid_models:
            raise ValueError("None of the provided models have performance data")
            
        # If market state is provided, use it for selection
        if market_state:
            # Check if we have market state specific metrics
            market_state_metrics = {}
            for model_id in valid_models:
                model_metrics = performance_data[model_id]
                
                # Look for market state specific metric
                state_metric_key = f"{metric}_{market_state}"
                if state_metric_key in model_metrics:
                    market_state_metrics[model_id] = model_metrics[state_metric_key]
            
            # If we have market state specific metrics, use them
            if market_state_metrics:
                logger.debug(f"Using market state {market_state} specific metrics for model selection")
                best_model_id = max(market_state_metrics.items(), key=lambda x: x[1])[0]
                return best_model_id
        
        # Fall back to general metric if no market state metrics
        general_metrics = {}
        for model_id in valid_models:
            if metric in performance_data[model_id]:
                general_metrics[model_id] = performance_data[model_id][metric]
        
        if not general_metrics:
            raise ValueError(f"Metric '{metric}' not found in performance data for any model")
            
        # Select best model
        best_model_id = max(general_metrics.items(), key=lambda x: x[1])[0]
        
        logger.info(f"Selected model {best_model_id} as best model with {metric} = {general_metrics[best_model_id]:.4f}")
        
        return best_model_id
    
    def create_ensemble(self, model_ids: List[str], ensemble_id: str, 
                       weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Create an ensemble configuration from multiple models.
        
        Args:
            model_ids (List[str]): List of model IDs to include in ensemble
            ensemble_id (str): Identifier for the ensemble
            weights (Dict[str, float], optional): Weights for each model
            
        Returns:
            Dict[str, Any]: Ensemble configuration
        """
        # Validate models
        for model_id in model_ids:
            if model_id not in self.models:
                raise ValueError(f"Model with ID {model_id} not found")
                
        # Create default weights if not provided
        if weights is None:
            weights = {model_id: 1.0 / len(model_ids) for model_id in model_ids}
        else:
            # Validate weights
            missing_models = set(model_ids) - set(weights.keys())
            if missing_models:
                logger.warning(f"Weights not provided for models: {missing_models}")
                for model_id in missing_models:
                    weights[model_id] = 0.0
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight == 0:
                raise ValueError("Sum of weights cannot be zero")
                
            weights = {k: v / total_weight for k, v in weights.items()}
        
        # Create ensemble configuration
        ensemble_config = {
            "ensemble_id": ensemble_id,
            "model_ids": model_ids,
            "weights": weights,
            "creation_date": pd.Timestamp.now().isoformat()
        }
        
        # Save configuration
        ensemble_path = os.path.join(self.model_dir, f"{ensemble_id}.ensemble.json")
        
        with open(ensemble_path, 'w') as f:
            json.dump(ensemble_config, f, indent=2)
            
        logger.info(f"Created ensemble {ensemble_id} with {len(model_ids)} models")
        
        return ensemble_config
    
    def load_ensemble(self, ensemble_id: str) -> Dict[str, Any]:
        """
        Load an ensemble configuration.
        
        Args:
            ensemble_id (str): Ensemble identifier
            
        Returns:
            Dict[str, Any]: Ensemble configuration
        """
        ensemble_path = os.path.join(self.model_dir, f"{ensemble_id}.ensemble.json")
        
        if not os.path.exists(ensemble_path):
            raise FileNotFoundError(f"Ensemble configuration not found: {ensemble_path}")
            
        with open(ensemble_path, 'r') as f:
            ensemble_config = json.load(f)
            
        logger.info(f"Loaded ensemble {ensemble_id} with {len(ensemble_config['model_ids'])} models")
        
        return ensemble_config