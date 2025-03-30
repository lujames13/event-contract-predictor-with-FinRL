"""
Base model class for all prediction models in the system.
This abstract class defines the common interface for all models.
"""

import os
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Union, Any, Tuple, Optional
import numpy as np
import pandas as pd

# Setup logging
logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """
    Abstract base class for all prediction models.
    
    This class defines the interface that all models must implement.
    Models can be deep reinforcement learning models or traditional machine learning models.
    """
    
    def __init__(self, model_id: str, model_params: Dict[str, Any] = None):
        """
        Initialize the base model.
        
        Args:
            model_id (str): Unique identifier for the model
            model_params (Dict[str, Any], optional): Model parameters
        """
        self.model_id = model_id
        self.model_params = model_params or {}
        self.model = None
        self.is_trained = False
        self.training_history = {}
        self.creation_date = datetime.now()
        self.last_training_date = None
        self.metadata = {}
        
        logger.debug(f"Initialized {self.__class__.__name__} with ID: {model_id}")
    
    @abstractmethod
    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Train the model on the provided data.
        
        Args:
            data (pd.DataFrame): Training data
            **kwargs: Additional training parameters
            
        Returns:
            Dict[str, Any]: Training metrics
        """
        pass
    
    @abstractmethod
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the model.
        
        Args:
            data (pd.DataFrame): Data to make predictions on
            
        Returns:
            np.ndarray: Predictions from the model
        """
        pass
    
    @abstractmethod
    def evaluate(self, data: pd.DataFrame, targets: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model's performance on the provided data.
        
        Args:
            data (pd.DataFrame): Evaluation data
            targets (np.ndarray): True target values
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        pass
    
    def save(self, directory: str) -> str:
        """
        Save the model to the specified directory.
        
        Args:
            directory (str): Directory to save the model to
            
        Returns:
            str: Path to the saved model
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        model_path = os.path.join(directory, f"{self.model_id}.model")
        metadata_path = os.path.join(directory, f"{self.model_id}.metadata.json")
        
        self._save_model(model_path)
        self._save_metadata(metadata_path)
        
        logger.info(f"Model {self.model_id} saved to {model_path}")
        return model_path
    
    @abstractmethod
    def _save_model(self, path: str) -> None:
        """
        Save the model to a file.
        
        Args:
            path (str): Path to save the model to
        """
        pass
    
    def _save_metadata(self, path: str) -> None:
        """
        Save the model metadata to a file.
        
        Args:
            path (str): Path to save the metadata to
        """
        import json
        
        metadata = {
            "model_id": self.model_id,
            "model_type": self.__class__.__name__,
            "model_params": self.model_params,
            "is_trained": self.is_trained,
            "creation_date": self.creation_date.isoformat(),
            "last_training_date": self.last_training_date.isoformat() if self.last_training_date else None,
            "metadata": self.metadata
        }
        
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load(cls, directory: str, model_id: str) -> 'BaseModel':
        """
        Load a model from the specified directory.
        
        Args:
            directory (str): Directory to load the model from
            model_id (str): Model identifier
            
        Returns:
            BaseModel: Loaded model
        """
        import json
        
        model_path = os.path.join(directory, f"{model_id}.model")
        metadata_path = os.path.join(directory, f"{model_id}.metadata.json")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Create model instance
        model = cls(model_id=model_id, model_params=metadata.get("model_params", {}))
        
        # Load model data
        model._load_model(model_path)
        
        # Restore metadata
        model.is_trained = metadata.get("is_trained", False)
        model.creation_date = datetime.fromisoformat(metadata.get("creation_date", datetime.now().isoformat()))
        
        if metadata.get("last_training_date"):
            model.last_training_date = datetime.fromisoformat(metadata.get("last_training_date"))
            
        model.metadata = metadata.get("metadata", {})
        
        logger.info(f"Model {model_id} loaded from {model_path}")
        return model
    
    @abstractmethod
    def _load_model(self, path: str) -> None:
        """
        Load the model from a file.
        
        Args:
            path (str): Path to load the model from
        """
        pass
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the model if available.
        
        Returns:
            Dict[str, float]: Feature names and their importance scores
        """
        return {}
    
    def __str__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}(model_id={self.model_id}, trained={self.is_trained})"