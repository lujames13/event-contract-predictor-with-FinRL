"""
LSTM model implementation for price direction prediction.
This module provides an LSTM-based model for binary classification of price movements.
"""

import os
import logging
import pickle
import time
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Union, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from models.base_model import BaseModel

# Setup logging
logger = logging.getLogger(__name__)

# Set TensorFlow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=I, 2=I+W, 3=I+W+E

class LSTMModel(BaseModel):
    """
    LSTM model for price direction prediction.
    
    This model uses a Long Short-Term Memory (LSTM) neural network for
    binary classification to predict price direction.
    """
    
    def __init__(self, model_id: str, model_params: Dict[str, Any] = None):
        """
        Initialize the LSTM model.
        
        Args:
            model_id (str): Unique identifier for the model
            model_params (Dict[str, Any], optional): Model parameters
        """
        super().__init__(model_id, model_params)
        
        # Default model parameters
        default_params = {
            "sequence_length": 10,
            "lstm_units": [64, 32],
            "dropout_rate": 0.2,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 50,
            "patience": 10,
            "validation_split": 0.2,
            "use_batch_norm": True
        }
        
        # Update with provided parameters
        self.model_params.update({k: v for k, v in default_params.items() if k not in self.model_params})
        
        # Initialize model and scaler
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
        # Set random seed for reproducibility
        if "seed" in self.model_params:
            np.random.seed(self.model_params["seed"])
            tf.random.set_seed(self.model_params["seed"])
        
        logger.debug(f"LSTM model initialized with parameters: {self.model_params}")
    
    def _build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """
        Build the LSTM model architecture.
        
        Args:
            input_shape (Tuple[int, int]): Input shape (sequence_length, n_features)
            
        Returns:
            tf.keras.Model: Compiled LSTM model
        """
        model = Sequential()
        
        # LSTM layers
        lstm_units = self.model_params["lstm_units"]
        dropout_rate = self.model_params["dropout_rate"]
        use_batch_norm = self.model_params["use_batch_norm"]
        
        # First LSTM layer
        model.add(LSTM(
            units=lstm_units[0],
            input_shape=input_shape,
            return_sequences=len(lstm_units) > 1,
            activation='tanh',
            recurrent_activation='sigmoid'
        ))
        
        if use_batch_norm:
            model.add(BatchNormalization())
            
        model.add(Dropout(dropout_rate))
        
        # Additional LSTM layers
        for i, units in enumerate(lstm_units[1:]):
            return_sequences = i < len(lstm_units) - 2
            model.add(LSTM(
                units=units,
                return_sequences=return_sequences,
                activation='tanh',
                recurrent_activation='sigmoid'
            ))
            
            if use_batch_norm:
                model.add(BatchNormalization())
                
            model.add(Dropout(dropout_rate))
        
        # Output layer
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.model_params["learning_rate"]),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _prepare_sequences(self, data: np.ndarray, target: Optional[np.ndarray] = None) -> Tuple:
        """
        Prepare sequence data for LSTM input.
        
        Args:
            data (np.ndarray): Feature data
            target (np.ndarray, optional): Target data
            
        Returns:
            Tuple: Prepared X and y sequences
        """
        sequence_length = self.model_params["sequence_length"]
        n_samples = data.shape[0]
        
        # Create sequences
        X = []
        y = [] if target is not None else None
        
        for i in range(sequence_length, n_samples):
            X.append(data[i-sequence_length:i])
            if target is not None:
                y.append(target[i])
        
        X = np.array(X)
        
        if target is not None:
            y = np.array(y)
            return X, y
        else:
            return X
    
    def train(self, data: pd.DataFrame, target_col: str = 'target', 
              feature_cols: List[str] = None, test_size: float = 0.2, **kwargs) -> Dict[str, Any]:
        """
        Train the LSTM model on the provided data.
        
        Args:
            data (pd.DataFrame): Training data
            target_col (str): Name of the target column (1 for up, 0 for down)
            feature_cols (List[str], optional): List of feature column names
            test_size (float): Proportion of data to use for testing
            **kwargs: Additional training parameters
            
        Returns:
            Dict[str, Any]: Training metrics
        """
        logger.info(f"Starting training for LSTM model {self.model_id}")
        
        start_time = time.time()
        
        # Override model parameters with kwargs
        for key, value in kwargs.items():
            if key in self.model_params:
                self.model_params[key] = value
                logger.debug(f"Overriding parameter {key} with value {value}")
        
        # Validate inputs
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Determine feature columns
        if feature_cols is None:
            # Use all numeric columns except certain ones
            exclude_cols = [target_col, 'date', 'timestamp', 'day', 'tic', 'symbol']
            feature_cols = [col for col in data.columns if col not in exclude_cols and
                          np.issubdtype(data[col].dtype, np.number)]
        else:
            # Validate feature columns
            missing_cols = [col for col in feature_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Feature columns {missing_cols} not found in data")
        
        # Store feature names
        self.feature_names = feature_cols
        logger.debug(f"Using {len(feature_cols)} features: {feature_cols}")
        
        # Prepare data
        X = data[feature_cols].values
        y = data[target_col].values
        
        # Handle NaN values
        if np.isnan(X).any():
            logger.warning(f"Found {np.isnan(X).sum()} NaN values in features, filling with 0")
            X = np.nan_to_num(X, nan=0.0)
        
        # Scale features
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        # Prepare sequences
        X_seq, y_seq = self._prepare_sequences(X_scaled, y)
        logger.debug(f"Prepared sequences with shape {X_seq.shape}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y_seq, test_size=test_size, shuffle=False
        )
        
        # Build model
        if self.model is None:
            self.model = self._build_model((self.model_params["sequence_length"], X.shape[1]))
            logger.debug(f"Built LSTM model with architecture:\n{self.model.summary()}")
        
        # Set up callbacks
        checkpoint_path = f"./models/checkpoints/{self.model_id}_best.h5"
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.model_params["patience"],
                restore_best_weights=True
            ),
            ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=0
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.model_params["patience"] // 2,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            batch_size=self.model_params["batch_size"],
            epochs=self.model_params["epochs"],
            validation_split=self.model_params["validation_split"],
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test data
        test_results = self.model.evaluate(X_test, y_test, verbose=0)
        y_pred_proba = self.model.predict(X_test, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            "test_loss": test_results[0],
            "test_accuracy": test_results[1],
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "auc": roc_auc_score(y_test, y_pred_proba),
            "training_time": time.time() - start_time,
            "n_samples": len(X_seq),
            "n_features": len(feature_cols),
            "n_sequences": len(X_seq)
        }
        
        # Include training history
        metrics["history"] = {
            "loss": history.history["loss"],
            "val_loss": history.history["val_loss"],
            "accuracy": history.history["accuracy"],
            "val_accuracy": history.history["val_accuracy"]
        }
        
        # Update model metadata
        self.is_trained = True
        self.last_training_date = pd.Timestamp.now()
        self.metadata["feature_names"] = feature_cols
        self.metadata["metrics"] = {k: v for k, v in metrics.items() if k != "history"}
        self.metadata["n_samples"] = len(X_seq)
        self.metadata["n_parameters"] = self.model.count_params()
        
        logger.info(f"LSTM model {self.model_id} trained in {metrics['training_time']:.2f}s "
                   f"with accuracy {metrics['accuracy']:.4f}, AUC {metrics['auc']:.4f}")
        
        return metrics
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the trained model.
        
        Args:
            data (pd.DataFrame): Data to predict on
            
        Returns:
            np.ndarray: Predicted probabilities (of upward movement)
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model is not trained. Please train the model first.")
        
        # Get features
        if not self.feature_names:
            raise ValueError("Feature names not set. Please train the model first.")
        
        missing_cols = [col for col in self.feature_names if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")
        
        # Extract features
        X = data[self.feature_names].values
        
        # Handle NaN values
        if np.isnan(X).any():
            logger.warning(f"Found {np.isnan(X).sum()} NaN values in prediction features, filling with 0")
            X = np.nan_to_num(X, nan=0.0)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Prepare sequences
        sequence_length = self.model_params["sequence_length"]
        if len(X_scaled) < sequence_length:
            raise ValueError(f"Not enough data points for prediction. Need at least {sequence_length}, got {len(X_scaled)}")
            
        X_seq = self._prepare_sequences(X_scaled)
        
        # Generate predictions
        try:
            y_pred_proba = self.model.predict(X_seq, verbose=0).flatten()
            
            # Create result array with NaN for the initial sequence_length points
            full_result = np.full(len(data), np.nan)
            full_result[sequence_length:] = y_pred_proba
            
            return full_result
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}", exc_info=True)
            raise
    
    def predict_direction(self, data: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Predict price direction (1 for up, 0 for down).
        
        Args:
            data (pd.DataFrame): Data to predict on
            threshold (float): Probability threshold for binary classification
            
        Returns:
            np.ndarray: Binary predictions (1 for up, 0 for down)
        """
        y_pred_proba = self.predict(data)
        # Only apply threshold to non-NaN values
        result = np.full_like(y_pred_proba, np.nan)
        mask = ~np.isnan(y_pred_proba)
        result[mask] = (y_pred_proba[mask] > threshold).astype(int)
        return result
    
    def evaluate(self, data: pd.DataFrame, target_col: str = 'target') -> Dict[str, float]:
        """
        Evaluate the model's performance on the provided data.
        
        Args:
            data (pd.DataFrame): Evaluation data
            target_col (str): Name of the target column
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model is not trained. Please train the model first.")
        
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Get predictions
        y_pred_proba = self.predict(data)
        
        # Remove NaN values from predictions and corresponding targets
        mask = ~np.isnan(y_pred_proba)
        if not np.any(mask):
            raise ValueError("No valid predictions generated (all NaN)")
            
        y_pred_valid = y_pred_proba[mask]
        y_true_valid = data[target_col].values[mask]
        
        # Generate binary predictions
        y_pred = (y_pred_valid > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_true_valid, y_pred),
            "precision": precision_score(y_true_valid, y_pred, zero_division=0),
            "recall": recall_score(y_true_valid, y_pred, zero_division=0),
            "f1": f1_score(y_true_valid, y_pred, zero_division=0),
            "auc": roc_auc_score(y_true_valid, y_pred_valid),
            "n_samples": len(y_true_valid),
            "original_samples": len(data),
            "valid_ratio": len(y_true_valid) / len(data) if len(data) > 0 else 0
        }
        
        logger.info(f"Evaluation results for model {self.model_id}: "
                   f"Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}, "
                   f"Valid samples: {metrics['n_samples']}/{metrics['original_samples']}")
        
        return metrics
    
    def _save_model(self, path: str) -> None:
        """
        Save the model to a file.
        
        Args:
            path (str): Path to save the model to
        """
        if self.model is None:
            raise ValueError("No model to save.")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model and scaler
        model_dir = os.path.dirname(path)
        model_name = os.path.basename(path)
        
        # Remove .model extension if present
        if model_name.endswith('.model'):
            model_name = model_name[:-6]
        
        # Save Keras model with h5 extension
        keras_path = os.path.join(model_dir, f"{model_name}.h5")
        self.model.save(keras_path)
        
        # Save feature names
        with open(os.path.join(model_dir, f"{model_name}.features"), 'wb') as f:
            pickle.dump(self.feature_names, f)
        
        # Save scaler
        with open(os.path.join(model_dir, f"{model_name}.scaler"), 'wb') as f:
            pickle.dump(self.scaler, f)
            
        # Save model parameters
        with open(os.path.join(model_dir, f"{model_name}.params.json"), 'w') as f:
            json.dump(self.model_params, f, indent=2)
        
        logger.debug(f"LSTM model saved to {keras_path}")
    
    def _load_model(self, path: str) -> None:
        """
        Load the model from a file.
        
        Args:
            path (str): Path to load the model from
        """
        # Extract directory and model name
        model_dir = os.path.dirname(path)
        model_name = os.path.basename(path)
        
        # Remove .model extension if present
        if model_name.endswith('.model'):
            model_name = model_name[:-6]
        
        # Path to Keras model
        keras_path = os.path.join(model_dir, f"{model_name}.h5")
        
        if not os.path.exists(keras_path):
            raise FileNotFoundError(f"Model file not found: {keras_path}")
        
        # Load Keras model
        self.model = load_model(keras_path)
        
        # Load feature names
        feature_path = os.path.join(model_dir, f"{model_name}.features")
        if os.path.exists(feature_path):
            with open(feature_path, 'rb') as f:
                self.feature_names = pickle.load(f)
        else:
            logger.warning(f"Feature names file not found: {feature_path}")
        
        # Load scaler
        scaler_path = os.path.join(model_dir, f"{model_name}.scaler")
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        else:
            logger.warning(f"Scaler file not found: {scaler_path}")
            
        # Load model parameters
        params_path = os.path.join(model_dir, f"{model_name}.params.json")
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                loaded_params = json.load(f)
                self.model_params.update(loaded_params)
        else:
            logger.warning(f"Model parameters file not found: {params_path}")
        
        self.is_trained = True
        logger.debug(f"LSTM model loaded from {keras_path}")