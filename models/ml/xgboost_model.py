"""
XGBoost model implementation for price direction prediction.
This module provides an XGBoost-based model for binary classification of price movements.
"""

import os
import logging
import pickle
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb

from models.base_model import BaseModel

# Setup logging
logger = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    """
    XGBoost model for price direction prediction.
    
    This model uses XGBoost for binary classification to predict price direction.
    """
    
    def __init__(self, model_id: str, model_params: Dict[str, Any] = None):
        """
        Initialize the XGBoost model.
        
        Args:
            model_id (str): Unique identifier for the model
            model_params (Dict[str, Any], optional): Model parameters
        """
        super().__init__(model_id, model_params)
        
        # Default model parameters
        default_params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "eta": 0.1,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_estimators": 100,
            "early_stopping_rounds": 10,
            "seed": 42
        }
        
        # Update with provided parameters
        self.model_params.update({k: v for k, v in default_params.items() if k not in self.model_params})
        
        # Initialize model and scaler
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
        logger.debug(f"XGBoost model initialized with parameters: {self.model_params}")
    
    def train(self, data: pd.DataFrame, target_col: str = 'target', 
              feature_cols: List[str] = None, test_size: float = 0.2, 
              tune_hyperparams: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Train the XGBoost model on the provided data.
        
        Args:
            data (pd.DataFrame): Training data
            target_col (str): Name of the target column (1 for up, 0 for down)
            feature_cols (List[str], optional): List of feature column names
            test_size (float): Proportion of data to use for testing
            tune_hyperparams (bool): Whether to perform hyperparameter tuning
            **kwargs: Additional training parameters
            
        Returns:
            Dict[str, Any]: Training metrics
        """
        logger.info(f"Starting training for XGBoost model {self.model_id}")
        
        start_time = time.time()
        
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
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        # Scale features
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create DMatrix objects for XGBoost
        dtrain = xgb.DMatrix(X_train_scaled, label=y_train, feature_names=feature_cols)
        dtest = xgb.DMatrix(X_test_scaled, label=y_test, feature_names=feature_cols)
        
        # Hyperparameter tuning if requested
        if tune_hyperparams:
            self._tune_hyperparameters(X_train_scaled, y_train)
        
        # Set training parameters
        params = {k: v for k, v in self.model_params.items() 
                 if k not in ['n_estimators', 'early_stopping_rounds']}
        
        # Train the model
        evals_result = {}
        self.model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=self.model_params.get('n_estimators', 100),
            evals=[(dtrain, 'train'), (dtest, 'test')],
            early_stopping_rounds=self.model_params.get('early_stopping_rounds', 10),
            evals_result=evals_result,
            verbose_eval=False
        )
        
        # Get best iteration
        best_iteration = self.model.best_iteration
        
        # Evaluate on test data
        y_pred_proba = self.model.predict(dtest)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "auc": roc_auc_score(y_test, y_pred_proba),
            "train_loss": evals_result['train']['logloss'][-1],
            "test_loss": evals_result['test']['logloss'][-1],
            "best_iteration": best_iteration,
            "training_time": time.time() - start_time,
            "n_samples": len(X),
            "n_features": len(feature_cols)
        }
        
        # Update model metadata
        self.is_trained = True
        self.last_training_date = pd.Timestamp.now()
        self.metadata["feature_names"] = feature_cols
        self.metadata["metrics"] = metrics
        self.metadata["n_samples"] = len(X)
        self.metadata["best_iteration"] = best_iteration
        
        logger.info(f"XGBoost model {self.model_id} trained in {metrics['training_time']:.2f}s "
                   f"with accuracy {metrics['accuracy']:.4f}, AUC {metrics['auc']:.4f}")
        
        return metrics
    
    def _tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Perform hyperparameter tuning using grid search.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training targets
        """
        logger.info(f"Performing hyperparameter tuning for model {self.model_id}")
        
        # Define parameter grid
        param_grid = {
            'max_depth': [3, 5, 7],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'eta': [0.01, 0.1, 0.3]
        }
        
        # Create base model
        xgb_model = xgb.XGBClassifier(
            objective=self.model_params.get('objective', 'binary:logistic'),
            eval_metric=self.model_params.get('eval_metric', 'logloss'),
            use_label_encoder=False,
            seed=self.model_params.get('seed', 42)
        )
        
        # Grid search
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            scoring='roc_auc',
            cv=3,
            verbose=1,
            n_jobs=-1
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Update model parameters with best parameters
        best_params = grid_search.best_params_
        self.model_params.update(best_params)
        
        logger.info(f"Best parameters from tuning: {best_params}")
    
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
        
        # Create DMatrix
        dmatrix = xgb.DMatrix(X_scaled, feature_names=self.feature_names)
        
        # Generate predictions
        try:
            y_pred_proba = self.model.predict(dmatrix)
            return y_pred_proba
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
        return (y_pred_proba > threshold).astype(int)
    
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
        y_true = data[target_col].values
        y_pred_proba = self.predict(data)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "auc": roc_auc_score(y_true, y_pred_proba),
            "n_samples": len(data)
        }
        
        logger.info(f"Evaluation results for model {self.model_id}: "
                   f"Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}")
        
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
        
        # Save XGBoost model
        self.model.save_model(path)
        
        # Save feature names
        with open(os.path.join(model_dir, f"{model_name}.features"), 'wb') as f:
            pickle.dump(self.feature_names, f)
        
        # Save scaler
        with open(os.path.join(model_dir, f"{model_name}.scaler"), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        logger.debug(f"XGBoost model saved to {path}")
    
    def _load_model(self, path: str) -> None:
        """
        Load the model from a file.
        
        Args:
            path (str): Path to load the model from
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Load model
        model_dir = os.path.dirname(path)
        model_name = os.path.basename(path)
        
        # Load XGBoost model
        self.model = xgb.Booster()
        self.model.load_model(path)
        
        # Load feature names
        feature_path = os.path.join(model_dir, f"{model_name}.features")
        if os.path.exists(feature_path):
            with open(feature_path, 'rb') as f:
                self.feature_names = pickle.load(f)
        
        # Load scaler
        scaler_path = os.path.join(model_dir, f"{model_name}.scaler")
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        
        self.is_trained = True
        logger.debug(f"XGBoost model loaded from {path}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the model.
        
        Returns:
            Dict[str, float]: Feature names and their importance scores
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model is not trained. Please train the model first.")
        
        # Get feature scores
        scores = self.model.get_score(importance_type='gain')
        
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            normalized_scores = {k: v / total for k, v in scores.items()}
        else:
            normalized_scores = scores
        
        return normalized_scores