"""
A2C (Advantage Actor-Critic) model implementation.
This module provides an A2C model for BTC price prediction.
"""

import os
import logging
import time
from typing import Dict, List, Union, Any, Tuple, Optional
import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from models.base_model import BaseModel
from data.feature_engineering import TechnicalIndicators

# Setup logging
logger = logging.getLogger(__name__)

class CustomA2CCallback(BaseCallback):
    """
    Custom callback for logging training information during A2C training.
    """
    
    def __init__(self, verbose=0):
        super(CustomA2CCallback, self).__init__(verbose)
        self.training_metrics = {
            "rewards": [],
            "episode_lengths": [],
            "learning_rate": [],
            "loss": [],
            "value_loss": [],
            "policy_loss": [],
            "entropy_loss": [],
            "timestamps": []
        }
        
    def _on_step(self) -> bool:
        """
        Called at each step of training.
        
        Returns:
            bool: Whether training should continue
        """
        # Log training metrics
        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
            self.training_metrics["rewards"].append(
                np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
            )
            self.training_metrics["episode_lengths"].append(
                np.mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])
            )
            
        # Log learning rate
        if hasattr(self.model, "learning_rate"):
            self.training_metrics["learning_rate"].append(self.model.learning_rate)
            
        # Log losses if available in logger
        if hasattr(self.model, "logger") and hasattr(self.model.logger, "name_to_value"):
            logger_values = self.model.logger.name_to_value
            for key, metric_key in [
                ("loss", "loss"),
                ("value_loss", "value_loss"),
                ("policy_loss", "policy_loss"),
                ("entropy", "entropy_loss")
            ]:
                if key in logger_values:
                    self.training_metrics[metric_key].append(logger_values[key])
                
        # Log timestamp
        self.training_metrics["timestamps"].append(time.time())
        
        return True

class A2CModel(BaseModel):
    """
    A2C (Advantage Actor-Critic) model for BTC price prediction.
    
    This model uses A2C from Stable-Baselines3 for training and prediction.
    """
    
    def __init__(self, model_id: str, model_params: Dict[str, Any] = None):
        """
        Initialize the A2C model.
        
        Args:
            model_id (str): Unique identifier for the model
            model_params (Dict[str, Any], optional): Model parameters
        """
        super().__init__(model_id, model_params)
        
        # Default model parameters
        default_params = {
            "policy": "MlpPolicy",
            "learning_rate": 7e-4,
            "n_steps": 5,
            "gamma": 0.99,
            "gae_lambda": 1.0,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "rms_prop_eps": 1e-5,
            "use_rms_prop": True,
            "normalize_advantage": False,
            "verbose": 1,
            "device": "cpu",
            "tensorboard_log": f"./logs/{model_id}"
        }
        
        # Update with provided parameters
        self.model_params.update({k: v for k, v in default_params.items() if k not in self.model_params})
        
        logger.debug(f"A2C model initialized with parameters: {self.model_params}")
    
    def train(self, env: gym.Env, total_timesteps: int = 100000, eval_freq: int = 1000, 
              n_eval_episodes: int = 5, **kwargs) -> Dict[str, Any]:
        """
        Train the A2C model on the provided environment.
        
        Args:
            env (gym.Env): Training environment
            total_timesteps (int): Total timesteps for training
            eval_freq (int): Evaluation frequency
            n_eval_episodes (int): Number of evaluation episodes
            **kwargs: Additional training parameters
            
        Returns:
            Dict[str, Any]: Training metrics
        """
        logger.info(f"Starting training for model {self.model_id} for {total_timesteps} timesteps")
        
        start_time = time.time()
        
        # Create A2C model if it doesn't exist
        if self.model is None:
            self.model = A2C(
                env=env,
                **self.model_params
            )
            logger.debug(f"Created new A2C model with parameters: {self.model_params}")
        
        # Setup callbacks
        custom_callback = CustomA2CCallback()
        
        # Add evaluation callback if eval environment is provided
        callbacks = [custom_callback]
        if "eval_env" in kwargs:
            eval_callback = EvalCallback(
                eval_env=kwargs["eval_env"],
                best_model_save_path=f"./models/saved/{self.model_id}/best_model",
                log_path=f"./logs/{self.model_id}/eval",
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                render=False
            )
            callbacks.append(eval_callback)
        
        # Train the model
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                **{k: v for k, v in kwargs.items() if k != "eval_env"}
            )
            logger.info(f"Training completed for model {self.model_id}")
        except Exception as e:
            logger.error(f"Error during training: {str(e)}", exc_info=True)
            raise
        
        # Update model metadata
        self.is_trained = True
        self.last_training_date = pd.Timestamp.now()
        self.training_history = custom_callback.training_metrics
        self.metadata["training_time"] = time.time() - start_time
        self.metadata["total_timesteps"] = total_timesteps
        
        # Calculate and log training summary
        training_summary = {
            "training_time": self.metadata["training_time"],
            "total_timesteps": total_timesteps,
            "final_reward_mean": np.mean(custom_callback.training_metrics["rewards"][-10:]) if custom_callback.training_metrics["rewards"] else None,
            "final_episode_length_mean": np.mean(custom_callback.training_metrics["episode_lengths"][-10:]) if custom_callback.training_metrics["episode_lengths"] else None
        }
        
        logger.info(f"Training summary: {training_summary}")
        
        return training_summary
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Generate a prediction for the given observation.
        
        Args:
            observation (np.ndarray): Observation to predict on
            deterministic (bool): Whether to make deterministic predictions
            
        Returns:
            np.ndarray: Prediction (action)
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model is not trained. Please train the model first.")
        
        try:
            action, _ = self.model.predict(observation, deterministic=deterministic)
            return action
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}", exc_info=True)
            raise
    
    def evaluate(self, env: gym.Env, n_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate the model's performance on the provided environment.
        
        Args:
            env (gym.Env): Evaluation environment
            n_episodes (int): Number of evaluation episodes
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model is not trained. Please train the model first.")
        
        logger.info(f"Evaluating model {self.model_id} for {n_episodes} episodes")
        
        # Initialize metrics
        rewards = []
        episode_lengths = []
        successful_predictions = 0
        total_predictions = 0
        
        # Evaluate for n episodes
        for i in range(n_episodes):
            obs, _ = env.reset()
            done = False
            truncated = False
            episode_reward = 0
            episode_length = 0
            
            # Run one episode
            while not (done or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                # Track prediction accuracy if provided in info
                if "correct_prediction" in info:
                    successful_predictions += int(info["correct_prediction"])
                    total_predictions += 1
            
            rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            logger.debug(f"Episode {i+1}/{n_episodes}: Reward = {episode_reward}, Length = {episode_length}")
        
        # Calculate metrics
        metrics = {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
            "mean_episode_length": np.mean(episode_lengths)
        }
        
        # Add prediction accuracy if available
        if total_predictions > 0:
            metrics["prediction_accuracy"] = successful_predictions / total_predictions
            
        logger.info(f"Evaluation results: {metrics}")
        
        return metrics
    
    def _save_model(self, path: str) -> None:
        """
        Save the A2C model to a file.
        
        Args:
            path (str): Path to save the model to
        """
        if self.model is None:
            raise ValueError("No model to save.")
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model
        self.model.save(path)
        logger.debug(f"Model saved to {path}")
    
    def _load_model(self, path: str) -> None:
        """
        Load the A2C model from a file.
        
        Args:
            path (str): Path to load the model from
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
            
        # Load the model
        self.model = A2C.load(path)
        self.is_trained = True
        logger.debug(f"Model loaded from {path}")
        
    def prepare_environment_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for training environment.
        
        Args:
            data (pd.DataFrame): Raw price data
            
        Returns:
            pd.DataFrame: Processed data with technical indicators
        """
        # Add technical indicators 
        # Use the existing TechnicalIndicators class from Phase 1
        from data.feature_engineering import TechnicalIndicators
        data = TechnicalIndicators.add_indicators(data)
        
        # Ensure target columns for binary prediction
        for horizon in [1, 3, 5]:
            target_col = f'direction_{horizon}'
            if target_col not in data.columns:
                data[target_col] = (data['close'].shift(-horizon) > data['close']).astype(int)
                
        # Handle NaN values
        data = data.dropna()
        
        return data