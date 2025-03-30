"""
DRL model trainer module.
This module provides utilities for training deep reinforcement learning models.
"""

import os
import logging
import time
from typing import Dict, List, Union, Any, Tuple, Optional, Type
import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from models.base_model import BaseModel
from environments.binary_prediction_env import BinaryPredictionEnv
from environments.env_wrapper import FinRLEnvWrapper

# Setup logging
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Utility class for training deep reinforcement learning models.
    
    This class provides utilities for creating environments, training models,
    and evaluating models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model trainer.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
        """
        self.config = config
        self.log_dir = config.get("log_dir", "./logs")
        self.model_dir = config.get("model_dir", "./models/saved")
        self.n_envs = config.get("n_envs", 1)
        self.seed = config.get("seed", None)
        
        # Create directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        logger.debug(f"Model trainer initialized with config: {config}")
    
    def create_env(self, data: pd.DataFrame, env_type: str = "binary", **kwargs) -> gym.Env:
        """
        Create a reinforcement learning environment.
        
        Args:
            data (pd.DataFrame): Data for the environment
            env_type (str): Type of environment ('binary' or 'finrl')
            **kwargs: Additional environment parameters
            
        Returns:
            gym.Env: Reinforcement learning environment
        """
        if env_type == "binary":
            # Create a binary prediction environment
            env = BinaryPredictionEnv(
                data=data,
                window_size=kwargs.get("window_size", 10),
                features=kwargs.get("features", None),
                reward_config=kwargs.get("reward_config", {}),
                seed=self.seed
            )
            logger.debug(f"Created binary prediction environment with window size {kwargs.get('window_size', 10)}")
            
        elif env_type == "finrl":
            # Create a FinRL environment wrapper
            env = FinRLEnvWrapper(
                df=data,
                time_window=kwargs.get("time_window", "1h")
            )
            logger.debug(f"Created FinRL environment wrapper with time window {kwargs.get('time_window', '1h')}")
            
        else:
            raise ValueError(f"Unsupported environment type: {env_type}")
        
        # Wrap environment with Monitor
        env = Monitor(
            env, 
            filename=os.path.join(self.log_dir, f"{env_type}_env_log"),
            info_keywords=kwargs.get("info_keywords", ())
        )
        
        return env
    
    def create_vec_env(self, data: pd.DataFrame, env_type: str = "binary", **kwargs) -> gym.Env:
        """
        Create a vectorized environment for parallel training.
        
        Args:
            data (pd.DataFrame): Data for the environment
            env_type (str): Type of environment ('binary' or 'finrl')
            **kwargs: Additional environment parameters
            
        Returns:
            gym.Env: Vectorized environment
        """
        if self.n_envs == 1:
            # Use DummyVecEnv for a single environment
            env = DummyVecEnv([lambda: self.create_env(data, env_type, **kwargs)])
            logger.debug(f"Created DummyVecEnv with 1 environment")
            
        else:
            # Use SubprocVecEnv for multiple environments
            env = SubprocVecEnv([
                lambda: self.create_env(data, env_type, **kwargs)
                for _ in range(self.n_envs)
            ])
            logger.debug(f"Created SubprocVecEnv with {self.n_envs} environments")
            
        return env
    
    def train_model(self, model: BaseModel, data: pd.DataFrame, env_type: str = "binary", 
                    eval_data: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        """
        Train a model on the provided data.
        
        Args:
            model (BaseModel): Model to train
            data (pd.DataFrame): Training data
            env_type (str): Type of environment ('binary' or 'finrl')
            eval_data (pd.DataFrame, optional): Evaluation data
            **kwargs: Additional training parameters
            
        Returns:
            Dict[str, Any]: Training metrics
        """
        logger.info(f"Starting training for model {model.model_id}")
        
        start_time = time.time()
        
        # Create training environment
        train_env = self.create_vec_env(data, env_type, **kwargs)
        
        # Create evaluation environment if eval_data is provided
        eval_env = None
        if eval_data is not None:
            eval_env = self.create_env(eval_data, env_type, **kwargs)
            
        # Train the model
        try:
            training_metrics = model.train(
                env=train_env,
                total_timesteps=kwargs.get("total_timesteps", 50000),
                eval_freq=kwargs.get("eval_freq", 1000),
                n_eval_episodes=kwargs.get("n_eval_episodes", 5),
                eval_env=eval_env,
                **{k: v for k, v in kwargs.items() if k not in ["total_timesteps", "eval_freq", "n_eval_episodes"]}
            )
            
            # Save the model
            model_path = model.save(os.path.join(self.model_dir, model.model_id))
            logger.info(f"Model saved to {model_path}")
            
            # Add training environment info to metrics
            training_metrics["env_type"] = env_type
            training_metrics["n_envs"] = self.n_envs
            training_metrics["data_shape"] = data.shape
            training_metrics["training_time"] = time.time() - start_time
            
            logger.info(f"Training completed for model {model.model_id} in {training_metrics['training_time']:.2f}s")
            
            return training_metrics
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}", exc_info=True)
            raise
        finally:
            # Close environments
            if train_env:
                train_env.close()
            if eval_env:
                eval_env.close()
    
    def evaluate_model(self, model: BaseModel, data: pd.DataFrame, env_type: str = "binary", 
                       n_episodes: int = 10, **kwargs) -> Dict[str, float]:
        """
        Evaluate a model on the provided data.
        
        Args:
            model (BaseModel): Model to evaluate
            data (pd.DataFrame): Evaluation data
            env_type (str): Type of environment ('binary' or 'finrl')
            n_episodes (int): Number of evaluation episodes
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        logger.info(f"Evaluating model {model.model_id} for {n_episodes} episodes")
        
        # Create evaluation environment
        eval_env = self.create_env(data, env_type, **kwargs)
        
        try:
            # Evaluate the model
            metrics = model.evaluate(eval_env, n_episodes=n_episodes)
            
            # Add environment info to metrics
            metrics["env_type"] = env_type
            metrics["data_shape"] = data.shape
            
            logger.info(f"Evaluation completed for model {model.model_id}: {metrics}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
            raise
        finally:
            # Close environment
            if eval_env:
                eval_env.close()
                
    def compare_models(self, models: List[BaseModel], data: pd.DataFrame, env_type: str = "binary", 
                      n_episodes: int = 10, **kwargs) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple models on the provided data.
        
        Args:
            models (List[BaseModel]): Models to compare
            data (pd.DataFrame): Evaluation data
            env_type (str): Type of environment ('binary' or 'finrl')
            n_episodes (int): Number of evaluation episodes
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dict[str, Dict[str, float]]: Evaluation metrics for each model
        """
        results = {}
        
        for model in models:
            try:
                metrics = self.evaluate_model(model, data, env_type, n_episodes, **kwargs)
                results[model.model_id] = metrics
            except Exception as e:
                logger.error(f"Error evaluating model {model.model_id}: {str(e)}")
                results[model.model_id] = {"error": str(e)}
                
        # Compile comparison summary
        summary = {
            "models": list(results.keys()),
            "best_model": max(results.items(), key=lambda x: x[1].get("mean_reward", float("-inf")))[0] if results else None,
            "mean_rewards": {model_id: metrics.get("mean_reward", None) for model_id, metrics in results.items()},
            "prediction_accuracies": {model_id: metrics.get("prediction_accuracy", None) for model_id, metrics in results.items()}
        }
        
        logger.info(f"Model comparison summary: {summary}")
        
        return results, summary