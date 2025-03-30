"""
Performance metrics for evaluating prediction models.
This module provides utility functions for calculating performance metrics.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Any, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report

# Setup logging
logger = logging.getLogger(__name__)

def calculate_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                           y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate metrics for binary classification.
    
    Args:
        y_true (np.ndarray): True labels (0 or 1)
        y_pred (np.ndarray): Predicted labels (0 or 1)
        y_prob (np.ndarray, optional): Predicted probabilities
        
    Returns:
        Dict[str, float]: Performance metrics
    """
    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Calculate AUC if probabilities are provided
    auc = None
    if y_prob is not None:
        auc = roc_auc_score(y_true, y_prob)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative predictive value
    
    # Balanced accuracy
    balanced_acc = (rec + specificity) / 2
    
    # Calculate metrics for each class
    precision_up = tp / (tp + fp) if (tp + fp) > 0 else 0
    precision_down = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    recall_up = tp / (tp + fn) if (tp + fn) > 0 else 0
    recall_down = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    f1_up = 2 * (precision_up * recall_up) / (precision_up + recall_up) if (precision_up + recall_up) > 0 else 0
    f1_down = 2 * (precision_down * recall_down) / (precision_down + recall_down) if (precision_down + recall_down) > 0 else 0
    
    # Class distribution
    up_ratio = np.mean(y_true)
    
    # Combine into a dictionary
    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "specificity": specificity,
        "npv": npv,
        "balanced_accuracy": balanced_acc,
        "precision_up": precision_up,
        "precision_down": precision_down,
        "recall_up": recall_up,
        "recall_down": recall_down,
        "f1_up": f1_up,
        "f1_down": f1_down,
        "up_ratio": up_ratio,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn)
    }
    
    return metrics

def calculate_confident_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                              confidences: np.ndarray, threshold: float = 0.6) -> Dict[str, float]:
    """
    Calculate metrics for predictions with confidence above threshold.
    
    Args:
        y_true (np.ndarray): True labels (0 or 1)
        y_pred (np.ndarray): Predicted labels (0 or 1)
        confidences (np.ndarray): Confidence scores
        threshold (float): Confidence threshold
        
    Returns:
        Dict[str, float]: Performance metrics for confident predictions
    """
    # Filter for confident predictions
    mask = confidences >= threshold
    
    if not np.any(mask):
        logger.warning(f"No predictions with confidence >= {threshold}")
        return {
            "confident_count": 0,
            "confident_ratio": 0,
            "confident_metrics": None
        }
    
    # Get confident predictions and true labels
    conf_y_true = y_true[mask]
    conf_y_pred = y_pred[mask]
    
    # Calculate metrics
    metrics = calculate_binary_metrics(conf_y_true, conf_y_pred)
    
    # Add confidence info
    metrics["confident_count"] = int(np.sum(mask))
    metrics["confident_ratio"] = float(np.mean(mask))
    metrics["avg_confidence"] = float(np.mean(confidences[mask]))
    
    return metrics

def calculate_threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, 
                              thresholds: List[float] = None) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics for different probability thresholds.
    
    Args:
        y_true (np.ndarray): True labels (0 or 1)
        y_prob (np.ndarray): Predicted probabilities
        thresholds (List[float], optional): Probability thresholds to evaluate
        
    Returns:
        Dict[str, Dict[str, float]]: Performance metrics at each threshold
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    results = {}
    
    for threshold in thresholds:
        # Convert probabilities to binary predictions
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calculate metrics
        metrics = calculate_binary_metrics(y_true, y_pred, y_prob)
        
        # Store results
        results[f"threshold_{threshold}"] = metrics
    
    return results

def calculate_market_state_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                 y_prob: Optional[np.ndarray], market_states: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics segmented by market state.
    
    Args:
        y_true (np.ndarray): True labels (0 or 1)
        y_pred (np.ndarray): Predicted labels (0 or 1)
        y_prob (np.ndarray, optional): Predicted probabilities
        market_states (np.ndarray): Market state labels
        
    Returns:
        Dict[str, Dict[str, float]]: Performance metrics for each market state
    """
    # Get unique market states
    unique_states = np.unique(market_states)
    
    results = {}
    state_counts = {}
    
    # Calculate metrics for each market state
    for state in unique_states:
        # Filter data for this market state
        mask = market_states == state
        
        if np.sum(mask) < 10:  # Skip states with too few samples
            logger.warning(f"Too few samples ({np.sum(mask)}) for market state '{state}', skipping")
            continue
            
        state_y_true = y_true[mask]
        state_y_pred = y_pred[mask]
        state_y_prob = y_prob[mask] if y_prob is not None else None
        
        # Calculate metrics
        metrics = calculate_binary_metrics(state_y_true, state_y_pred, state_y_prob)
        
        # Store results
        results[f"state_{state}"] = metrics
        state_counts[f"state_{state}"] = int(np.sum(mask))
    
    # Add overall metrics
    overall_metrics = calculate_binary_metrics(y_true, y_pred, y_prob)
    results["overall"] = overall_metrics
    
    # Add state distribution
    state_distribution = {state: count / len(y_true) for state, count in state_counts.items()}
    results["state_distribution"] = state_distribution
    
    return results

def calculate_rolling_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                            window_size: int = 20) -> Dict[str, np.ndarray]:
    """
    Calculate rolling performance metrics.
    
    Args:
        y_true (np.ndarray): True labels (0 or 1)
        y_pred (np.ndarray): Predicted labels (0 or 1)
        window_size (int): Size of the rolling window
        
    Returns:
        Dict[str, np.ndarray]: Rolling performance metrics
    """
    n_samples = len(y_true)
    
    if n_samples < window_size:
        logger.warning(f"Number of samples {n_samples} less than window size {window_size}")
        return {}
    
    # Initialize arrays for metrics
    rolling_accuracy = np.full(n_samples, np.nan)
    rolling_precision = np.full(n_samples, np.nan)
    rolling_recall = np.full(n_samples, np.nan)
    rolling_f1 = np.full(n_samples, np.nan)
    
    # Calculate rolling metrics
    for i in range(window_size, n_samples + 1):
        window_y_true = y_true[i-window_size:i]
        window_y_pred = y_pred[i-window_size:i]
        
        try:
            rolling_accuracy[i-1] = accuracy_score(window_y_true, window_y_pred)
            rolling_precision[i-1] = precision_score(window_y_true, window_y_pred, zero_division=0)
            rolling_recall[i-1] = recall_score(window_y_true, window_y_pred, zero_division=0)
            rolling_f1[i-1] = f1_score(window_y_true, window_y_pred, zero_division=0)
        except Exception as e:
            logger.error(f"Error calculating rolling metrics at index {i}: {str(e)}")
    
    return {
        "rolling_accuracy": rolling_accuracy,
        "rolling_precision": rolling_precision,
        "rolling_recall": rolling_recall,
        "rolling_f1": rolling_f1
    }

def calculate_directional_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                price_changes: np.ndarray) -> Dict[str, float]:
    """
    Calculate metrics that consider the magnitude of price changes.
    
    Args:
        y_true (np.ndarray): True labels (0 or 1)
        y_pred (np.ndarray): Predicted labels (0 or 1)
        price_changes (np.ndarray): Percentage price changes
        
    Returns:
        Dict[str, float]: Directional performance metrics
    """
    # Regular metrics
    metrics = calculate_binary_metrics(y_true, y_pred)
    
    # Calculate weighted metrics
    abs_changes = np.abs(price_changes)
    
    # Identify correct predictions
    correct_predictions = (y_true == y_pred)
    
    # Weighted accuracy (by absolute price change)
    weighted_correct = np.sum(correct_predictions * abs_changes)
    weighted_total = np.sum(abs_changes)
    weighted_accuracy = weighted_correct / weighted_total if weighted_total > 0 else 0
    
    # Profit factor (sum of gains on correct predictions / sum of losses on incorrect predictions)
    gains = np.sum(price_changes[correct_predictions])
    losses = np.sum(price_changes[~correct_predictions])
    profit_factor = abs(gains / losses) if losses != 0 else float('inf')
    
    # Expected return (average return per prediction)
    expected_return = np.mean(price_changes * (2 * correct_predictions - 1))
    
    # Add to metrics
    metrics.update({
        "weighted_accuracy": weighted_accuracy,
        "profit_factor": profit_factor,
        "expected_return": expected_return,
        "avg_price_change": np.mean(price_changes),
        "avg_abs_price_change": np.mean(abs_changes)
    })
    
    return metrics

def calculate_ensemble_metrics(individual_predictions: Dict[str, np.ndarray], y_true: np.ndarray, 
                             weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Calculate metrics for ensemble predictions.
    
    Args:
        individual_predictions (Dict[str, np.ndarray]): Predictions from individual models
        y_true (np.ndarray): True labels (0 or 1)
        weights (Dict[str, float], optional): Weights for each model
        
    Returns:
        Dict[str, Any]: Ensemble performance metrics
    """
    # Number of models
    n_models = len(individual_predictions)
    
    if n_models == 0:
        logger.warning("No individual predictions provided")
        return {}
    
    # Default weights
    if weights is None:
        weights = {model_id: 1.0 / n_models for model_id in individual_predictions.keys()}
    
    # Calculate ensemble prediction (weighted average)
    ensemble_proba = np.zeros_like(y_true, dtype=float)
    
    for model_id, preds in individual_predictions.items():
        if model_id in weights:
            ensemble_proba += preds * weights[model_id]
    
    # Convert to binary predictions
    ensemble_pred = (ensemble_proba > 0.5).astype(int)
    
    # Calculate metrics
    ensemble_metrics = calculate_binary_metrics(y_true, ensemble_pred, ensemble_proba)
    
    # Calculate agreement metrics
    agreement_count = np.zeros_like(y_true, dtype=int)
    
    for model_id, preds in individual_predictions.items():
        agreement_count += (preds == ensemble_pred)
    
    avg_agreement = np.mean(agreement_count) / n_models
    
    # Add agreement metrics
    ensemble_metrics.update({
        "avg_agreement": avg_agreement,
        "n_models": n_models,
        "weights": weights
    })
    
    return ensemble_metrics

def format_metrics_for_display(metrics: Dict[str, Any], round_digits: int = 4) -> Dict[str, Any]:
    """
    Format metrics for display by rounding and converting to percentages.
    
    Args:
        metrics (Dict[str, Any]): Performance metrics
        round_digits (int): Number of digits to round to
        
    Returns:
        Dict[str, Any]: Formatted metrics
    """
    formatted = {}
    
    for key, value in metrics.items():
        if isinstance(value, (float, np.float32, np.float64)):
            # Convert to percentage for certain metrics
            percentage_metrics = [
                "accuracy", "precision", "recall", "f1", "auc", "specificity", "npv", 
                "balanced_accuracy", "precision_up", "precision_down", "recall_up", 
                "recall_down", "f1_up", "f1_down", "up_ratio", "confident_ratio"
            ]
            
            if any(key.startswith(metric) or key.endswith(metric) or key == metric for metric in percentage_metrics):
                formatted[key] = f"{value*100:.{round_digits}f}%"
            else:
                formatted[key] = f"{value:.{round_digits}f}"
        elif isinstance(value, dict):
            formatted[key] = format_metrics_for_display(value, round_digits)
        else:
            formatted[key] = value
    
    return formatted