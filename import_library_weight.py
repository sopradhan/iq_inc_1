"""
Utility module for model and weight persistence operations.
Provides functions to save/load ML models and weight configurations.
"""

# Standard library imports
import os
import json
import logging
from typing import Dict, Any

# Third-party imports
import joblib
import numpy as np
import torch
from safetensors.torch import load_file
from scipy.spatial.distance import cosine

# Data science libraries
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Configure module logger
logger = logging.getLogger(__name__)


def save_weights(weights: Dict[str, Any], filepath: str) -> None:
    """
    Persist weight dictionary to JSON file with validation.
    
    Args:
        weights: Dictionary containing weight parameters
        filepath: Destination file path for JSON output
        
    Raises:
        ValueError: If weights dict is empty or None
        IOError: If file write operation fails
    """
    if not weights:
        raise ValueError("Weights dictionary cannot be empty or None")
    
    try:
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Write weights with proper formatting
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(weights, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Successfully saved weights to: {filepath}")
        
    except Exception as e:
        logger.error(f"Failed to save weights to {filepath}: {str(e)}")
        raise IOError(f"Weight save operation failed: {str(e)}") from e


def load_weights(filepath: str) -> Dict[str, Any]:
    """
    Load weight dictionary from JSON file with validation.
    
    Args:
        filepath: Source file path for JSON input
        
    Returns:
        Dictionary containing loaded weight parameters
        
    Raises:
        FileNotFoundError: If weight file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    if not os.path.exists(filepath):
        logger.error(f"Weight file not found: {filepath}")
        raise FileNotFoundError(f"Weight file not found: {filepath}")
    
    try:
        # Load and validate JSON content
        with open(filepath, 'r', encoding='utf-8') as f:
            weights = json.load(f)
        
        if not isinstance(weights, dict):
            raise ValueError("Loaded weights must be a dictionary")
        
        logger.info(f"Successfully loaded weights from: {filepath}")
        return weights
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in weight file {filepath}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Failed to load weights from {filepath}: {str(e)}")
        raise


def save_model(model: Any, filepath: str) -> None:
    """
    Serialize and persist ML model to disk using joblib.
    
    Args:
        model: Trained scikit-learn or compatible model object
        filepath: Destination file path for model serialization
        
    Raises:
        ValueError: If model is None
        IOError: If model save operation fails
    """
    if model is None:
        raise ValueError("Model cannot be None")
    
    try:
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Serialize model with compression
        joblib.dump(model, filepath, compress=3)
        
        logger.info(f"Successfully saved model to: {filepath}")
        
    except Exception as e:
        logger.error(f"Failed to save model to {filepath}: {str(e)}")
        raise IOError(f"Model save operation failed: {str(e)}") from e


def load_model(filepath: str) -> Any:
    """
    Deserialize and load ML model from disk using joblib.
    
    Args:
        filepath: Source file path for model deserialization
        
    Returns:
        Loaded model object
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        Exception: If model deserialization fails
    """
    if not os.path.exists(filepath):
        logger.error(f"Model file not found: {filepath}")
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    try:
        # Deserialize model from disk
        model = joblib.load(filepath)
        
        logger.info(f"Successfully loaded model from: {filepath}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model from {filepath}: {str(e)}")
        raise