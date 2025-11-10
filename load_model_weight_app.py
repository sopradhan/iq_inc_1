"""
Main application for training incident severity classification models.
Orchestrates data loading, feature extraction, model training, and persistence.

Usage:
    python load_model_weight_app.py
"""

import os
import sys
import sqlite3
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# Custom module imports
from embedding import BertEmbedder
from model_classifier import (
    vectorize_incidents,
    load_pattern_info,
    train_and_evaluate_models,
    optimize_weights,
    save_evaluation_results
)


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure application-wide logging with formatted output.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Setup file and console handlers with UTF-8 encoding
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_dir / "training.log", encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set UTF-8 encoding for console output on Windows
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
    
    # Get logger for this module
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("Incident Severity Classification - Training Pipeline")
    logger.info("=" * 80)


def validate_paths(paths: dict) -> None:
    """
    Verify all required file paths exist before processing.
    
    Args:
        paths: Dictionary of path names to file paths
        
    Raises:
        FileNotFoundError: If any required file is missing
    """
    logger = logging.getLogger(__name__)
    
    for name, path in paths.items():
        if not os.path.exists(path):
            error_msg = f"Required file not found - {name}: {path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        else:
            logger.info(f"[OK] Validated {name}: {path}")


def check_bert_model_files(model_dir: Path) -> bool:
    """
    Check if all required BERT model files exist locally.
    
    Args:
        model_dir: Path to BERT model directory
        
    Returns:
        True if all files exist, False otherwise
    """
    required_files = [
        "config.json",
        "vocab.txt",
        "pytorch_model.bin"  # or model.safetensors
    ]
    
    for file in required_files:
        file_path = model_dir / file
        if not file_path.exists():
            # Check for alternative model file
            if file == "pytorch_model.bin":
                alt_path = model_dir / "model.safetensors"
                if alt_path.exists():
                    continue
            return False
    
    return True


def initialize_bert_embedder(model_dir: Path) -> BertEmbedder:
    """
    Initialize BERT embedder with proper error handling.
    
    Args:
        model_dir: Path to BERT model directory
        
    Returns:
        Initialized BertEmbedder instance
        
    Raises:
        RuntimeError: If BERT initialization fails
    """
    logger = logging.getLogger(__name__)
    
    # Ensure model directory exists
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if local model files exist
    has_local_model = check_bert_model_files(model_dir)
    
    if not has_local_model:
        logger.warning(f"BERT model files not found in {model_dir}")
        logger.info("Downloading BERT model from HuggingFace Hub...")
        logger.info("This may take a few minutes on first run...")
        
        try:
            # Import here to download model
            from transformers import BertTokenizer, BertModel
            
            # Download and cache the model
            model_name = "bert-base-uncased"
            logger.info(f"Downloading {model_name}...")
            
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = BertModel.from_pretrained(model_name)
            
            # Save to local directory
            logger.info(f"Saving model to {model_dir}...")
            tokenizer.save_pretrained(str(model_dir))
            model.save_pretrained(str(model_dir))
            
            logger.info("[OK] BERT model downloaded and saved successfully")
            
        except Exception as e:
            error_msg = f"Failed to download BERT model: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    # Now initialize the embedder
    try:
        logger.info(f"Initializing BERT embedder from {model_dir}...")
        embedder = BertEmbedder(model_dir=str(model_dir))
        logger.info("[OK] BERT embedder initialized successfully")
        return embedder
        
    except Exception as e:
        error_msg = f"Failed to initialize BERT embedder: {str(e)}"
        logger.error(error_msg)
        logger.error("Full error details:", exc_info=True)
        raise RuntimeError(error_msg) from e


def load_training_data(db_path: str) -> list:
    """
    Load labeled training data from SQLite database.
    
    Args:
        db_path: Path to SQLite database
        
    Returns:
        List of tuples containing (json_payload, severity_level, azure_service, resource_type)
        
    Raises:
        sqlite3.Error: If database query fails
        ValueError: If no training data found
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading training data from database...")
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Query all training samples
            cursor.execute("""
                SELECT json_payload, severity_level, azure_service, resource_type 
                FROM synthetic_training_data
            """)
            
            training_data = cursor.fetchall()
        
        if not training_data:
            raise ValueError("No training data found in database")
        
        logger.info(f"Successfully loaded {len(training_data)} training samples")
        return training_data
        
    except sqlite3.Error as e:
        logger.error(f"Database error loading training data: {str(e)}")
        raise


def main() -> None:
    """
    Main execution pipeline for model training workflow.
    
    Pipeline Steps:
        1. Setup logging and validate paths
        2. Initialize BERT embedder (download if needed)
        3. Load training data
        4. Extract features and vectorize incidents
        5. Split data and standardize features
        6. Train multiple classifiers
        7. Save best model, scaler, and weights
        8. Store evaluation metrics
    """
    # Configure application logging
    setup_logging(log_level="INFO")
    logger = logging.getLogger(__name__)
    
    try:
        # ==================== Configuration ====================
        logger.info("Step 1: Configuring paths and parameters")
        
        # Model artifact directory
        MODEL_DIR = Path(r"D:\incident_management\models\bert")
        
        # Database path
        DB_PATH = Path(r"D:\incident_management\data\sqlite\incident_management_v2.db")
        
        # Output paths for trained artifacts
        INC_SEV_SCALER = MODEL_DIR / "incident_severity_scaler.joblib"
        INC_SEV_CLASSIFIER = MODEL_DIR / "incident_severity_classifier.joblib"
        SEVERITY_WEIGHTS_PATH = MODEL_DIR / "severity_weights.json"
        
        # Training parameters
        TEST_SIZE = 0.2
        RANDOM_STATE = 42
        MODEL_VERSION = "v1.0"
        
        # ==================== Path Validation ====================
        logger.info("Step 2: Validating file paths")
        
        # Only validate database path - BERT model downloads automatically
        required_paths = {
            "Database": str(DB_PATH)
        }
        
        validate_paths(required_paths)
        
        # Ensure output directory exists
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory ready: {MODEL_DIR}")
        
        # ==================== Initialize Embedder ====================
        logger.info("Step 3: Initializing BERT embedder")
        
        # Initialize BERT embedder with automatic download if needed
        embedder = initialize_bert_embedder(MODEL_DIR)
        
        # ==================== Data Preparation ====================
        logger.info("Step 4: Preparing training data")
        
        # Load severity rule patterns and encode them
        pattern_info = load_pattern_info(embedder, str(DB_PATH))
        
        # Load labeled training data from database
        training_data = load_training_data(str(DB_PATH))
        
        # ==================== Feature Engineering ====================
        logger.info("Step 5: Extracting features from incidents")
        
        # Transform incidents into feature vectors
        X, y = vectorize_incidents(training_data, embedder, pattern_info)
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        # ==================== Train-Test Split ====================
        logger.info("Step 6: Splitting dataset")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y  # Maintain class distribution
        )
        
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Test samples: {len(X_test)}")
        
        # ==================== Feature Standardization ====================
        logger.info("Step 7: Standardizing features")
        
        # Fit scaler on training data only
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        logger.info("[OK] Features standardized successfully")
        
        # ==================== Model Training ====================
        logger.info("Step 8: Training classifiers")
        
        # Train all models and select best performer
        best_model_name, best_model = train_and_evaluate_models(
            X_train_scaled, y_train,
            X_test_scaled, y_test
        )
        
        logger.info(f"[OK] Best model identified: {best_model_name}")
        
        # ==================== Model Persistence ====================
        logger.info("Step 9: Saving trained artifacts")
        
        # Save feature scaler
        joblib.dump(scaler, str(INC_SEV_SCALER))
        logger.info(f"[OK] Saved scaler to: {INC_SEV_SCALER}")
        
        # Save best classifier
        joblib.dump(best_model, str(INC_SEV_CLASSIFIER))
        logger.info(f"[OK] Saved classifier to: {INC_SEV_CLASSIFIER}")
        
        # ==================== Weight Optimization ====================
        logger.info("Step 10: Optimizing feature weights")
        
        # Learn optimal feature combination weights
        optimized_weights = optimize_weights(
            X, y,
            save_path=str(SEVERITY_WEIGHTS_PATH)
        )
        
        logger.info(f"[OK] Saved weights to: {SEVERITY_WEIGHTS_PATH}")
        
        # ==================== Results Persistence ====================
        logger.info("Step 11: Saving evaluation metrics")
        
        # Get predictions - handle both sklearn and custom models
        if hasattr(best_model, 'predict'):
            predictions = best_model.predict(X_test_scaled)
        else:
            logger.error("Best model doesn't have predict method")
            raise AttributeError("Model must have predict method")
        
        # Compute results
        results = {best_model_name: accuracy_score(y_test, predictions)}
        confusions = {best_model_name: confusion_matrix(y_test, predictions)}
        
        # Save to database
        save_evaluation_results(
            db_path=str(DB_PATH),
            results=results,
            confusions=confusions,
            version=MODEL_VERSION
        )
        
        logger.info("[OK] Evaluation results saved to database")
        
        # ==================== Summary ====================
        logger.info("=" * 80)
        logger.info("Training Pipeline Completed Successfully!")
        logger.info("=" * 80)
        logger.info(f"Best Model: {best_model_name}")
        logger.info(f"Test Accuracy: {results[best_model_name]:.4f}")
        logger.info(f"Model Version: {MODEL_VERSION}")
        logger.info(f"Artifacts saved to: {MODEL_DIR}")
        logger.info("=" * 80)
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        sys.exit(1)
        
    except ValueError as e:
        logger.error(f"Data validation error: {str(e)}")
        sys.exit(1)
        
    except sqlite3.Error as e:
        logger.error(f"Database error: {str(e)}")
        sys.exit(1)
        
    except RuntimeError as e:
        logger.error(f"Runtime error: {str(e)}")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    # Execute main training pipeline
    main()