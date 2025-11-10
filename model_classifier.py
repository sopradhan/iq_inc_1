"""
Machine learning classifiers for incident severity prediction.
Supports multiple algorithms including Logistic Regression, Random Forest, 
XGBoost, and Neural Networks with hyperparameter tuning.
"""

import json
import logging
import sqlite3
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.spatial.distance import cosine
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

from import_library_weight import save_weights

logger = logging.getLogger(__name__)


class NeuralNet(nn.Module):
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )
        logger.debug(f"Initialized NeuralNet: input={input_dim}, hidden={hidden_dim}, output=4")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ModelBase:
    
    def __init__(self, name: str):
        self.name = name
        self.best_model = None
        self.best_params = None
        self.best_score = None
        logger.info(f"Initialized {self.name} classifier")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        raise NotImplementedError("Subclasses must implement train() method")
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self.best_model is None:
            raise ValueError("Model must be trained before making predictions")
        return self.best_model.predict(X_test)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, np.ndarray]:
        predictions = self.predict(X_test)
        logger.info(f"\n{'='*50}\n{self.name} Evaluation\n{'='*50}")
        print(f"\n{'='*50}\n{self.name} Evaluation\n{'='*50}")
        print(classification_report(y_test, predictions, zero_division=0))
        accuracy = accuracy_score(y_test, predictions)
        logger.info(f"Accuracy: {accuracy:.4f}")
        print(f"Accuracy: {accuracy:.4f}\n")
        conf_matrix = confusion_matrix(y_test, predictions)
        return accuracy, conf_matrix


class LogisticRegressionModel(ModelBase):
    
    def __init__(self):
        super().__init__("Logistic Regression")
        self.param_grid = {
            "max_iter": [200, 500],
            "C": [0.1, 1, 10],
            "multi_class": ["multinomial"],
            "solver": ["lbfgs"]
        }
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        logger.info(f"Training {self.name} with GridSearchCV...")
        grid_search = GridSearchCV(
            estimator=LogisticRegression(random_state=42),
            param_grid=self.param_grid,
            cv=3,
            scoring="accuracy",
            n_jobs=1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        logger.info(f"Best params: {self.best_params}, CV score: {self.best_score:.4f}")


class RandomForestModel(ModelBase):
    
    def __init__(self):
        super().__init__("Random Forest")
        self.param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        }
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        logger.info(f"Training {self.name} with GridSearchCV...")
        grid_search = GridSearchCV(
            estimator=RandomForestClassifier(random_state=42),
            param_grid=self.param_grid,
            cv=3,
            scoring="accuracy",
            n_jobs=1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        logger.info(f"Best params: {self.best_params}, CV score: {self.best_score:.4f}")


class XGBoostModel(ModelBase):
    
    def __init__(self):
        super().__init__("XGBoost")
        self.param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 6, 9],
            "learning_rate": [0.01, 0.1, 0.3],
            "subsample": [0.8, 1.0]
        }
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        logger.info(f"Training {self.name} with GridSearchCV...")
        grid_search = GridSearchCV(
            estimator=xgb.XGBClassifier(
                use_label_encoder=False,
                eval_metric="mlogloss",
                random_state=42
            ),
            param_grid=self.param_grid,
            cv=3,
            scoring="accuracy",
            n_jobs=1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        logger.info(f"Best params: {self.best_params}, CV score: {self.best_score:.4f}")


class NeuralNetworkModel(ModelBase):
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, 
                 lr: float = 0.001, epochs: int = 20, batch_size: int = 32):
        super().__init__("Neural Network")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.model = NeuralNet(input_dim, hidden_dim).to(self.device)
        self.best_model = self.model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        logger.info(f"Training {self.name} for {self.epochs} epochs...")
        dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")
        logger.info("Neural network training completed")
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
            outputs = self.model(test_tensor)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        return predictions


def extract_analysis_text(incident_data: Dict[str, Any]) -> str:
    text_parts = []
    
    if not isinstance(incident_data, dict):
        logger.warning("Incident data is not a dictionary")
        return ""
    
    properties = incident_data.get("properties", {})
    if isinstance(properties, dict):
        error = properties.get("error", {})
        if isinstance(error, dict):
            error_msg = error.get("message", "")
            error_code = error.get("code", "")
            if error_msg:
                text_parts.append(f"error: {error_msg}")
            if error_code:
                text_parts.append(f"code: {error_code}")
    
    operation = incident_data.get("operationName")
    if operation:
        if isinstance(operation, dict):
            op_value = operation.get("localizedValue") or operation.get("value", "")
            text_parts.append(f"operation: {op_value}")
        else:
            text_parts.append(f"operation: {operation}")
    
    status = incident_data.get("status")
    if status:
        if isinstance(status, dict):
            status_value = status.get("localizedValue") or status.get("value", "")
            text_parts.append(f"status: {status_value}")
        else:
            text_parts.append(f"status: {status}")
    
    category = incident_data.get("category")
    if category:
        text_parts.append(f"category: {category}")
    
    tags = incident_data.get("tags", {})
    if isinstance(tags, dict):
        service = tags.get("Service", "")
        if service:
            text_parts.append(f"service: {service}")
    
    metric_name = incident_data.get("metricName", "")
    if metric_name:
        text_parts.append(f"metric: {metric_name}")
    
    combined_text = " ".join(filter(None, text_parts))
    
    if not combined_text:
        combined_text = str(incident_data.get("id", ""))[:100]
    
    logger.debug(f"Extracted text: {combined_text[:150]}...")
    return combined_text


def get_environment_factor(incident_data: Dict[str, Any]) -> float:
    environment = None
    
    if not isinstance(incident_data, dict):
        return 1.0
    
    properties = incident_data.get("properties", {})
    if isinstance(properties, dict):
        env_value = properties.get("environment") or properties.get("env")
        if env_value:
            environment = str(env_value).lower()
    
    if not environment:
        tags = incident_data.get("tags", {})
        if isinstance(tags, dict):
            env_tag = tags.get("Environment", "")
            if env_tag:
                environment = str(env_tag).lower()
    
    if not environment:
        resource_id = str(incident_data.get("resourceId", "")).lower()
        if "prod-" in resource_id or "/prod/" in resource_id:
            environment = "prod"
        elif "uat-" in resource_id or "/uat/" in resource_id:
            environment = "uat"
        elif "dev-" in resource_id or "/dev/" in resource_id:
            environment = "dev"
    
    env_factor_map = {
        "prod": 1.2,
        "production": 1.2,
        "uat": 1.0,
        "staging": 1.0,
        "test": 1.0,
        "dev": 0.8,
        "development": 0.8
    }
    
    factor = env_factor_map.get(environment, 1.0)
    logger.debug(f"Environment '{environment}' mapped to factor {factor}")
    
    return factor


def load_pattern_info(embedder, db_path: str) -> Dict[str, Dict[str, Any]]:
    logger.info("Loading severity rule patterns from database...")
    pattern_info = {}
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT pattern, base_score FROM severity_rules")
            rules = cursor.fetchall()
        
        for pattern, base_score in rules:
            pattern_lower = pattern.lower()
            embedding = embedder.encode(pattern_lower)
            pattern_info[pattern] = {
                "embedding": embedding,
                "base_score": base_score
            }
        
        logger.info(f"Loaded {len(pattern_info)} severity rule patterns")
        
    except sqlite3.Error as e:
        logger.error(f"Database error loading patterns: {str(e)}")
        raise
    
    return pattern_info


def vectorize_incidents(data: List[Tuple], embedder, 
                       pattern_info: Dict[str, Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    logger.info(f"Vectorizing {len(data)} incidents with enhanced features...")
    
    feature_matrix = []
    label_vector = []
    
    severity_map = {
        "S1": 0,
        "S2": 1,
        "S3": 2,
        "S4": 3
    }
    
    for idx, (incident_json, severity_label, _, _) in enumerate(data):
        try:
            incident_data = json.loads(incident_json)
            incident_text = extract_analysis_text(incident_data)
            
            if not incident_text:
                logger.warning(f"Empty text for incident {idx}, skipping")
                continue
            
            incident_embedding = embedder.encode(incident_text.lower())
            
            max_similarity = 0.0
            best_rule_score = 0.0
            
            for pattern, info in pattern_info.items():
                similarity = 1.0 - cosine(incident_embedding, info["embedding"])
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_rule_score = info["base_score"]
            
            env_factor = get_environment_factor(incident_data)
            weighted_rule_score = best_rule_score * env_factor
            
            properties = incident_data.get("properties", {})
            error_rate = properties.get("error_rate", 0.0)
            latency = properties.get("latency_ms", 0.0)
            availability = properties.get("availability_percent", 100.0)
            
            error_rate_norm = min(error_rate / 100.0, 1.0)
            latency_norm = min(latency / 10000.0, 1.0)
            availability_norm = availability / 100.0
            
            features = [
                max_similarity,
                weighted_rule_score,
                env_factor,
                error_rate_norm,
                latency_norm,
                1.0 - availability_norm
            ]
            feature_matrix.append(features)
            
            label = severity_map.get(severity_label, 3)
            label_vector.append(label)
            
        except Exception as e:
            logger.warning(f"Error processing incident {idx}: {str(e)}")
            continue
    
    X = np.array(feature_matrix)
    y = np.array(label_vector)
    
    logger.info(f"Generated feature matrix of shape {X.shape}")
    logger.info(f"Features: BERT_sim, Rule_score, Env_factor, Error_rate, Latency, Unavailability")
    
    return X, y


def train_and_evaluate_models(X_train: np.ndarray, y_train: np.ndarray,
                               X_test: np.ndarray, y_test: np.ndarray) -> Tuple[str, Any]:
    logger.info("Training and evaluating all classifiers...")
    
    classifiers = [
        LogisticRegressionModel(),
        RandomForestModel(),
        XGBoostModel(),
        NeuralNetworkModel(input_dim=X_train.shape[1], hidden_dim=128)
    ]
    
    results = {}
    
    for classifier in classifiers:
        try:
            classifier.train(X_train, y_train)
            accuracy, conf_matrix = classifier.evaluate(X_test, y_test)
            results[classifier.name] = {
                "accuracy": accuracy,
                "model": classifier,
                "confusion_matrix": conf_matrix
            }
            
            if hasattr(classifier, 'best_params') and classifier.best_params:
                logger.info(f"{classifier.name} - Best params: {classifier.best_params}")
                logger.info(f"{classifier.name} - CV score: {classifier.best_score:.4f}")
            else:
                logger.info(f"{classifier.name} - No hyperparameter tuning performed")
            
        except Exception as e:
            logger.error(f"Error training {classifier.name}: {str(e)}")
            continue
    
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best_model = results[best_name]["model"].best_model
    
    logger.info(f"\nBest model: {best_name} with accuracy {results[best_name]['accuracy']:.4f}")
    
    return best_name, best_model


def optimize_weights(X: np.ndarray, y: np.ndarray, save_path: str) -> Dict[str, float]:
    logger.info("Optimizing feature combination weights...")
    
    severity_score_map = {
        0: 100,
        1: 75,
        2: 50,
        3: 25
    }
    
    y_scores = np.array([severity_score_map[label] for label in y])
    
    regression = LinearRegression()
    regression.fit(X, y_scores)
    
    weights = {
        "bert_weight": float(regression.coef_[0]),
        "rule_weight": float(regression.coef_[1]),
        "env_weight": float(regression.coef_[2]),
        "intercept": float(regression.intercept_)
    }
    
    logger.info(f"Optimized weights: {weights}")
    
    try:
        save_weights(weights, save_path)
    except Exception as e:
        logger.error(f"Failed to save weights: {str(e)}")
        raise
    
    return weights


def save_evaluation_results(db_path: str, results: Dict[str, float],
                           confusions: Dict[str, np.ndarray], version: str) -> None:
    logger.info(f"Saving evaluation results for version {version}...")
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_results (
                    model_name TEXT,
                    version TEXT,
                    accuracy REAL,
                    trained_at TEXT,
                    classification_report TEXT,
                    PRIMARY KEY (model_name, version)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS confusion_matrices (
                    model_name TEXT,
                    version TEXT,
                    true_label INTEGER,
                    predicted_label INTEGER,
                    count INTEGER,
                    PRIMARY KEY (model_name, version, true_label, predicted_label)
                )
            """)
            
            conn.commit()
            
            trained_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            for model_name, accuracy in results.items():
                cursor.execute("""
                    INSERT OR REPLACE INTO model_results
                    (model_name, version, accuracy, trained_at, classification_report)
                    VALUES (?, ?, ?, ?, ?)
                """, (model_name, version, accuracy, trained_at, ""))
            
            for model_name, conf_matrix in confusions.items():
                for true_label in range(conf_matrix.shape[0]):
                    for pred_label in range(conf_matrix.shape[1]):
                        count = int(conf_matrix[true_label, pred_label])
                        cursor.execute("""
                            INSERT OR REPLACE INTO confusion_matrices
                            (model_name, version, true_label, predicted_label, count)
                            VALUES (?, ?, ?, ?, ?)
                        """, (model_name, version, true_label, pred_label, count))
            
            conn.commit()
            logger.info("Successfully saved evaluation results to database")
            
    except sqlite3.Error as e:
        logger.error(f"Database error saving results: {str(e)}")
        raise