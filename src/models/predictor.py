"""
Model management utilities for the Customer Churn Prediction System.
"""
import joblib
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix
)
import warnings

from config.settings import MODEL_CONFIG
from src.utils.preprocessing import DataPreprocessor

logger = logging.getLogger(__name__)


class ChurnPredictor:
    """Main class for churn prediction model management."""

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.preprocessor = DataPreprocessor()
        self.model_path = model_path or str(
            MODEL_CONFIG['model_path'] / MODEL_CONFIG['best_model_name'])
        self.is_trained = False

    def load_model(self) -> bool:
        """
        Load the trained model from disk.

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if Path(self.model_path).exists():
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    self.model = joblib.load(self.model_path)

                logger.info(
                    f"Model loaded successfully from {self.model_path}")
                self.is_trained = True
                return True
            else:
                logger.error(f"Model file not found: {self.model_path}")
                return False

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    def save_model(self, model_path: Optional[str] = None) -> bool:
        """
        Save the trained model to disk.

        Args:
            model_path: Optional custom path to save the model

        Returns:
            True if model saved successfully, False otherwise
        """
        try:
            if self.model is None:
                logger.error("No model to save")
                return False

            save_path = model_path or self.model_path

            # Ensure directory exists
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

            joblib.dump(self.model, save_path)
            logger.info(f"Model saved successfully to {save_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction for a single input.

        Args:
            input_data: Dictionary containing input features

        Returns:
            Dictionary containing prediction results
        """
        try:
            if self.model is None:
                raise ValueError("Model not loaded. Call load_model() first.")

            # Validate input data
            validation_errors = self.preprocessor.validate_input_data(
                input_data)
            if validation_errors:
                raise ValueError(
                    f"Input validation errors: {validation_errors}")

            # Preprocess input
            processed_input = self.preprocessor.preprocess_single_input(
                input_data)

            # Make prediction
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                prediction = self.model.predict(processed_input)[0]
                prediction_proba = self.model.predict_proba(processed_input)[0]

            # Get feature importance if available
            feature_importance = self._get_feature_importance(processed_input)

            # Determine confidence level
            probability = prediction_proba[1]  # Probability of churning
            confidence = self._get_confidence_level(probability)

            result = {
                "prediction": int(prediction),
                "prediction_label": "Churn" if prediction == 1 else "No Churn",
                "probability": float(probability),
                "confidence": confidence,
                "feature_importance": feature_importance
            }

            logger.info(
                f"Prediction made: {result['prediction_label']} (probability: {probability:.3f})")
            return result

        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise

    def predict_batch(self, input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Make predictions for multiple inputs.

        Args:
            input_data: List of dictionaries containing input features

        Returns:
            List of prediction results
        """
        results = []
        for i, data in enumerate(input_data):
            try:
                result = self.predict(data)
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting for input {i}: {str(e)}")
                results.append({
                    "error": str(e),
                    "prediction": None,
                    "probability": None
                })

        return results

    def train_model(self,
                    training_data: pd.DataFrame,
                    target_column: str = 'customer_status',
                    test_size: float = 0.2,
                    random_state: int = 42) -> Dict[str, float]:
        """
        Train the churn prediction model.

        Args:
            training_data: DataFrame containing training data
            target_column: Name of the target column
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility

        Returns:
            Dictionary containing training metrics
        """
        try:
            logger.info("Starting model training...")

            # Prepare data
            X = training_data.drop(columns=[target_column])
            y = training_data[target_column]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )

            logger.info(
                f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

            # Train model with hyperparameter tuning
            self.model = self._train_with_hyperparameter_tuning(
                X_train, y_train)

            # Evaluate model
            metrics = self._evaluate_model(X_test, y_test)

            # Save feature columns for preprocessing
            self.preprocessor.feature_columns = X.columns.tolist()

            self.is_trained = True
            logger.info("Model training completed successfully")

            return metrics

        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

    def _train_with_hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train Random Forest with hyperparameter tuning."""

        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }

        # Initialize Random Forest
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)

        # Perform grid search
        logger.info("Performing hyperparameter tuning...")
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
        )

        grid_search.fit(X_train, y_train)

        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(
            f"Best cross-validation score: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_

    def _evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate the trained model on test data."""

        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'auc_score': roc_auc_score(y_test, y_pred_proba)
        }

        # Log results
        logger.info("Model evaluation results:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")

        # Save metrics
        self._save_metrics(metrics)

        return metrics

    def _get_feature_importance(self, processed_input: pd.DataFrame) -> Optional[List[Dict[str, Any]]]:
        """Get feature importance for the prediction."""

        if not hasattr(self.model, 'feature_importances_'):
            return None

        importance = self.model.feature_importances_
        feature_names = processed_input.columns

        # Create list of feature importance dictionaries
        feature_importance = [
            {"feature": str(feature), "importance": float(imp)}
            for feature, imp in zip(feature_names, importance)
            if imp > 0
        ]

        # Sort by importance and return top 10
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        return feature_importance[:10]

    def _get_confidence_level(self, probability: float) -> str:
        """Determine confidence level based on probability."""

        if probability <= 0.3 or probability >= 0.7:
            return "High"
        elif probability <= 0.4 or probability >= 0.6:
            return "Medium"
        else:
            return "Low"

    def _save_metrics(self, metrics: Dict[str, float]) -> None:
        """Save model metrics to file."""

        try:
            metrics_path = MODEL_CONFIG['model_metrics_path']
            Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)

            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)

            logger.info(f"Metrics saved to {metrics_path}")

        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""

        if self.model is None:
            return {"status": "No model loaded"}

        info = {
            "model_type": type(self.model).__name__,
            "is_trained": self.is_trained,
            "model_loaded": True
        }

        # Add Random Forest specific info
        if hasattr(self.model, 'n_estimators'):
            info["n_estimators"] = self.model.n_estimators

        if hasattr(self.model, 'feature_names_in_'):
            info["n_features"] = len(self.model.feature_names_in_)
            info["feature_names"] = self.model.feature_names_in_.tolist()

        return info


def load_model_for_api() -> ChurnPredictor:
    """
    Load model for API usage.

    Returns:
        Initialized ChurnPredictor instance
    """
    predictor = ChurnPredictor()

    if predictor.load_model():
        logger.info("Model loaded successfully for API")
    else:
        logger.warning("Failed to load model for API")

    return predictor
