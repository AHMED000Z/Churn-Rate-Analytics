"""
Data preprocessing utilities for the Customer Churn Prediction System.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

from config.settings import (
    FEATURE_MAPPINGS,
    FEATURE_LISTS,
    CATEGORICAL_OPTIONS,
    NORMALIZATION_FACTORS
)

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Data preprocessing pipeline for churn prediction."""

    def __init__(self):
        self.scaler = None
        self.feature_columns = None

    def preprocess_single_input(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocess a single input for prediction.

        Args:
            data: Dictionary containing input features

        Returns:
            Preprocessed DataFrame ready for model prediction
        """
        try:
            # Convert input data to DataFrame
            input_df = pd.DataFrame([data])
            logger.info(f"Input data received: {list(data.keys())}")

            # Apply preprocessing steps
            processed_df = self._apply_preprocessing_pipeline(input_df)

            return processed_df

        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise

    def preprocess_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess training data.

        Args:
            df: Raw training DataFrame

        Returns:
            Preprocessed DataFrame ready for training
        """
        try:
            logger.info("Starting training data preprocessing...")

            # Apply preprocessing pipeline
            processed_df = self._apply_preprocessing_pipeline(df)

            # Store feature columns for later use
            self.feature_columns = processed_df.columns.tolist()

            logger.info(
                f"Training data preprocessing completed. Shape: {processed_df.shape}")
            return processed_df

        except Exception as e:
            logger.error(f"Error in training data preprocessing: {str(e)}")
            raise

    def _apply_preprocessing_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the complete preprocessing pipeline."""

        # Make a copy to avoid modifying original data
        processed_df = df.copy()

        # 1. Handle missing values
        processed_df = self._handle_missing_values(processed_df)

        # 2. Map categorical features
        processed_df = self._map_categorical_features(processed_df)

        # 3. Apply label encoding for binary features
        processed_df = self._apply_label_encoding(processed_df)

        # 4. Apply one-hot encoding for multi-category features
        processed_df = self._apply_one_hot_encoding(processed_df)

        # 5. Normalize numerical features
        processed_df = self._normalize_features(processed_df)

        # 6. Ensure feature consistency
        processed_df = self._ensure_feature_consistency(processed_df)

        return processed_df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""

        # Define default values for different feature types
        defaults = {
            'phone_service': 'No',
            'multiple_lines': 'No',
            'internet_service': 'No',
            'online_security': 'No',
            'online_backup': 'No',
            'device_protection_plan': 'No',
            'premium_support': 'No',
            'streaming_tv': 'No',
            'streaming_movies': 'No',
            'streaming_music': 'No',
            'unlimited_data': 'No',
            'paperless_billing': 'No',
            'value_deal': 'No Deal',
            'internet_type': 'DSL'
        }

        for column, default_value in defaults.items():
            if column in df.columns:
                df[column] = df[column].fillna(default_value)

        # Fill numerical columns with median or 0
        numerical_columns = FEATURE_LISTS['numerical_features']
        for col in numerical_columns:
            if col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        return df

    def _map_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map categorical features to numerical values."""

        # Map state
        if 'state' in df.columns:
            df['state'] = df['state'].map(FEATURE_MAPPINGS['state_mapping'])
            if df['state'].isnull().any():
                logger.warning(
                    "Unknown states found, filling with default value")
                df['state'] = df['state'].fillna(1)  # Default to first state

        # Map gender
        if 'gender' in df.columns:
            df['gender'] = df['gender'].map(FEATURE_MAPPINGS['gender_mapping'])

        # Map internet type
        if 'internet_type' in df.columns:
            df['internet_type'] = df['internet_type'].map(
                FEATURE_MAPPINGS['internet_type_mapping'])

        return df

    def _apply_label_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply label encoding to binary features."""

        binary_features = FEATURE_LISTS['binary_features']

        for feature in binary_features:
            if feature in df.columns:
                df[feature] = df[feature].map(
                    FEATURE_MAPPINGS['binary_mapping'])
                # Handle any unmapped values
                if df[feature].isnull().any():
                    logger.warning(
                        f"Unmapped values found in {feature}, filling with 0")
                    df[feature] = df[feature].fillna(0)

        return df

    def _apply_one_hot_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply one-hot encoding to categorical features."""

        categorical_features = FEATURE_LISTS['categorical_features']

        for feature in categorical_features:
            if feature in df.columns:
                # Create dummy variables
                dummies = pd.get_dummies(
                    df[feature], prefix=feature, drop_first=False)

                # Drop original column and add dummies
                df = df.drop(feature, axis=1)
                df = pd.concat([df, dummies], axis=1)

        return df

    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize numerical features."""

        normalize_features = FEATURE_LISTS['normalization_features']

        for feature in normalize_features:
            if feature in df.columns:
                factor = NORMALIZATION_FACTORS.get(feature, 1)
                df[feature] = df[feature] / factor

        return df

    def _ensure_feature_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure consistent features for model prediction."""

        # Add missing one-hot encoded columns
        self._add_missing_one_hot_columns(df)

        # Remove any unexpected columns
        if self.feature_columns:
            # Keep only expected features
            missing_features = set(self.feature_columns) - set(df.columns)
            extra_features = set(df.columns) - set(self.feature_columns)

            # Add missing features with default values
            for feature in missing_features:
                df[feature] = 0
                logger.info(f"Added missing feature: {feature}")

            # Remove extra features
            if extra_features:
                df = df.drop(columns=list(extra_features))
                logger.info(f"Removed extra features: {extra_features}")

            # Reorder columns to match training order
            df = df[self.feature_columns]

        return df

    def _add_missing_one_hot_columns(self, df: pd.DataFrame) -> None:
        """Add missing one-hot encoded columns."""

        # Value deals
        for deal in CATEGORICAL_OPTIONS['value_deals']:
            col_name = f'value_deal_{deal}'
            if col_name not in df.columns:
                df[col_name] = 0

        # Contracts
        for contract in CATEGORICAL_OPTIONS['contract_options']:
            col_name = f'contract_{contract}'
            if col_name not in df.columns:
                df[col_name] = 0

        # Payment methods
        for method in CATEGORICAL_OPTIONS['payment_methods']:
            col_name = f'payment_method_{method}'
            if col_name not in df.columns:
                df[col_name] = 0

    def validate_input_data(self, data: Dict[str, Any]) -> List[str]:
        """
        Validate input data and return list of validation errors.

        Args:
            data: Input data dictionary

        Returns:
            List of validation error messages
        """
        errors = []

        # Check required fields
        required_fields = [
            'gender', 'age', 'state', 'number_of_referrals', 'tenure_in_months',
            'phone_service', 'multiple_lines', 'internet_service', 'internet_type',
            'online_security', 'online_backup', 'device_protection_plan',
            'premium_support', 'streaming_tv', 'streaming_movies', 'streaming_music',
            'unlimited_data', 'paperless_billing', 'value_deal', 'contract',
            'payment_method', 'monthly_charge', 'total_charges', 'total_refunds',
            'total_extra_data_charges', 'total_long_distance_charges', 'total_revenue'
        ]

        for field in required_fields:
            if field not in data or data[field] is None:
                errors.append(f"Missing required field: {field}")

        # Validate ranges
        if 'age' in data:
            if not (18 <= data['age'] <= 100):
                errors.append("Age must be between 18 and 100")

        if 'monthly_charge' in data:
            if data['monthly_charge'] < 0:
                errors.append("Monthly charge cannot be negative")

        # Validate categorical values
        if 'state' in data:
            if data['state'] not in FEATURE_MAPPINGS['state_mapping']:
                errors.append(f"Invalid state: {data['state']}")

        return errors


def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Load and preprocess data from CSV file.

    Args:
        file_path: Path to the CSV file

    Returns:
        Preprocessed DataFrame
    """
    try:
        # Load data
        df = pd.read_csv(file_path)
        logger.info(f"Loaded data with shape: {df.shape}")

        # Initialize preprocessor
        preprocessor = DataPreprocessor()

        # Preprocess data
        processed_df = preprocessor.preprocess_training_data(df)

        return processed_df

    except Exception as e:
        logger.error(f"Error loading and preprocessing data: {str(e)}")
        raise
