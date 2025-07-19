"""
Configuration settings for the Customer Churn Prediction System.
"""
import os
from pathlib import Path
from typing import Dict, List

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data configuration
DATA_CONFIG = {
    "raw_data_path": PROJECT_ROOT / "artifacts" / "data" / "raw",
    "processed_data_path": PROJECT_ROOT / "artifacts" / "data" / "processed",
    "features_path": PROJECT_ROOT / "artifacts" / "data" / "features",
}

# Model configuration
MODEL_CONFIG = {
    "model_path": PROJECT_ROOT / "artifacts" / "models",
    "best_model_name": "best_model.pkl",
    "model_metrics_path": PROJECT_ROOT / "artifacts" / "models" / "metrics.json",
    "feature_importance_path": PROJECT_ROOT / "artifacts" / "models" / "feature_importance.json",
}

# API configuration
API_CONFIG = {
    "title": "Customer Churn Prediction API",
    "description": "API for predicting customer churn using machine learning models",
    "version": "1.0.0",
    "host": "0.0.0.0",
    "port": 8000,
    "debug": False,
}

# Database configuration (for data import)
DATABASE_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("DB_NAME", "churn_db"),
    "user": os.getenv("DB_USER", "user"),
    "password": os.getenv("DB_PASSWORD", "password"),
    "port": int(os.getenv("DB_PORT", 5432)),
}

# Feature mappings
FEATURE_MAPPINGS = {
    "state_mapping": {
        'Andhra Pradesh': 1, 'Assam': 2, 'Bihar': 3, 'Chhattisgarh': 4,
        'Delhi': 5, 'Gujarat': 6, 'Haryana': 7, 'Jammu & Kashmir': 8,
        'Jharkhand': 9, 'Karnataka': 10, 'Kerala': 11, 'Madhya Pradesh': 12,
        'Maharashtra': 13, 'Odisha': 14, 'Puducherry': 15, 'Punjab': 16,
        'Rajasthan': 17, 'Tamil Nadu': 18, 'Telangana': 19, 'Uttar Pradesh': 20,
        'Uttarakhand': 21, 'West Bengal': 22
    },
    "gender_mapping": {'Male': 1, 'Female': 0},
    "binary_mapping": {'Yes': 1, 'No': 0},
    "internet_type_mapping": {'DSL': 0, 'Fiber Optic': 1, 'Cable': 2},
}

# Categorical options
CATEGORICAL_OPTIONS = {
    "value_deals": ["Deal 1", "Deal 2", "Deal 3", "Deal 4", "Deal 5", "No Deal"],
    "contract_options": ["Month-to-Month", "One Year", "Two Year"],
    "payment_methods": ["Bank Withdrawal", "Credit Card", "Mailed Check"],
    "internet_types": ["DSL", "Fiber Optic", "Cable"],
}

# Feature lists
FEATURE_LISTS = {
    "binary_features": [
        'phone_service', 'multiple_lines', 'internet_service',
        'online_security', 'online_backup', 'device_protection_plan',
        'premium_support', 'streaming_tv', 'streaming_movies',
        'streaming_music', 'unlimited_data', 'paperless_billing'
    ],
    "categorical_features": ['value_deal', 'contract', 'payment_method'],
    "numerical_features": [
        'age', 'number_of_referrals', 'tenure_in_months',
        'monthly_charge', 'total_charges', 'total_refunds',
        'total_extra_data_charges', 'total_long_distance_charges', 'total_revenue'
    ],
    "normalization_features": [
        'age', 'number_of_referrals', 'tenure_in_months',
        'monthly_charge', 'total_charges', 'total_refunds',
        'total_long_distance_charges', 'total_revenue'
    ]
}

# Normalization factors
NORMALIZATION_FACTORS = {
    'age': 100,
    'number_of_referrals': 15,
    'tenure_in_months': 70,
    'monthly_charge': 1000,
    'total_charges': 1000,
    'total_refunds': 1000,
    'total_extra_data_charges': 1000,
    'total_long_distance_charges': 1000,
    'total_revenue': 1000,
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["default"],
    },
}
