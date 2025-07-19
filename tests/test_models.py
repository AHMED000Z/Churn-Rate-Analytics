"""
Unit tests for the Customer Churn Prediction System.
"""
import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import json

from src.models.predictor import ChurnPredictor
from src.utils.preprocessing import DataPreprocessor
from src.data.data_manager import DatabaseManager
from config.settings import FEATURE_MAPPINGS


class TestDataPreprocessor(unittest.TestCase):
    """Test cases for data preprocessing."""

    def setUp(self):
        self.preprocessor = DataPreprocessor()

        # Sample input data
        self.sample_input = {
            'gender': 'Male',
            'age': 35,
            'state': 'California',
            'number_of_referrals': 2,
            'tenure_in_months': 24,
            'phone_service': 'Yes',
            'multiple_lines': 'No',
            'internet_service': 'Yes',
            'internet_type': 'Fiber Optic',
            'online_security': 'Yes',
            'online_backup': 'No',
            'device_protection_plan': 'Yes',
            'premium_support': 'No',
            'streaming_tv': 'Yes',
            'streaming_movies': 'No',
            'streaming_music': 'Yes',
            'unlimited_data': 'Yes',
            'paperless_billing': 'Yes',
            'value_deal': 'Deal 1',
            'contract': 'One Year',
            'payment_method': 'Credit Card',
            'monthly_charge': 75.50,
            'total_charges': 1812.00,
            'total_refunds': 0.00,
            'total_extra_data_charges': 25.00,
            'total_long_distance_charges': 15.50,
            'total_revenue': 1852.50
        }

    def test_preprocess_single_input(self):
        """Test preprocessing of single input."""
        result = self.preprocessor.preprocess_single_input(self.sample_input)

        # Check that result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)

        # Check that categorical variables are properly encoded
        self.assertIn('gender', result.columns)
        self.assertEqual(result['gender'].iloc[0], 1)  # Male = 1

    def test_validate_input_data(self):
        """Test input data validation."""
        # Valid input should return no errors
        errors = self.preprocessor.validate_input_data(self.sample_input)
        self.assertEqual(len(errors), 0)

        # Missing required field should return error
        invalid_input = self.sample_input.copy()
        del invalid_input['age']
        errors = self.preprocessor.validate_input_data(invalid_input)
        self.assertGreater(len(errors), 0)
        self.assertTrue(any('age' in error for error in errors))

        # Invalid age should return error
        invalid_input = self.sample_input.copy()
        invalid_input['age'] = 150
        errors = self.preprocessor.validate_input_data(invalid_input)
        self.assertGreater(len(errors), 0)
        self.assertTrue(
            any('Age must be between' in error for error in errors))

    def test_handle_missing_values(self):
        """Test missing value handling."""
        data_with_missing = pd.DataFrame([{
            'gender': 'Male',
            'age': 35,
            'phone_service': None,
            'monthly_charge': None
        }])

        result = self.preprocessor._handle_missing_values(data_with_missing)

        # Check that missing values are filled
        self.assertEqual(result['phone_service'].iloc[0], 'No')
        self.assertEqual(result['monthly_charge'].iloc[0], 0)


class TestChurnPredictor(unittest.TestCase):
    """Test cases for churn prediction model."""

    def setUp(self):
        self.predictor = ChurnPredictor()

        # Mock model
        self.mock_model = Mock()
        self.mock_model.predict.return_value = np.array([1])
        self.mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])
        self.mock_model.feature_importances_ = np.array([0.1, 0.2, 0.3, 0.4])

        self.predictor.model = self.mock_model
        self.predictor.preprocessor.feature_columns = [
            'feature1', 'feature2', 'feature3', 'feature4']

        # Sample input
        self.sample_input = {
            'gender': 'Male',
            'age': 35,
            'state': 'Maharashtra',
            'number_of_referrals': 2,
            'tenure_in_months': 24,
            'phone_service': 'Yes',
            'multiple_lines': 'No',
            'internet_service': 'Yes',
            'internet_type': 'Fiber Optic',
            'online_security': 'Yes',
            'online_backup': 'No',
            'device_protection_plan': 'Yes',
            'premium_support': 'No',
            'streaming_tv': 'Yes',
            'streaming_movies': 'No',
            'streaming_music': 'Yes',
            'unlimited_data': 'Yes',
            'paperless_billing': 'Yes',
            'value_deal': 'Deal 1',
            'contract': 'One Year',
            'payment_method': 'Credit Card',
            'monthly_charge': 75.50,
            'total_charges': 1812.00,
            'total_refunds': 0.00,
            'total_extra_data_charges': 25.00,
            'total_long_distance_charges': 15.50,
            'total_revenue': 1852.50
        }

    @patch('src.models.predictor.joblib.load')
    def test_load_model_success(self, mock_joblib_load):
        """Test successful model loading."""
        mock_joblib_load.return_value = self.mock_model

        with patch('pathlib.Path.exists', return_value=True):
            result = self.predictor.load_model()

        self.assertTrue(result)
        self.assertTrue(self.predictor.is_trained)
        mock_joblib_load.assert_called_once()

    @patch('src.models.predictor.joblib.load')
    def test_load_model_file_not_found(self, mock_joblib_load):
        """Test model loading when file doesn't exist."""
        with patch('pathlib.Path.exists', return_value=False):
            result = self.predictor.load_model()

        self.assertFalse(result)
        self.assertFalse(self.predictor.is_trained)
        mock_joblib_load.assert_not_called()

    def test_predict_success(self):
        """Test successful prediction."""
        with patch.object(self.predictor.preprocessor, 'validate_input_data', return_value=[]):
            with patch.object(self.predictor.preprocessor, 'preprocess_single_input') as mock_preprocess:
                mock_preprocess.return_value = pd.DataFrame(
                    [[1, 2, 3, 4]], columns=['feature1', 'feature2', 'feature3', 'feature4'])

                result = self.predictor.predict(self.sample_input)

                self.assertEqual(result['prediction'], 1)
                self.assertEqual(result['prediction_label'], 'Churn')
                self.assertEqual(result['probability'], 0.8)
                self.assertEqual(result['confidence'], 'High')
                self.assertIsInstance(result['feature_importance'], list)

    def test_predict_no_model_loaded(self):
        """Test prediction when no model is loaded."""
        self.predictor.model = None

        with self.assertRaises(ValueError) as context:
            self.predictor.predict(self.sample_input)

        self.assertIn('Model not loaded', str(context.exception))

    def test_predict_validation_error(self):
        """Test prediction with invalid input."""
        with patch.object(self.predictor.preprocessor, 'validate_input_data', return_value=['Age is required']):
            with self.assertRaises(ValueError) as context:
                self.predictor.predict(self.sample_input)

            self.assertIn('Input validation errors', str(context.exception))

    def test_get_confidence_level(self):
        """Test confidence level calculation."""
        # High confidence (probability close to 0 or 1)
        self.assertEqual(self.predictor._get_confidence_level(0.1), 'High')
        self.assertEqual(self.predictor._get_confidence_level(0.9), 'High')

        # Medium confidence
        self.assertEqual(self.predictor._get_confidence_level(0.35), 'Medium')
        self.assertEqual(self.predictor._get_confidence_level(0.65), 'Medium')

        # Low confidence (probability around 0.5)
        self.assertEqual(self.predictor._get_confidence_level(0.5), 'Low')


class TestDatabaseManager(unittest.TestCase):
    """Test cases for database operations."""

    def setUp(self):
        self.db_manager = DatabaseManager({
            'host': 'localhost',
            'database': 'test_db',
            'user': 'test_user',
            'password': 'test_pass',
            'port': 5432
        })

    @patch('src.data.data_manager.psycopg2.connect')
    def test_connect_success(self, mock_connect):
        """Test successful database connection."""
        mock_connection = Mock()
        mock_connect.return_value = mock_connection

        result = self.db_manager.connect()

        self.assertTrue(result)
        self.assertEqual(self.db_manager.connection, mock_connection)
        mock_connect.assert_called_once_with(
            **self.db_manager.connection_params)

    @patch('src.data.data_manager.psycopg2.connect')
    def test_connect_failure(self, mock_connect):
        """Test database connection failure."""
        mock_connect.side_effect = Exception('Connection failed')

        result = self.db_manager.connect()

        self.assertFalse(result)
        self.assertIsNone(self.db_manager.connection)

    @patch('src.data.data_manager.pd.read_sql_query')
    def test_execute_query_success(self, mock_read_sql):
        """Test successful query execution."""
        mock_df = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
        mock_read_sql.return_value = mock_df

        self.db_manager.connection = Mock()

        result = self.db_manager.execute_query('SELECT * FROM test_table')

        pd.testing.assert_frame_equal(result, mock_df)
        mock_read_sql.assert_called_once_with(
            'SELECT * FROM test_table', self.db_manager.connection)

    def test_execute_query_no_connection(self):
        """Test query execution without connection."""
        with self.assertRaises(ConnectionError):
            self.db_manager.execute_query('SELECT * FROM test_table')


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""

    def setUp(self):
        self.predictor = ChurnPredictor()

        # Sample training data
        self.training_data = pd.DataFrame({
            'gender': ['Male', 'Female', 'Male', 'Female'],
            'age': [25, 35, 45, 55],
            'state': ['Maharashtra', 'Delhi', 'Karnataka', 'Tamil Nadu'],
            'number_of_referrals': [0, 1, 2, 3],
            'tenure_in_months': [12, 24, 36, 48],
            'phone_service': ['Yes', 'Yes', 'No', 'Yes'],
            'multiple_lines': ['No', 'Yes', 'No', 'No'],
            'internet_service': ['Yes', 'Yes', 'No', 'Yes'],
            'internet_type': ['DSL', 'Fiber Optic', 'DSL', 'Cable'],
            'online_security': ['No', 'Yes', 'No', 'Yes'],
            'online_backup': ['No', 'No', 'Yes', 'Yes'],
            'device_protection_plan': ['No', 'Yes', 'No', 'Yes'],
            'premium_support': ['No', 'No', 'Yes', 'Yes'],
            'streaming_tv': ['No', 'Yes', 'No', 'Yes'],
            'streaming_movies': ['No', 'No', 'Yes', 'Yes'],
            'streaming_music': ['Yes', 'Yes', 'No', 'No'],
            'unlimited_data': ['No', 'Yes', 'No', 'Yes'],
            'paperless_billing': ['Yes', 'No', 'Yes', 'No'],
            'value_deal': ['Deal 1', 'Deal 2', 'No Deal', 'Deal 3'],
            'contract': ['Month-to-Month', 'One Year', 'Two Year', 'One Year'],
            'payment_method': ['Credit Card', 'Bank Withdrawal', 'Mailed Check', 'Credit Card'],
            'monthly_charge': [50.0, 75.0, 100.0, 80.0],
            'total_charges': [600.0, 1800.0, 3600.0, 3840.0],
            'total_refunds': [0.0, 50.0, 0.0, 100.0],
            'total_extra_data_charges': [10.0, 0.0, 20.0, 0.0],
            'total_long_distance_charges': [5.0, 15.0, 0.0, 25.0],
            'total_revenue': [615.0, 1765.0, 3620.0, 3765.0],
            'customer_status': [0, 1, 0, 1]  # Target variable
        })

    def test_end_to_end_prediction_pipeline(self):
        """Test the complete prediction pipeline."""
        # Mock a trained model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        mock_model.feature_importances_ = np.random.random(10)

        self.predictor.model = mock_model
        self.predictor.preprocessor.feature_columns = [
            f'feature_{i}' for i in range(10)]

        # Test input
        test_input = {
            'gender': 'Male',
            'age': 35,
            'state': 'Maharashtra',
            'number_of_referrals': 2,
            'tenure_in_months': 24,
            'phone_service': 'Yes',
            'multiple_lines': 'No',
            'internet_service': 'Yes',
            'internet_type': 'Fiber Optic',
            'online_security': 'Yes',
            'online_backup': 'No',
            'device_protection_plan': 'Yes',
            'premium_support': 'No',
            'streaming_tv': 'Yes',
            'streaming_movies': 'No',
            'streaming_music': 'Yes',
            'unlimited_data': 'Yes',
            'paperless_billing': 'Yes',
            'value_deal': 'Deal 1',
            'contract': 'One Year',
            'payment_method': 'Credit Card',
            'monthly_charge': 75.50,
            'total_charges': 1812.00,
            'total_refunds': 0.00,
            'total_extra_data_charges': 25.00,
            'total_long_distance_charges': 15.50,
            'total_revenue': 1852.50
        }

        # Make prediction
        with patch.object(self.predictor.preprocessor, 'validate_input_data', return_value=[]):
            with patch.object(self.predictor.preprocessor, 'preprocess_single_input') as mock_preprocess:
                mock_preprocess.return_value = pd.DataFrame(
                    [np.random.random(10)],
                    columns=[f'feature_{i}' for i in range(10)]
                )

                result = self.predictor.predict(test_input)

                # Verify result structure
                self.assertIn('prediction', result)
                self.assertIn('prediction_label', result)
                self.assertIn('probability', result)
                self.assertIn('confidence', result)
                self.assertIn('feature_importance', result)

                # Verify result values
                self.assertEqual(result['prediction'], 1)
                self.assertEqual(result['prediction_label'], 'Churn')
                self.assertEqual(result['probability'], 0.7)
                self.assertIsInstance(result['feature_importance'], list)


if __name__ == '__main__':
    unittest.main()
