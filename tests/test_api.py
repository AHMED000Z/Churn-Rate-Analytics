"""
API tests for the Customer Churn Prediction System.
"""
import unittest
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from src.api.main import app
from src.models.predictor import ChurnPredictor


class TestChurnPredictionAPI(unittest.TestCase):
    """Test cases for the FastAPI application."""

    def setUp(self):
        self.client = TestClient(app)

        # Sample valid input data
        self.valid_input = {
            "gender": "Male",
            "age": 35,
            "state": "Maharashtra",
            "number_of_referrals": 2,
            "tenure_in_months": 24,
            "phone_service": "Yes",
            "multiple_lines": "No",
            "internet_service": "Yes",
            "internet_type": "Fiber Optic",
            "online_security": "Yes",
            "online_backup": "No",
            "device_protection_plan": "Yes",
            "premium_support": "No",
            "streaming_tv": "Yes",
            "streaming_movies": "No",
            "streaming_music": "Yes",
            "unlimited_data": "Yes",
            "paperless_billing": "Yes",
            "value_deal": "Deal 1",
            "contract": "One Year",
            "payment_method": "Credit Card",
            "monthly_charge": 75.50,
            "total_charges": 1812.00,
            "total_refunds": 0.00,
            "total_extra_data_charges": 25.00,
            "total_long_distance_charges": 15.50,
            "total_revenue": 1852.50
        }

    def test_root_endpoint(self):
        """Test the root endpoint."""
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("message", data)
        self.assertIn("version", data)

    def test_health_endpoint(self):
        """Test the health check endpoint."""
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("model_loaded", data)

    def test_metadata_endpoint(self):
        """Test the metadata endpoint."""
        response = self.client.get("/metadata")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("states", data)
        self.assertIn("value_deals", data)
        self.assertIn("contracts", data)
        self.assertIn("payment_methods", data)
        self.assertIn("internet_types", data)

        # Check that the lists are not empty
        self.assertGreater(len(data["states"]), 0)
        self.assertGreater(len(data["value_deals"]), 0)

    @patch('src.api.main.predictor')
    def test_predict_endpoint_success(self, mock_predictor):
        """Test successful prediction."""
        # Mock predictor response
        mock_prediction_result = {
            "prediction": 1,
            "prediction_label": "Churn",
            "probability": 0.75,
            "confidence": "High",
            "feature_importance": [
                {"feature": "tenure_in_months", "importance": 0.3},
                {"feature": "monthly_charge", "importance": 0.2}
            ]
        }

        mock_predictor.model = Mock()
        mock_predictor.predict.return_value = mock_prediction_result

        response = self.client.post("/predict", json=self.valid_input)

        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["prediction"], 1)
        self.assertEqual(data["prediction_label"], "Churn")
        self.assertEqual(data["probability"], 0.75)
        self.assertEqual(data["confidence"], "High")
        self.assertIsInstance(data["feature_importance"], list)

    @patch('src.api.main.predictor')
    def test_predict_endpoint_no_model(self, mock_predictor):
        """Test prediction when no model is loaded."""
        mock_predictor.model = None

        response = self.client.post("/predict", json=self.valid_input)

        self.assertEqual(response.status_code, 503)
        data = response.json()
        self.assertIn("Model not loaded", data["detail"])

    def test_predict_endpoint_invalid_input(self):
        """Test prediction with invalid input data."""
        invalid_input = self.valid_input.copy()
        invalid_input["age"] = 150  # Invalid age

        response = self.client.post("/predict", json=invalid_input)

        self.assertEqual(response.status_code, 422)  # Validation error

    def test_predict_endpoint_missing_fields(self):
        """Test prediction with missing required fields."""
        invalid_input = self.valid_input.copy()
        del invalid_input["age"]  # Remove required field

        response = self.client.post("/predict", json=invalid_input)

        self.assertEqual(response.status_code, 422)  # Validation error

    def test_predict_endpoint_invalid_categorical_values(self):
        """Test prediction with invalid categorical values."""
        invalid_input = self.valid_input.copy()
        invalid_input["gender"] = "Other"  # Invalid gender

        response = self.client.post("/predict", json=invalid_input)

        self.assertEqual(response.status_code, 422)  # Validation error

    @patch('src.api.main.predictor')
    def test_predict_endpoint_internal_error(self, mock_predictor):
        """Test prediction when an internal error occurs."""
        mock_predictor.model = Mock()
        mock_predictor.predict.side_effect = Exception("Internal error")

        response = self.client.post("/predict", json=self.valid_input)

        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertIn("Internal server error", data["detail"])

    @patch('src.api.main.predictor')
    def test_model_info_endpoint(self, mock_predictor):
        """Test the model info endpoint."""
        mock_model_info = {
            "model_type": "RandomForestClassifier",
            "is_trained": True,
            "model_loaded": True,
            "n_features": 25
        }

        mock_predictor.get_model_info.return_value = mock_model_info

        response = self.client.get("/model/info")

        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["model_type"], "RandomForestClassifier")
        self.assertTrue(data["is_trained"])
        self.assertTrue(data["model_loaded"])
        self.assertEqual(data["n_features"], 25)

    @patch('src.api.main.predictor')
    def test_model_info_endpoint_no_predictor(self, mock_predictor):
        """Test model info endpoint when predictor is not initialized."""
        mock_predictor = None

        with patch('src.api.main.predictor', None):
            response = self.client.get("/model/info")

        self.assertEqual(response.status_code, 503)


class TestAPIValidation(unittest.TestCase):
    """Test cases for API input validation."""

    def setUp(self):
        self.client = TestClient(app)

    def test_age_validation(self):
        """Test age field validation."""
        # Valid ages
        valid_ages = [18, 25, 50, 100]
        base_input = {
            "gender": "Male",
            "age": 35,
            "state": "Maharashtra",
            "number_of_referrals": 0,
            "tenure_in_months": 12,
            "phone_service": "Yes",
            "multiple_lines": "No",
            "internet_service": "Yes",
            "internet_type": "DSL",
            "online_security": "No",
            "online_backup": "No",
            "device_protection_plan": "No",
            "premium_support": "No",
            "streaming_tv": "No",
            "streaming_movies": "No",
            "streaming_music": "No",
            "unlimited_data": "No",
            "paperless_billing": "No",
            "value_deal": "No Deal",
            "contract": "Month-to-Month",
            "payment_method": "Credit Card",
            "monthly_charge": 50.0,
            "total_charges": 600.0,
            "total_refunds": 0.0,
            "total_extra_data_charges": 0.0,
            "total_long_distance_charges": 0.0,
            "total_revenue": 600.0
        }

        with patch('src.api.main.predictor') as mock_predictor:
            mock_predictor.model = Mock()
            mock_predictor.predict.return_value = {
                "prediction": 0,
                "prediction_label": "No Churn",
                "probability": 0.3,
                "confidence": "High",
                "feature_importance": []
            }

            for age in valid_ages:
                test_input = base_input.copy()
                test_input["age"] = age

                response = self.client.post("/predict", json=test_input)
                self.assertEqual(response.status_code, 200,
                                 f"Age {age} should be valid")

        # Invalid ages
        invalid_ages = [17, 101, -5, 150]

        for age in invalid_ages:
            test_input = base_input.copy()
            test_input["age"] = age

            response = self.client.post("/predict", json=test_input)
            self.assertEqual(response.status_code, 422,
                             f"Age {age} should be invalid")

    def test_financial_field_validation(self):
        """Test validation of financial fields."""
        base_input = {
            "gender": "Male",
            "age": 35,
            "state": "Maharashtra",
            "number_of_referrals": 0,
            "tenure_in_months": 12,
            "phone_service": "Yes",
            "multiple_lines": "No",
            "internet_service": "Yes",
            "internet_type": "DSL",
            "online_security": "No",
            "online_backup": "No",
            "device_protection_plan": "No",
            "premium_support": "No",
            "streaming_tv": "No",
            "streaming_movies": "No",
            "streaming_music": "No",
            "unlimited_data": "No",
            "paperless_billing": "No",
            "value_deal": "No Deal",
            "contract": "Month-to-Month",
            "payment_method": "Credit Card",
            "monthly_charge": 50.0,
            "total_charges": 600.0,
            "total_refunds": 0.0,
            "total_extra_data_charges": 0.0,
            "total_long_distance_charges": 0.0,
            "total_revenue": 600.0
        }

        # Test negative monthly charge
        test_input = base_input.copy()
        test_input["monthly_charge"] = -10.0

        response = self.client.post("/predict", json=test_input)
        self.assertEqual(response.status_code, 422)

        # Test very high monthly charge
        test_input = base_input.copy()
        test_input["monthly_charge"] = 2000.0

        response = self.client.post("/predict", json=test_input)
        self.assertEqual(response.status_code, 422)

    def test_enum_validation(self):
        """Test validation of enum fields."""
        base_input = {
            "gender": "Male",
            "age": 35,
            "state": "Maharashtra",
            "number_of_referrals": 0,
            "tenure_in_months": 12,
            "phone_service": "Yes",
            "multiple_lines": "No",
            "internet_service": "Yes",
            "internet_type": "DSL",
            "online_security": "No",
            "online_backup": "No",
            "device_protection_plan": "No",
            "premium_support": "No",
            "streaming_tv": "No",
            "streaming_movies": "No",
            "streaming_music": "No",
            "unlimited_data": "No",
            "paperless_billing": "No",
            "value_deal": "No Deal",
            "contract": "Month-to-Month",
            "payment_method": "Credit Card",
            "monthly_charge": 50.0,
            "total_charges": 600.0,
            "total_refunds": 0.0,
            "total_extra_data_charges": 0.0,
            "total_long_distance_charges": 0.0,
            "total_revenue": 600.0
        }

        # Test invalid gender
        test_input = base_input.copy()
        test_input["gender"] = "Other"

        response = self.client.post("/predict", json=test_input)
        self.assertEqual(response.status_code, 422)

        # Test invalid contract
        test_input = base_input.copy()
        test_input["contract"] = "Invalid Contract"

        response = self.client.post("/predict", json=test_input)
        self.assertEqual(response.status_code, 422)


if __name__ == '__main__':
    unittest.main()
