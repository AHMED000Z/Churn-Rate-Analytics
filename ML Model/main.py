import joblib
import pandas as pd
import numpy as np
import warnings
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from typing import Optional, List

# Initialize FastAPI app
app = FastAPI(title="Churn Prediction API",
    description="API for predicting customer churn using a Random Forest model")

# Allow CORS for all origins to make frontend development easier
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
try:
    model = joblib.load("model//best_model.pkl")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# State mapping for reference
state_mapping = {
    'Andhra Pradesh': 1, 'Assam': 2, 'Bihar': 3, 'Chhattisgarh': 4,
    'Delhi': 5, 'Gujarat': 6, 'Haryana': 7, 'Jammu & Kashmir': 8,
    'Jharkhand': 9, 'Karnataka': 10, 'Kerala': 11, 'Madhya Pradesh': 12,
    'Maharashtra': 13, 'Odisha': 14, 'Puducherry': 15, 'Punjab': 16,
    'Rajasthan': 17, 'Tamil Nadu': 18, 'Telangana': 19, 'Uttar Pradesh': 20,
    'Uttarakhand': 21, 'West Bengal': 22
}

# Value deal options
value_deals = ["Deal 1", "Deal 2", "Deal 3", "Deal 4", "Deal 5", "No Deal"]

# Contract options
contract_options = ["Month-to-Month", "One Year", "Two Year"]

# Payment method options
payment_methods = ["Bank Withdrawal", "Credit Card", "Mailed Check"]

# Input data model
class ChurnData(BaseModel):
    gender: str  # Male/Female
    age: int
    state: str
    number_of_referrals: int
    tenure_in_months: int
    phone_service: str  # Yes/No
    multiple_lines: str  # Yes/No
    internet_service: str  # Yes/No
    internet_type: str
    online_security: str  # Yes/No
    online_backup: str  # Yes/No
    device_protection_plan: str  # Yes/No
    premium_support: str  # Yes/No
    streaming_tv: str  # Yes/No
    streaming_movies: str  # Yes/No
    streaming_music: str  # Yes/No
    unlimited_data: str  # Yes/No
    paperless_billing: str  # Yes/No
    monthly_charge: float
    total_charges: float
    total_refunds: float
    total_extra_data_charges: float
    total_long_distance_charges: float
    total_revenue: float
    value_deal: str
    contract: str
    payment_method: str

# Response model
class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    probability: float
    feature_importance: Optional[List[dict]] = None

@app.get("/")
async def root():
    return {"message": "Welcome to the Churn Prediction API"}

@app.get("/metadata")
async def get_metadata():
    """Returns metadata for form dropdowns"""
    return {
        "states": list(state_mapping.keys()),
        "value_deals": value_deals,
        "contracts": contract_options,
        "payment_methods": payment_methods
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(data: ChurnData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert input data to dataframe with detailed logging
        print("Converting input data to DataFrame...")
        input_data = data.dict()
        print(f"Input data received: {input_data}")
        input_df = pd.DataFrame([input_data])
        
        # Preprocessing steps similar to training with logging
        print("Starting preprocessing steps...")
        
        # 1. Map state
        print(f"Mapping state: {input_df['state'].values[0]}")
        if input_df['state'].values[0] not in state_mapping:
            print(f"Warning: State {input_df['state'].values[0]} not found in mapping")
            raise HTTPException(status_code=400, detail=f"Invalid state: {input_df['state'].values[0]}")
        input_df['state'] = input_df['state'].map(state_mapping)
        
        # 2. Label encoding
        print("Applying label encoding...")
        binary_features = ['gender', 'phone_service', 'multiple_lines', 'internet_service', 
                           'online_security', 'online_backup', 'device_protection_plan', 
                           'premium_support', 'streaming_tv', 'streaming_movies', 
                           'streaming_music', 'unlimited_data', 'paperless_billing']
        
        for feature in binary_features:
            print(f"Encoding {feature}: {input_df[feature].values[0]}")
            if feature == 'gender':
                input_df[feature] = input_df[feature].map({'Male': 1, 'Female': 0})
            else:
                input_df[feature] = input_df[feature].map({'Yes': 1, 'No': 0})
        
        # Internet type encoding
        print(f"Encoding internet_type: {input_df['internet_type'].values[0]}")
        internet_types = ['DSL', 'Fiber Optic', 'Cable']
        if input_df['internet_type'].values[0] not in internet_types:
            print(f"Warning: Internet type {input_df['internet_type'].values[0]} not found in list")
            raise HTTPException(status_code=400, detail=f"Invalid internet type: {input_df['internet_type'].values[0]}")
        input_df['internet_type'] = input_df['internet_type'].map({v: i for i, v in enumerate(internet_types)})
        
        # 3. One-hot encoding
        print("Applying one-hot encoding...")
        print(f"Value deal: {input_df['value_deal'].values[0]}")
        print(f"Contract: {input_df['contract'].values[0]}")
        print(f"Payment method: {input_df['payment_method'].values[0]}")
        
        # Check if values are valid
        if input_df['value_deal'].values[0] not in value_deals:
            print(f"Warning: Value deal {input_df['value_deal'].values[0]} not found in list")
            raise HTTPException(status_code=400, detail=f"Invalid value deal: {input_df['value_deal'].values[0]}")
        
        if input_df['contract'].values[0] not in contract_options:
            print(f"Warning: Contract {input_df['contract'].values[0]} not found in list")
            raise HTTPException(status_code=400, detail=f"Invalid contract: {input_df['contract'].values[0]}")
        
        if input_df['payment_method'].values[0] not in payment_methods:
            print(f"Warning: Payment method {input_df['payment_method'].values[0]} not found in list")
            raise HTTPException(status_code=400, detail=f"Invalid payment method: {input_df['payment_method'].values[0]}")
        
        # Create dummy variables
        value_deal_dummies = pd.get_dummies(input_df['value_deal'], prefix='value_deal')
        contract_dummies = pd.get_dummies(input_df['contract'], prefix='contract')
        payment_dummies = pd.get_dummies(input_df['payment_method'], prefix='payment_method')
        
        # Drop original columns
        input_df = input_df.drop(['value_deal', 'contract', 'payment_method'], axis=1)
        
        # Combine all dataframes
        input_df = pd.concat([input_df, value_deal_dummies, contract_dummies, payment_dummies], axis=1)
        
        # Ensure we have all the one-hot columns needed
        print("Ensuring all required one-hot columns exist...")
        for deal in value_deals:
            col_name = f'value_deal_{deal}'
            if col_name not in input_df.columns:
                print(f"Adding missing column: {col_name}")
                input_df[col_name] = 0
                
        for contract in contract_options:
            col_name = f'contract_{contract}'
            if col_name not in input_df.columns:
                print(f"Adding missing column: {col_name}")
                input_df[col_name] = 0
                
        for method in payment_methods:
            col_name = f'payment_method_{method}'
            if col_name not in input_df.columns:
                print(f"Adding missing column: {col_name}")
                input_df[col_name] = 0
        
        # 4. Normalization
        print("Applying normalization...")
        normalize_features = ['age', 'number_of_referrals', 'tenure_in_months',
                             'monthly_charge', 'total_charges', 'total_refunds', 
                             'total_long_distance_charges', 'total_revenue']
        
        for feature in normalize_features:
            print(f"Normalizing {feature}: {input_df[feature].values[0]}")
            if feature == 'age':
                input_df[feature] = input_df[feature] / 100
            elif feature == 'number_of_referrals':
                input_df[feature] = input_df[feature] / 15
            elif feature == 'tenure_in_months':
                input_df[feature] = input_df[feature] / 70
            else:
                input_df[feature] = input_df[feature] / 1000
        
        # Remove 'customer_status' if it exists
        if 'customer_status' in input_df.columns:
            print("Removing customer_status column...")
            input_df = input_df.drop(['customer_status'], axis=1)
        
        # Print all columns before prediction
        print(f"Final input dataframe columns: {input_df.columns.tolist()}")
        print(f"Number of features: {len(input_df.columns)}")
        
        # If model expects specific features, check them
        if hasattr(model, 'feature_names_in_'):
            model_features = model.feature_names_in_.tolist()
            print(f"Model expects {len(model_features)} features: {model_features}")
            
            missing_features = [f for f in model_features if f not in input_df.columns]
            extra_features = [f for f in input_df.columns if f not in model_features]
            
            if missing_features:
                print(f"Warning: Missing features required by model: {missing_features}")
                for feature in missing_features:
                    input_df[feature] = 0  # Add missing features with default values
                    
            if extra_features:
                print(f"Warning: Extra features not used by model: {extra_features}")
                input_df = input_df.drop(extra_features, axis=1)
                
            # Reorder columns to match model's expected order
            input_df = input_df[model_features]
        
        # Make prediction
        print("Making prediction...")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]  # Probability of churning
        
        print(f"Prediction result: {prediction}, Probability: {probability}")
        
        # Get feature importances if it's a Random Forest model
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_names = input_df.columns
            feature_importance = [
                {"feature": str(feature), "importance": float(imp)}
                for feature, imp in sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
                if imp > 0
            ][:10]  # Top 10 features
        
        return {
            "prediction": int(prediction),
            "prediction_label": "Churn" if prediction == 1 else "No Churn",
            "probability": float(probability),
            "feature_importance": feature_importance
        }
    
    except Exception as e:
        import traceback
        print(f"Error during prediction: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Mount static files for the frontend
app.mount("/app", StaticFiles(directory="static", html=True), name="static")