import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load the saved model

    # Define the mapping for the 'state' column
state_mapping = {
        'Andhra Pradesh': 1,
        'Assam': 2,
        'Bihar': 3,
        'Chhattisgarh': 4,
        'Delhi': 5,
        'Gujarat': 6,
        'Haryana': 7,
        'Jammu & Kashmir': 8,
        'Jharkhand': 9,
        'Karnataka': 10,
        'Kerala': 11,
        'Madhya Pradesh': 12,
        'Maharashtra': 13,
        'Odisha': 14,
        'Puducherry': 15,
        'Punjab': 16,
        'Rajasthan': 17,
        'Tamil Nadu': 18,
        'Telangana': 19,
        'Uttar Pradesh': 20,
        'Uttarakhand': 21,
        'West Bengal': 22
    }
label_encode_features = ['gender', 'phone_service', 'multiple_lines', 'internet_service', 'internet_type', 'online_security', 'online_backup', 'device_protection_plan', 'premium_support', 'streaming_tv', 'streaming_movies', 'streaming_music', 'unlimited_data', 'paperless_billing']
one_hot_encode_features = ['value_deal', 'contract', 'payment_method']
# Define preprocessing steps
def preprocess_input(data):


    # Apply the mapping to the 'state' column
    data['state'] = data['state'].map(state_mapping)
    # Define the features for label encoding and one hot encoding


    # Create a dictionary of label encoders for each feature
    label_encoders = {feature: LabelEncoder() for feature in label_encode_features}

    # Apply label encoding to the specified features
    for feature in label_encode_features:
        data[feature] = label_encoders[feature].fit_transform(data[feature])

    # Apply one hot encoding to the specified features with consistent naming
    data = pd.get_dummies(data, columns=one_hot_encode_features)
    
    # Ensure consistent feature names
    data.columns = data.columns.str.replace(' ', '_').str.replace('-', '_').str.lower()
    
    # Define the features to be normalized
    normalize_features = ['age', 'number_of_referrals', 'tenure_in_months',  'device_protection_plan', 'premium_support', 'monthly_charge', 'total_charges', 'total_refunds', 'total_long_distance_charges', 'total_revenue']

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Apply the scaler to the specified features
    data[normalize_features] = scaler.fit_transform(data[normalize_features])
    return data

# Streamlit app
st.title('Customer Churn Prediction')

# Input form
with st.form("input_form"):
    st.header("Customer Information")
    
    # Personal Information
    age = st.number_input('Age', min_value=18, max_value=100)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    state = st.selectbox('State', ['Andhra Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Delhi', 'Gujarat', 'Haryana', 'Jammu & Kashmir', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Odisha', 'Puducherry', 'Punjab', 'Rajasthan', 'Tamil Nadu', 'Telangana', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal'])
    
    # Service Information
    number_of_referrals = st.number_input('Number of Referrals', min_value=0, max_value=100)
    tenure_in_months = st.number_input('Tenure in Months', min_value=0, max_value=120)
    value_deal = st.selectbox('Value deal', ['Deal 1', 'Deal 2', 'Deal 3','Deal 4', 'Deal 5', 'No Deal'])
    phone_service = st.selectbox('Phone Service', ['Yes', 'No'])
    multiple_lines = st.selectbox('Multiple Lines', ['Yes', 'No', 'No phone service'])
    internet_service = st.selectbox('Internet Service', ['Yes', 'No'])
    internet_type = st.selectbox('Internet Type', ['DSL', 'Fiber Optic', 'Cable'])
    online_security = st.selectbox('Online Security', ['Yes', 'No'])
    online_backup = st.selectbox('Online Backup', ['Yes', 'No'])
    device_protection_plan = st.selectbox('Device Protection Plan', ['Yes', 'No'])
    premium_support = st.selectbox('Premium Support', ['Yes', 'No'])
    streaming_tv = st.selectbox('Streaming TV', ['Yes', 'No'])
    streaming_movies = st.selectbox('Streaming Movies', ['Yes', 'No'])
    streaming_music = st.selectbox('Streaming Music', ['Yes', 'No'])
    unlimited_data = st.selectbox('Unlimited Data', ['Yes', 'No'])
    contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
    payment_method = st.selectbox('Payment Method', ['Credit card', 'Bank Withdrawl', 'Mailed check'])
    monthly_charge = st.number_input('Monthly Charge', min_value=0.0, max_value=500.0)
    total_charges = st.number_input('Total Charges', min_value=0.0, max_value=10000.0)
    total_refunds = st.number_input('Total Refunds', min_value=0.0, max_value=1000.0)
    total_long_distance_charges = st.number_input('Total Long Distance Charges', min_value=0.0, max_value=1000.0)
    total_revenue = st.number_input('Total Revenue', min_value=0.0, max_value=10000.0)
    # Add more input fields for all required features...
    
    submitted = st.form_submit_button("Predict Churn")

if submitted:
    # Create input DataFrame
    input_data = pd.DataFrame({
        'gender': [gender],
        'age': [age],
        'state': [state],
        'number_of_referrals': [number_of_referrals],
        'tenure_in_months': [tenure_in_months],
        'value_deal': [value_deal],
        'phone_service': [phone_service],
        'multiple_lines': [multiple_lines],
        'internet_service': [internet_service],
        'internet_type': [internet_type],
        'online_security': [online_security],
        'online_backup': [online_backup],
        'device_protection_plan': [device_protection_plan],
        'premium_support': [premium_support],
        'streaming_tv': [streaming_tv],
        'streaming_movies': [streaming_movies],
        'streaming_music': [streaming_music],
        'unlimited_data': [unlimited_data],
        'contract': [contract],
        'paperless_billing': [paperless_billing],
        'payment_method': [payment_method],
        'monthly_charge': [monthly_charge],
        'total_charges': [total_charges],
        'total_refunds': [total_refunds],
        'total_long_distance_charges': [total_long_distance_charges],
        'total_revenue': [total_revenue]
            # Add all other features here...
    })
    
    # Preprocess input
    processed_data = preprocess_input(input_data)
    model = joblib.load('best_model.pkl')
    # Ensure processed data has same columns as training data
    missing_cols = set(model.feature_names_in_) - set(processed_data.columns)
    for col in missing_cols:
        processed_data[col] = 0
    
    # Reorder columns to match training data
    processed_data = processed_data[model.feature_names_in_]
    
    # Make prediction
    prediction = model.predict(processed_data)
    
    # Display result
    if prediction[0] == 1:
        st.error('This customer is likely to churn.')
    else:
        st.success('This customer is not likely to churn.')