"""
Streamlit web application for Customer Churn Prediction.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any
import logging

from src.models.predictor import ChurnPredictor
from config.settings import CATEGORICAL_OPTIONS, FEATURE_MAPPINGS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2E86AB;
    }
    .success-card {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
    }
    .warning-card {
        background-color: #fff3cd;
        border-color: #ffeaa7;
        color: #856404;
    }
    .danger-card {
        background-color: #f8d7da;
        border-color: #f5c6cb;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    """Load the churn prediction model."""
    try:
        predictor = ChurnPredictor()
        if predictor.load_model():
            return predictor
        else:
            st.error("Failed to load the prediction model.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def create_input_form() -> Dict[str, Any]:
    """Create the input form for customer data."""

    st.markdown('<div class="section-header">📋 Customer Information</div>',
                unsafe_allow_html=True)

    # Personal Information
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Personal Details")
        gender = st.selectbox(
            "Gender", ["Male", "Female"], help="Customer's gender")
        age = st.number_input("Age", min_value=18,
                              max_value=100, value=35, help="Customer's age")
        state = st.selectbox(
            "State",
            list(FEATURE_MAPPINGS["state_mapping"].keys()),
            help="Customer's state of residence"
        )
        number_of_referrals = st.number_input(
            "Number of Referrals",
            min_value=0,
            max_value=100,
            value=0,
            help="Number of customers referred by this customer"
        )

    with col2:
        st.subheader("Service Details")
        tenure_in_months = st.number_input(
            "Tenure (Months)",
            min_value=0,
            max_value=120,
            value=12,
            help="How long the customer has been with the company"
        )
        value_deal = st.selectbox(
            "Value Deal",
            CATEGORICAL_OPTIONS["value_deals"],
            help="Special deal or promotion"
        )
        contract = st.selectbox(
            "Contract Type",
            CATEGORICAL_OPTIONS["contract_options"],
            help="Type of contract"
        )
        payment_method = st.selectbox(
            "Payment Method",
            CATEGORICAL_OPTIONS["payment_methods"],
            help="How the customer pays"
        )

    # Service Features
    st.markdown('<div class="section-header">📞 Service Features</div>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])
        internet_service = st.selectbox("Internet Service", ["Yes", "No"])
        internet_type = st.selectbox(
            "Internet Type", CATEGORICAL_OPTIONS["internet_types"])

    with col2:
        online_security = st.selectbox("Online Security", ["Yes", "No"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No"])
        device_protection_plan = st.selectbox(
            "Device Protection", ["Yes", "No"])
        premium_support = st.selectbox("Premium Support", ["Yes", "No"])

    with col3:
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])
        streaming_music = st.selectbox("Streaming Music", ["Yes", "No"])
        unlimited_data = st.selectbox("Unlimited Data", ["Yes", "No"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

    # Financial Information
    st.markdown('<div class="section-header">💰 Financial Information</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        monthly_charge = st.number_input(
            "Monthly Charge ($)",
            min_value=0.0,
            max_value=1000.0,
            value=50.0,
            step=5.0,
            help="Monthly service charge"
        )
        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=50000.0,
            value=monthly_charge * tenure_in_months,
            step=50.0,
            help="Total amount charged to the customer"
        )
        total_refunds = st.number_input(
            "Total Refunds ($)",
            min_value=0.0,
            max_value=5000.0,
            value=0.0,
            step=10.0,
            help="Total refunds given to the customer"
        )

    with col2:
        total_extra_data_charges = st.number_input(
            "Extra Data Charges ($)",
            min_value=0.0,
            max_value=5000.0,
            value=0.0,
            step=5.0,
            help="Charges for extra data usage"
        )
        total_long_distance_charges = st.number_input(
            "Long Distance Charges ($)",
            min_value=0.0,
            max_value=5000.0,
            value=0.0,
            step=5.0,
            help="Long distance call charges"
        )
        total_revenue = st.number_input(
            "Total Revenue ($)",
            min_value=0.0,
            max_value=50000.0,
            value=total_charges - total_refunds,
            step=50.0,
            help="Total revenue from the customer"
        )

    # Return all input values as a dictionary
    return {
        "gender": gender,
        "age": age,
        "state": state,
        "number_of_referrals": number_of_referrals,
        "tenure_in_months": tenure_in_months,
        "phone_service": phone_service,
        "multiple_lines": multiple_lines,
        "internet_service": internet_service,
        "internet_type": internet_type,
        "online_security": online_security,
        "online_backup": online_backup,
        "device_protection_plan": device_protection_plan,
        "premium_support": premium_support,
        "streaming_tv": streaming_tv,
        "streaming_movies": streaming_movies,
        "streaming_music": streaming_music,
        "unlimited_data": unlimited_data,
        "paperless_billing": paperless_billing,
        "value_deal": value_deal,
        "contract": contract,
        "payment_method": payment_method,
        "monthly_charge": monthly_charge,
        "total_charges": total_charges,
        "total_refunds": total_refunds,
        "total_extra_data_charges": total_extra_data_charges,
        "total_long_distance_charges": total_long_distance_charges,
        "total_revenue": total_revenue
    }


def display_prediction_results(result: Dict[str, Any]):
    """Display prediction results with visualizations."""

    st.markdown('<div class="section-header">🎯 Prediction Results</div>',
                unsafe_allow_html=True)

    # Main prediction result
    col1, col2, col3 = st.columns(3)

    with col1:
        prediction_class = "success-card" if result["prediction"] == 0 else "danger-card"
        st.markdown(f"""
        <div class="metric-card {prediction_class}">
            <h3>Prediction</h3>
            <h2>{result["prediction_label"]}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        probability_class = "success-card" if result["probability"] < 0.5 else "danger-card"
        st.markdown(f"""
        <div class="metric-card {probability_class}">
            <h3>Churn Probability</h3>
            <h2>{result["probability"]:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        confidence_color = {
            "High": "success-card",
            "Medium": "warning-card",
            "Low": "danger-card"
        }.get(result["confidence"], "metric-card")

        st.markdown(f"""
        <div class="metric-card {confidence_color}">
            <h3>Confidence</h3>
            <h2>{result["confidence"]}</h2>
        </div>
        """, unsafe_allow_html=True)

    # Probability gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=result["probability"] * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Churn Probability (%)"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig_gauge.update_layout(height=300)
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Feature importance
    if result.get("feature_importance"):
        st.markdown(
            '<div class="section-header">📊 Top Feature Influences</div>', unsafe_allow_html=True)

        # Prepare data for visualization
        features = [item["feature"]
                    for item in result["feature_importance"][:10]]
        importances = [item["importance"]
                       for item in result["feature_importance"][:10]]

        # Create horizontal bar chart
        fig_features = px.bar(
            x=importances,
            y=features,
            orientation='h',
            title="Top 10 Most Important Features",
            labels={'x': 'Importance Score', 'y': 'Features'},
            color=importances,
            color_continuous_scale='viridis'
        )

        fig_features.update_layout(
            height=400,
            yaxis={'categoryorder': 'total ascending'}
        )

        st.plotly_chart(fig_features, use_container_width=True)


def display_recommendations(result: Dict[str, Any], input_data: Dict[str, Any]):
    """Display actionable recommendations based on prediction."""

    st.markdown('<div class="section-header">💡 Recommendations</div>',
                unsafe_allow_html=True)

    if result["prediction"] == 1:  # High churn risk
        st.error("⚠️ **High Churn Risk Detected!**")

        recommendations = [
            "🎯 **Immediate Action Required**: Contact customer within 24 hours",
            "💰 **Retention Offer**: Consider special pricing or value deals",
            "📞 **Personal Touch**: Assign dedicated account manager",
            "🔧 **Service Review**: Analyze and address service issues",
            "📊 **Usage Analysis**: Review and optimize service plan"
        ]

        if input_data.get("contract") == "Month-to-Month":
            recommendations.append(
                "📋 **Contract Upgrade**: Offer incentives for longer-term contracts")

        if input_data.get("tenure_in_months", 0) < 12:
            recommendations.append(
                "🕒 **New Customer Focus**: Implement early retention program")

        for rec in recommendations:
            st.markdown(f"- {rec}")

    else:  # Low churn risk
        st.success("✅ **Low Churn Risk - Customer Likely to Stay**")

        recommendations = [
            "🌟 **Upselling Opportunity**: Customer is satisfied, consider premium services",
            "📈 **Growth Potential**: Explore additional service offerings",
            "🎉 **Loyalty Program**: Enroll in rewards program",
            "📝 **Feedback Collection**: Gather insights for service improvement",
            "🔄 **Regular Check-ins**: Maintain positive relationship"
        ]

        for rec in recommendations:
            st.markdown(f"- {rec}")


def main():
    """Main Streamlit application."""

    # Header
    st.markdown('<h1 class="main-header">📊 Customer Churn Prediction System</h1>',
                unsafe_allow_html=True)
    st.markdown("---")

    # Load model
    predictor = load_predictor()
    if not predictor:
        st.stop()

    # Sidebar with model info
    with st.sidebar:
        st.header("Model Information")
        model_info = predictor.get_model_info()

        st.metric("Model Type", model_info.get("model_type", "Unknown"))
        if "n_features" in model_info:
            st.metric("Features", model_info["n_features"])
        if "n_estimators" in model_info:
            st.metric("Trees", model_info["n_estimators"])

        st.markdown("---")
        st.markdown("### 📋 Instructions")
        st.markdown("""
        1. Fill in the customer information
        2. Click 'Predict Churn' button
        3. Review prediction results
        4. Follow recommendations
        """)

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        # Input form
        input_data = create_input_form()

        # Predict button
        if st.button("🔮 Predict Churn", type="primary"):
            try:
                with st.spinner("Making prediction..."):
                    result = predictor.predict(input_data)

                # Store results in session state
                st.session_state.prediction_result = result
                st.session_state.input_data = input_data

            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

    with col2:
        # Display results if available
        if "prediction_result" in st.session_state:
            display_prediction_results(st.session_state.prediction_result)

    # Recommendations section
    if "prediction_result" in st.session_state:
        st.markdown("---")
        display_recommendations(
            st.session_state.prediction_result,
            st.session_state.input_data
        )


if __name__ == "__main__":
    main()
