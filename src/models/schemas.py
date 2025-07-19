"""
Data schemas and validation models for the Customer Churn Prediction System.
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from enum import Enum


class GenderEnum(str, Enum):
    MALE = "Male"
    FEMALE = "Female"


class YesNoEnum(str, Enum):
    YES = "Yes"
    NO = "No"


class ValueDealEnum(str, Enum):
    DEAL_1 = "Deal 1"
    DEAL_2 = "Deal 2"
    DEAL_3 = "Deal 3"
    DEAL_4 = "Deal 4"
    DEAL_5 = "Deal 5"
    NO_DEAL = "No Deal"


class ContractEnum(str, Enum):
    MONTH_TO_MONTH = "Month-to-Month"
    ONE_YEAR = "One Year"
    TWO_YEAR = "Two Year"


class PaymentMethodEnum(str, Enum):
    BANK_WITHDRAWAL = "Bank Withdrawal"
    CREDIT_CARD = "Credit Card"
    MAILED_CHECK = "Mailed Check"


class InternetTypeEnum(str, Enum):
    DSL = "DSL"
    FIBER_OPTIC = "Fiber Optic"
    CABLE = "Cable"


class StateEnum(str, Enum):
    ANDHRA_PRADESH = "Andhra Pradesh"
    ASSAM = "Assam"
    BIHAR = "Bihar"
    CHHATTISGARH = "Chhattisgarh"
    DELHI = "Delhi"
    GUJARAT = "Gujarat"
    HARYANA = "Haryana"
    JAMMU_KASHMIR = "Jammu & Kashmir"
    JHARKHAND = "Jharkhand"
    KARNATAKA = "Karnataka"
    KERALA = "Kerala"
    MADHYA_PRADESH = "Madhya Pradesh"
    MAHARASHTRA = "Maharashtra"
    ODISHA = "Odisha"
    PUDUCHERRY = "Puducherry"
    PUNJAB = "Punjab"
    RAJASTHAN = "Rajasthan"
    TAMIL_NADU = "Tamil Nadu"
    TELANGANA = "Telangana"
    UTTAR_PRADESH = "Uttar Pradesh"
    UTTARAKHAND = "Uttarakhand"
    WEST_BENGAL = "West Bengal"


class ChurnPredictionInput(BaseModel):
    """Input schema for churn prediction."""

    # Personal Information
    gender: GenderEnum = Field(..., description="Customer gender")
    age: int = Field(..., ge=18, le=100, description="Customer age")
    state: StateEnum = Field(..., description="Customer state")

    # Service Information
    number_of_referrals: int = Field(..., ge=0,
                                     le=100, description="Number of referrals")
    tenure_in_months: int = Field(..., ge=0, le=120,
                                  description="Tenure in months")

    # Service Features
    phone_service: YesNoEnum = Field(..., description="Has phone service")
    multiple_lines: YesNoEnum = Field(..., description="Has multiple lines")
    internet_service: YesNoEnum = Field(...,
                                        description="Has internet service")
    internet_type: InternetTypeEnum = Field(...,
                                            description="Type of internet service")
    online_security: YesNoEnum = Field(..., description="Has online security")
    online_backup: YesNoEnum = Field(..., description="Has online backup")
    device_protection_plan: YesNoEnum = Field(
        ..., description="Has device protection plan")
    premium_support: YesNoEnum = Field(..., description="Has premium support")
    streaming_tv: YesNoEnum = Field(..., description="Has streaming TV")
    streaming_movies: YesNoEnum = Field(...,
                                        description="Has streaming movies")
    streaming_music: YesNoEnum = Field(..., description="Has streaming music")
    unlimited_data: YesNoEnum = Field(..., description="Has unlimited data")

    # Billing Information
    paperless_billing: YesNoEnum = Field(...,
                                         description="Uses paperless billing")
    value_deal: ValueDealEnum = Field(..., description="Value deal type")
    contract: ContractEnum = Field(..., description="Contract type")
    payment_method: PaymentMethodEnum = Field(...,
                                              description="Payment method")

    # Financial Information
    monthly_charge: float = Field(..., ge=0, le=1000,
                                  description="Monthly charge")
    total_charges: float = Field(..., ge=0, le=50000,
                                 description="Total charges")
    total_refunds: float = Field(..., ge=0, le=5000,
                                 description="Total refunds")
    total_extra_data_charges: float = Field(..., ge=0,
                                            le=5000, description="Total extra data charges")
    total_long_distance_charges: float = Field(
        ..., ge=0, le=5000, description="Total long distance charges")
    total_revenue: float = Field(..., ge=0, le=50000,
                                 description="Total revenue")

    @validator('total_charges')
    def validate_total_charges(cls, v, values):
        if 'monthly_charge' in values and 'tenure_in_months' in values:
            expected_min = values['monthly_charge'] * \
                values['tenure_in_months'] * 0.5
            if v < expected_min:
                raise ValueError(
                    'Total charges seem too low for the given monthly charge and tenure')
        return v


class ChurnPredictionOutput(BaseModel):
    """Output schema for churn prediction."""

    prediction: int = Field(...,
                            description="Churn prediction (0: No Churn, 1: Churn)")
    prediction_label: str = Field(..., description="Human-readable prediction")
    probability: float = Field(..., ge=0, le=1,
                               description="Probability of churning")
    confidence: str = Field(...,
                            description="Confidence level (Low/Medium/High)")
    feature_importance: Optional[List[dict]] = Field(
        None, description="Top feature importances")

    @validator('confidence', pre=True, always=True)
    def set_confidence(cls, v, values):
        if 'probability' in values:
            prob = values['probability']
            if prob <= 0.3 or prob >= 0.7:
                return "High"
            elif prob <= 0.4 or prob >= 0.6:
                return "Medium"
            else:
                return "Low"
        return "Unknown"


class FeatureImportance(BaseModel):
    """Schema for feature importance."""

    feature: str = Field(..., description="Feature name")
    importance: float = Field(..., ge=0, le=1, description="Importance score")


class ModelMetrics(BaseModel):
    """Schema for model performance metrics."""

    accuracy: float = Field(..., ge=0, le=1)
    precision: float = Field(..., ge=0, le=1)
    recall: float = Field(..., ge=0, le=1)
    f1_score: float = Field(..., ge=0, le=1)
    auc_score: float = Field(..., ge=0, le=1)


class APIMetadata(BaseModel):
    """Schema for API metadata."""

    states: List[str] = Field(..., description="Available states")
    value_deals: List[str] = Field(..., description="Available value deals")
    contracts: List[str] = Field(..., description="Available contract types")
    payment_methods: List[str] = Field(...,
                                       description="Available payment methods")
    internet_types: List[str] = Field(...,
                                      description="Available internet types")


class HealthCheckResponse(BaseModel):
    """Schema for health check response."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_type: Optional[str] = Field(None, description="Type of loaded model")
    features_count: Optional[int] = Field(
        None, description="Number of features expected by model")
