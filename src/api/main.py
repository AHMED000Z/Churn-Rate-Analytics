"""
FastAPI application for Customer Churn Prediction System.
"""
import logging
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn

from config.settings import API_CONFIG, CATEGORICAL_OPTIONS, FEATURE_MAPPINGS
from src.models.schemas import (
    ChurnPredictionInput,
    ChurnPredictionOutput,
    APIMetadata,
    HealthCheckResponse
)
from src.models.predictor import load_model_for_api

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=API_CONFIG["title"],
    description=API_CONFIG["description"],
    version=API_CONFIG["version"],
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
predictor = None


@app.on_event("startup")
async def startup_event():
    """Load model when the application starts."""
    global predictor
    try:
        predictor = load_model_for_api()
        logger.info("Application startup completed")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with welcome message."""
    return {
        "message": "Welcome to the Customer Churn Prediction API",
        "version": API_CONFIG["version"],
        "docs_url": "/docs",
        "health_check": "/health"
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""

    model_info = predictor.get_model_info() if predictor else {
        "model_loaded": False}

    return HealthCheckResponse(
        status="healthy" if predictor and predictor.model else "degraded",
        model_loaded=model_info.get("model_loaded", False),
        model_type=model_info.get("model_type"),
        features_count=model_info.get("n_features")
    )


@app.get("/metadata", response_model=APIMetadata, tags=["Metadata"])
async def get_metadata():
    """Get metadata for form dropdowns and validation."""

    return APIMetadata(
        states=list(FEATURE_MAPPINGS["state_mapping"].keys()),
        value_deals=CATEGORICAL_OPTIONS["value_deals"],
        contracts=CATEGORICAL_OPTIONS["contract_options"],
        payment_methods=CATEGORICAL_OPTIONS["payment_methods"],
        internet_types=CATEGORICAL_OPTIONS["internet_types"]
    )


@app.post("/predict", response_model=ChurnPredictionOutput, tags=["Prediction"])
async def predict_churn(data: ChurnPredictionInput):
    """
    Predict customer churn based on input features.

    Args:
        data: Customer data for prediction

    Returns:
        Churn prediction with probability and feature importance
    """

    if predictor is None or predictor.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check service health."
        )

    try:
        # Convert Pydantic model to dictionary
        input_data = data.dict()

        # Make prediction
        result = predictor.predict(input_data)

        return ChurnPredictionOutput(**result)

    except ValueError as e:
        # Input validation errors
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        # Internal server errors
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Internal server error during prediction")


@app.get("/model/info", tags=["Model"])
async def get_model_info():
    """Get information about the loaded model."""

    if predictor is None:
        raise HTTPException(
            status_code=503, detail="Predictor not initialized")

    return predictor.get_model_info()


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""

    logger.error(f"Unhandled exception: {str(exc)}")

    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected error occurred",
            "type": "internal_server_error"
        }
    )


# Mount static files for the web interface
try:
    app.mount("/app", StaticFiles(directory="src/web/static",
              html=True), name="static")
    logger.info("Static files mounted successfully")
except Exception as e:
    logger.warning(f"Could not mount static files: {str(e)}")


def run_server():
    """Run the FastAPI server."""
    uvicorn.run(
        "src.api.main:app",
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        reload=API_CONFIG["debug"],
        log_level="info"
    )


if __name__ == "__main__":
    run_server()
