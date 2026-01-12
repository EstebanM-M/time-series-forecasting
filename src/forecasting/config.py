"""
Configuration settings for the forecasting system
"""
import os
from pathlib import Path
from typing import Dict, Any

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"

# Model directory
MODELS_DIR = PROJECT_ROOT / "models"

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    f"sqlite:///{PROJECT_ROOT / 'forecasting.db'}"
)

# Model configurations
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "prophet": {
        "name": "Prophet",
        "description": "Facebook's Prophet - Robust to outliers and missing data",
        "default_params": {
            "seasonality_mode": "multiplicative",
            "yearly_seasonality": True,
            "weekly_seasonality": True,
            "daily_seasonality": False,
            "changepoint_prior_scale": 0.05,
        },
        "supports_exogenous": True,
        "training_time": "fast",  # 15-30s
    },
    "arima": {
        "name": "ARIMA/SARIMA",
        "description": "Statistical model for time series forecasting",
        "default_params": {
            "order": (1, 1, 1),
            "seasonal_order": (1, 1, 1, 12),
        },
        "supports_exogenous": True,
        "training_time": "medium",  # 30-60s
    },
    "xgboost": {
        "name": "XGBoost",
        "description": "Gradient boosting for time series with feature engineering",
        "default_params": {
            "n_estimators": 1000,
            "learning_rate": 0.01,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "early_stopping_rounds": 50,
        },
        "supports_exogenous": True,
        "training_time": "fast",  # 10-20s
    },
    "lstm": {
        "name": "LSTM",
        "description": "Deep learning model for complex patterns",
        "default_params": {
            "units": 50,
            "epochs": 100,
            "batch_size": 32,
            "learning_rate": 0.001,
            "lookback": 30,
        },
        "supports_exogenous": True,
        "training_time": "slow",  # 60-120s
    },
}

# Validation settings
VALIDATION_CONFIG = {
    "min_data_points": 30,  # Minimum required data points
    "max_data_points": 100000,  # Maximum for free tier
    "required_columns": ["date", "value"],
    "allowed_frequencies": ["D", "H", "W", "M"],  # Daily, Hourly, Weekly, Monthly
    "max_missing_percentage": 20,  # Maximum 20% missing values
}

# Forecast settings
FORECAST_CONFIG = {
    "min_horizon": 1,
    "max_horizon": 365,
    "default_horizon": 30,
    "confidence_intervals": [0.8, 0.95],  # 80% and 95% CI
}

# Business metrics settings
BUSINESS_METRICS_CONFIG = {
    "cost_per_unit_overforecast": 10,  # Cost of overestimating by 1 unit
    "cost_per_unit_underforecast": 20,  # Cost of underestimating by 1 unit
    "baseline_method": "naive",  # naive, seasonal_naive, or historical_average
}

# Streamlit dashboard settings
DASHBOARD_CONFIG = {
    "page_title": "⚡ Energy Demand Forecasting System",
    "page_icon": "⚡",
    "layout": "wide",
    "theme": {
        "primaryColor": "#FF4B4B",
        "backgroundColor": "#FFFFFF",
        "secondaryBackgroundColor": "#F0F2F6",
        "textColor": "#262730",
    },
}

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                  SAMPLE_DATA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)