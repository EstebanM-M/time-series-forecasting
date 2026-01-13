"""
Prophet model implementation for time series forecasting
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from prophet import Prophet
import joblib
from pathlib import Path

from forecasting.config import MODELS_DIR, MODEL_CONFIGS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProphetForecaster:
    """
    Wrapper for Facebook Prophet forecasting model
    
    Prophet is designed for forecasting time series data with:
    - Strong seasonal patterns
    - Historical trend changes
    - Holiday effects
    - Missing data robustness
    """
    
    def __init__(self, **kwargs):
        """
        Initialize Prophet forecaster
        
        Args:
            **kwargs: Prophet model parameters
                - seasonality_mode: 'additive' or 'multiplicative'
                - yearly_seasonality: bool or int
                - weekly_seasonality: bool or int
                - daily_seasonality: bool or int
                - changepoint_prior_scale: float (0.001-0.5)
                - seasonality_prior_scale: float (default 10.0)
        """
        # Get default config
        default_params = MODEL_CONFIGS['prophet']['default_params'].copy()
        
        # Override with user params
        default_params.update(kwargs)
        
        self.params = default_params
        self.model = None
        self.is_fitted = False
        self.train_data = None
        self.forecast = None
        
        logger.info(f"Initialized Prophet with params: {self.params}")
    
    def prepare_data(self, df: pd.DataFrame, 
                    value_col: str = 'consumption_mw') -> pd.DataFrame:
        """
        Prepare data for Prophet (requires 'ds' and 'y' columns)
        
        Args:
            df: DataFrame with datetime index and value column
            value_col: Name of the value column
            
        Returns:
            DataFrame in Prophet format
        """
        # Prophet expects columns named 'ds' and 'y'
        prophet_df = pd.DataFrame({
            'ds': df.index,
            'y': df[value_col].values
        })
        
        return prophet_df
    
    def fit(self, df: pd.DataFrame, 
            value_col: str = 'consumption_mw') -> 'ProphetForecaster':
        """
        Fit Prophet model to training data
        
        Args:
            df: DataFrame with datetime index and value column
            value_col: Name of the value column
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting Prophet model...")
        
        # Prepare data
        train_df = self.prepare_data(df, value_col)
        self.train_data = train_df.copy()
        
        # Initialize model with parameters
        self.model = Prophet(**self.params)
        
        # Fit model
        self.model.fit(train_df)
        
        self.is_fitted = True
        logger.info("✅ Prophet model fitted successfully")
        
        return self
    
    def predict(self, horizon: int = 30, 
                freq: str = 'H') -> pd.DataFrame:
        """
        Generate forecast for specified horizon
        
        Args:
            horizon: Number of periods to forecast
            freq: Frequency of predictions ('H', 'D', 'W', 'M')
            
        Returns:
            DataFrame with predictions and confidence intervals
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        logger.info(f"Generating {horizon}-period forecast...")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(
            periods=horizon,
            freq=freq
        )
        
        # Make predictions
        forecast = self.model.predict(future)
        
        self.forecast = forecast
        
        logger.info("✅ Forecast generated successfully")
        
        return forecast
    
    def get_forecast_summary(self, horizon: int = 30) -> pd.DataFrame:
        """
        Get clean forecast summary with key columns
        
        Args:
            horizon: Number of future periods to return
            
        Returns:
            DataFrame with datetime, prediction, and confidence intervals
        """
        if self.forecast is None:
            raise ValueError("Must call predict() first")
        
        # Get only future predictions
        forecast = self.forecast.tail(horizon).copy()
        
        # Select key columns
        summary = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        summary.columns = ['datetime', 'prediction', 'lower_bound', 'upper_bound']
        
        return summary
    
    def get_components(self) -> pd.DataFrame:
        """
        Get forecast components (trend, seasonality, etc.)
        
        Returns:
            DataFrame with component values
        """
        if self.forecast is None:
            raise ValueError("Must call predict() first")
        
        components = self.forecast[[
            'ds', 'trend', 'weekly', 'yearly'
        ]].copy()
        
        return components
    
    def evaluate_on_test(self, test_df: pd.DataFrame,
                    value_col: str = 'consumption_mw') -> Dict[str, float]:
        """
        Evaluate model on test data
        
        Args:
            test_df: Test DataFrame with datetime index
            value_col: Name of the value column
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        logger.info("Evaluating on test set...")
        
        # Create dataframe with test dates for Prophet to predict
        test_dates = pd.DataFrame({'ds': test_df.index})
        
        # Make predictions on test dates
        forecast = self.model.predict(test_dates)
        
        # Extract actual and predicted values
        y_true = test_df[value_col].values
        y_pred = forecast['yhat'].values
        
        # Ensure same length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        logger.info(f"Evaluation on {min_len} test samples")
        logger.info(f"Mean actual: {np.mean(y_true):.2f}, Mean predicted: {np.mean(y_pred):.2f}")
        
        # Calculate metrics
        from forecasting.evaluation.metrics import calculate_metrics
        metrics = calculate_metrics(y_true, y_pred)
        
        return metrics
    
    def save_model(self, filename: Optional[str] = None) -> Path:
        """
        Save trained model to disk
        
        Args:
            filename: Optional filename (default: prophet_model.pkl)
            
        Returns:
            Path to saved model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        if filename is None:
            filename = "prophet_model.pkl"
        
        filepath = MODELS_DIR / filename
        
        # Save model
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to: {filepath}")
        
        return filepath
    
    @classmethod
    def load_model(cls, filepath: Path) -> 'ProphetForecaster':
        """
        Load trained model from disk
        
        Args:
            filepath: Path to saved model
            
        Returns:
            ProphetForecaster instance with loaded model
        """
        forecaster = cls()
        forecaster.model = joblib.load(filepath)
        forecaster.is_fitted = True
        
        logger.info(f"Model loaded from: {filepath}")
        
        return forecaster
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        return self.params.copy()
    
    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return f"ProphetForecaster({status})"