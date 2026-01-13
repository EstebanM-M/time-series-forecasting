"""
ARIMA/SARIMA model implementation for time series forecasting
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA as StatsARIMA
import joblib
from pathlib import Path
import warnings

from forecasting.config import MODELS_DIR, MODEL_CONFIGS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class ARIMAForecaster:
    """
    Wrapper for ARIMA/SARIMA forecasting model
    
    ARIMA is designed for:
    - Time series with trends
    - Stationary or near-stationary data
    - Linear relationships
    - Short to medium-term forecasting
    """
    
    def __init__(self, 
                 order: Tuple[int, int, int] = (1, 1, 1),
                 seasonal_order: Optional[Tuple[int, int, int, int]] = None,
                 **kwargs):
        """
        Initialize ARIMA forecaster
        
        Args:
            order: ARIMA order (p, d, q)
                - p: autoregressive order
                - d: differencing order
                - q: moving average order
            seasonal_order: Seasonal ARIMA order (P, D, Q, s)
                - P, D, Q: seasonal AR, differencing, MA orders
                - s: seasonal period (e.g., 24 for hourly with daily seasonality)
            **kwargs: Additional SARIMAX parameters
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.kwargs = kwargs
        
        self.model = None
        self.model_fit = None
        self.is_fitted = False
        self.train_data = None
        
        model_type = "SARIMA" if seasonal_order else "ARIMA"
        logger.info(f"Initialized {model_type} with order: {order}")
        if seasonal_order:
            logger.info(f"Seasonal order: {seasonal_order}")
    
    def fit(self, df: pd.DataFrame, 
            value_col: str = 'consumption_mw') -> 'ARIMAForecaster':
        """
        Fit ARIMA model to training data
        
        Args:
            df: DataFrame with datetime index and value column
            value_col: Name of the value column
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting ARIMA model...")
        
        # Extract time series
        y = df[value_col].values
        self.train_data = df.copy()
        
        # Create and fit model
        if self.seasonal_order:
            self.model = SARIMAX(
                y,
                order=self.order,
                seasonal_order=self.seasonal_order,
                **self.kwargs
            )
        else:
            self.model = StatsARIMA(
                y,
                order=self.order,
                **self.kwargs
            )
        
        # Fit model
        self.model_fit = self.model.fit()
        
        self.is_fitted = True
        logger.info("✅ ARIMA model fitted successfully")
        
        return self
    
    def predict(self, horizon: int = 30) -> pd.DataFrame:
        """
        Generate forecast for specified horizon
        
        Args:
            horizon: Number of periods to forecast
            
        Returns:
            DataFrame with predictions and confidence intervals
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        logger.info(f"Generating {horizon}-period forecast...")
        
        # Generate forecast
        forecast_result = self.model_fit.get_forecast(steps=horizon)
        
        # Get predictions and confidence intervals
        predictions = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=0.05)  # 95% CI
        
        # Create future dates
        last_date = self.train_data.index[-1]
        freq = pd.infer_freq(self.train_data.index)
        if freq is None:
            freq = 'H'  # Default to hourly
        
        future_dates = pd.date_range(
            start=last_date,
            periods=horizon + 1,
            freq=freq
        )[1:]  # Exclude the last training date
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'datetime': future_dates,
            'prediction': predictions,  
            'lower_bound': conf_int[:, 0],
            'upper_bound': conf_int[:, 1]
        })
        
        logger.info("✅ Forecast generated successfully")
        
        return forecast_df
    
    def predict_in_sample(self) -> np.ndarray:
        """
        Get in-sample predictions (fitted values)
        
        Returns:
            Array of fitted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        return self.model_fit.fittedvalues
    
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
        
        # Generate forecast for test period length
        test_horizon = len(test_df)
        forecast_df = self.predict(horizon=test_horizon)
        
        # Extract actual and predicted values
        y_true = test_df[value_col].values
        y_pred = forecast_df['prediction'].values
        
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
    
    def get_model_summary(self) -> str:
        """
        Get model summary statistics
        
        Returns:
            Model summary as string
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        return str(self.model_fit.summary())
    
    def save_model(self, filename: Optional[str] = None) -> Path:
        """
        Save trained model to disk
        
        Args:
            filename: Optional filename (default: arima_model.pkl)
            
        Returns:
            Path to saved model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        if filename is None:
            filename = "arima_model.pkl"
        
        filepath = MODELS_DIR / filename
        
        # Save model fit object
        joblib.dump(self.model_fit, filepath)
        logger.info(f"Model saved to: {filepath}")
        
        return filepath
    
    @classmethod
    def load_model(cls, filepath: Path) -> 'ARIMAForecaster':
        """
        Load trained model from disk
        
        Args:
            filepath: Path to saved model
            
        Returns:
            ARIMAForecaster instance with loaded model
        """
        forecaster = cls()
        forecaster.model_fit = joblib.load(filepath)
        forecaster.is_fitted = True
        
        logger.info(f"Model loaded from: {filepath}")
        
        return forecaster
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        return {
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            **self.kwargs
        }
    
    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        model_type = "SARIMA" if self.seasonal_order else "ARIMA"
        return f"{model_type}Forecaster({status})"