"""
XGBoost model implementation for time series forecasting
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
import xgboost as xgb
import joblib
from pathlib import Path

from forecasting.config import MODELS_DIR, MODEL_CONFIGS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XGBoostForecaster:
    """
    XGBoost wrapper for time series forecasting
    
    Converts time series to supervised learning problem through:
    - Lag features (past values)
    - Rolling statistics (mean, std)
    - Time-based features (hour, day, month, etc.)
    """
    
    def __init__(self, 
                 n_lags: int = 24,
                 n_estimators: int = 1000,
                 learning_rate: float = 0.01,
                 max_depth: int = 5,
                 **kwargs):
        """
        Initialize XGBoost forecaster
        
        Args:
            n_lags: Number of lag features to create
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate
            max_depth: Maximum tree depth
            **kwargs: Additional XGBoost parameters
        """
        self.n_lags = n_lags
        self.model_params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'objective': 'reg:squarederror',
            'random_state': 42,
            **kwargs
        }
        
        self.model = None
        self.is_fitted = False
        self.train_data = None
        self.value_col = None
        self.feature_names = []
        self.scaler_mean = None
        self.scaler_std = None
        
        logger.info(f"Initialized XGBoost with {n_lags} lags")
        logger.info(f"Model params: n_estimators={n_estimators}, lr={learning_rate}, depth={max_depth}")
    
    def create_features(self, df: pd.DataFrame, 
                       value_col: str = 'consumption_mw') -> pd.DataFrame:
        """
        Create features for XGBoost
        
        Features:
        - Lag features: t-1, t-2, ..., t-n_lags
        - Rolling statistics: mean, std, min, max
        - Time features: hour, day_of_week, month, etc.
        
        Args:
            df: DataFrame with datetime index
            value_col: Name of the value column
            
        Returns:
            DataFrame with features
        """
        features = pd.DataFrame(index=df.index)
        
        # Target variable
        features['target'] = df[value_col]
        
        # 1. Lag features
        for i in range(1, self.n_lags + 1):
            features[f'lag_{i}'] = df[value_col].shift(i)
        
        # 2. Rolling statistics (24-hour window)
        features['rolling_mean_24'] = df[value_col].rolling(window=24, min_periods=1).mean()
        features['rolling_std_24'] = df[value_col].rolling(window=24, min_periods=1).std()
        features['rolling_min_24'] = df[value_col].rolling(window=24, min_periods=1).min()
        features['rolling_max_24'] = df[value_col].rolling(window=24, min_periods=1).max()
        
        # 3. Rolling statistics (7-day window = 168 hours)
        features['rolling_mean_168'] = df[value_col].rolling(window=168, min_periods=1).mean()
        features['rolling_std_168'] = df[value_col].rolling(window=168, min_periods=1).std()
        
        # 4. Time-based features
        features['hour'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek
        features['day_of_month'] = df.index.day
        features['month'] = df.index.month
        features['quarter'] = df.index.quarter
        features['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        # 5. Cyclical encoding for hour (to capture circular nature)
        features['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        
        # 6. Cyclical encoding for day of week
        features['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        features['dow_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        
        return features
    
    def fit(self, df: pd.DataFrame, 
            value_col: str = 'consumption_mw') -> 'XGBoostForecaster':
        """
        Fit XGBoost model to training data
        
        Args:
            df: DataFrame with datetime index and value column
            value_col: Name of the value column
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting XGBoost model...")
        logger.info("Creating features...")
        
        # Create features
        features_df = self.create_features(df, value_col)
        
        # Drop rows with NaN (from lagging)
        features_df = features_df.dropna()
        
        logger.info(f"Features created: {len(features_df)} samples, {len(features_df.columns)-1} features")
        
        # Separate X and y
        X = features_df.drop('target', axis=1)
        y = features_df['target']
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Store training data for recursive prediction
        self.train_data = df.copy()
        self.value_col = value_col

        # Calculate scaling parameters (for normalization)
        self.scaler_mean = y.mean()
        self.scaler_std = y.std()
        
        # Train model
        self.model = xgb.XGBRegressor(**self.model_params)
        self.model.fit(X, y, verbose=False)
        
        self.is_fitted = True
        logger.info("✅ XGBoost model fitted successfully")
        
        return self
    
    def predict(self, horizon: int = 30) -> pd.DataFrame:
        """
        Generate forecast for specified horizon
        
        Uses recursive prediction: predicts one step, adds to history, predicts next
        
        Args:
            horizon: Number of periods to forecast
            
        Returns:
            DataFrame with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        logger.info(f"Generating {horizon}-period forecast (recursive)...")
        
        # Start with training data
        history = self.train_data.copy()
        value_col = self.value_col  # Assume this column name
        
        predictions = []
        
        # Generate future dates
        last_date = history.index[-1]
        freq = pd.infer_freq(history.index)
        if freq is None:
            freq = 'H'
        
        future_dates = pd.date_range(
            start=last_date,
            periods=horizon + 1,
            freq=freq
        )[1:]
        
        # Recursive prediction
        for future_date in future_dates:
            # Create features for this timestamp
            features_df = self.create_features(history, value_col)
            
            # Get last row features (most recent)
            X_pred = features_df.iloc[[-1]].drop('target', axis=1)
            
            # Ensure feature order matches training
            X_pred = X_pred[self.feature_names]
            
            # Predict
            pred = self.model.predict(X_pred)[0]
            predictions.append(pred)
            
            # Add prediction to history for next iteration
            new_row = pd.DataFrame(
                {value_col: [pred]},
                index=[future_date]
            )
            history = pd.concat([history, new_row])
        
        # Create forecast dataframe
        # Simple confidence intervals (±2 std)
        pred_std = self.scaler_std * 0.5  # Conservative estimate
        
        forecast_df = pd.DataFrame({
            'datetime': future_dates,
            'prediction': predictions,
            'lower_bound': np.array(predictions) - 2 * pred_std,
            'upper_bound': np.array(predictions) + 2 * pred_std
        })
        
        logger.info("✅ Forecast generated successfully")
        
        return forecast_df
    
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
        
        # Combine train and test for feature creation
        full_df = pd.concat([self.train_data, test_df])
        
        # Create features
        features_df = self.create_features(full_df, value_col)
        
        # Get only test portion (after dropna from lagging)
        # We need to align with test_df indices
        test_features = features_df.loc[test_df.index]
        test_features = test_features.dropna()
        
        if len(test_features) == 0:
            logger.error("No valid test samples after feature creation")
            raise ValueError("Insufficient data for evaluation")
        
        # Separate X and y
        X_test = test_features.drop('target', axis=1)
        y_true = test_features['target'].values
        
        # Ensure feature order
        X_test = X_test[self.feature_names]
        
        # Predict
        y_pred = self.model.predict(X_test)
        
        logger.info(f"Evaluation on {len(y_true)} test samples")
        logger.info(f"Mean actual: {np.mean(y_true):.2f}, Mean predicted: {np.mean(y_pred):.2f}")
        
        # Calculate metrics
        from forecasting.evaluation.metrics import calculate_metrics
        metrics = calculate_metrics(y_true, y_pred)
        
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores
        
        Returns:
            DataFrame with features and their importance
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        importance = self.model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filename: Optional[str] = None) -> Path:
        """
        Save trained model to disk
        
        Args:
            filename: Optional filename (default: xgboost_model.pkl)
            
        Returns:
            Path to saved model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        if filename is None:
            filename = "xgboost_model.pkl"
        
        filepath = MODELS_DIR / filename
        
        # Save entire forecaster object
        joblib.dump(self, filepath)
        logger.info(f"Model saved to: {filepath}")
        
        return filepath
    
    @classmethod
    def load_model(cls, filepath: Path) -> 'XGBoostForecaster':
        """
        Load trained model from disk
        
        Args:
            filepath: Path to saved model
            
        Returns:
            XGBoostForecaster instance with loaded model
        """
        forecaster = joblib.load(filepath)
        logger.info(f"Model loaded from: {filepath}")
        
        return forecaster
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        return {
            'n_lags': self.n_lags,
            **self.model_params
        }
    
    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return f"XGBoostForecaster({status}, {self.n_lags} lags)"