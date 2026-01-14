"""
LSTM model implementation for time series forecasting
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import joblib
from pathlib import Path

from forecasting.config import MODELS_DIR, MODEL_CONFIGS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')


class LSTMForecaster:
    """
    LSTM neural network wrapper for time series forecasting
    
    Converts time series to sequences for supervised learning through:
    - Sliding window approach (lookback periods)
    - Normalization for neural network training
    - Multi-step or single-step prediction
    """
    
    def __init__(self, 
                 lookback: int = 30,
                 units: int = 50,
                 epochs: int = 100,
                 batch_size: int = 32,
                 learning_rate: float = 0.001,
                 dropout: float = 0.2,
                 **kwargs):
        """
        Initialize LSTM forecaster
        
        Args:
            lookback: Number of past timesteps to use as input
            units: Number of LSTM units in hidden layer
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            dropout: Dropout rate for regularization
            **kwargs: Additional parameters
        """
        self.lookback = lookback
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.kwargs = kwargs
        
        self.model = None
        self.is_fitted = False
        self.train_data = None
        self.value_col = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.history = None
        
        logger.info(f"Initialized LSTM with lookback={lookback}, units={units}")
    
    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training
        
        Converts time series [1,2,3,4,5,6] with lookback=3 to:
        X: [[1,2,3], [2,3,4], [3,4,5]]
        y: [4, 5, 6]
        
        Args:
            data: 1D array of time series values
            
        Returns:
            X (sequences), y (targets)
        """
        X, y = [], []
        
        for i in range(self.lookback, len(data)):
            X.append(data[i-self.lookback:i])
            y.append(data[i])
        
        X = np.array(X)
        y = np.array(y)
        
        # Reshape X to [samples, timesteps, features] for LSTM
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        return X, y
    
    def build_model(self) -> Sequential:
        """
        Build LSTM model architecture
        
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            # First LSTM layer with return sequences
            LSTM(units=self.units, 
                 return_sequences=True, 
                 input_shape=(self.lookback, 1)),
            Dropout(self.dropout),
            
            # Second LSTM layer
            LSTM(units=self.units // 2, 
                 return_sequences=False),
            Dropout(self.dropout),
            
            # Dense output layer
            Dense(units=1)
        ])
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        return model
    
    def fit(self, df: pd.DataFrame, 
            value_col: str = 'consumption_mw',
            validation_split: float = 0.2) -> 'LSTMForecaster':
        """
        Fit LSTM model to training data
        
        Args:
            df: DataFrame with datetime index and value column
            value_col: Name of the value column
            validation_split: Fraction of data to use for validation
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting LSTM model...")
        
        # Store for later use
        self.train_data = df.copy()
        self.value_col = value_col
        
        # Extract values
        data = df[value_col].values.reshape(-1, 1)
        
        # Normalize data
        logger.info("Normalizing data...")
        data_scaled = self.scaler.fit_transform(data)
        
        # Create sequences
        logger.info(f"Creating sequences with lookback={self.lookback}...")
        X, y = self.create_sequences(data_scaled.flatten())
        
        logger.info(f"Training samples: {len(X)}, Validation split: {validation_split}")
        
        # Build model
        logger.info("Building LSTM architecture...")
        self.model = self.build_model()
        
        # Early stopping callback
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        logger.info(f"Training LSTM for up to {self.epochs} epochs...")
        self.history = self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=0  # Suppress epoch-by-epoch output
        )
        
        self.is_fitted = True
        
        # Get final metrics
        final_loss = self.history.history['loss'][-1]
        final_val_loss = self.history.history['val_loss'][-1]
        
        logger.info(f"✅ LSTM model fitted successfully")
        logger.info(f"Final training loss: {final_loss:.4f}")
        logger.info(f"Final validation loss: {final_val_loss:.4f}")
        
        return self
    
    def predict(self, horizon: int = 30) -> pd.DataFrame:
        """
        Generate forecast for specified horizon
        
        Uses recursive prediction: predicts one step, adds to sequence, predicts next
        
        Args:
            horizon: Number of periods to forecast
            
        Returns:
            DataFrame with predictions and confidence intervals
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        logger.info(f"Generating {horizon}-period forecast (recursive)...")
        
        # Start with training data
        data = self.train_data[self.value_col].values.reshape(-1, 1)
        data_scaled = self.scaler.transform(data)
        
        # Get last lookback values
        last_sequence = data_scaled[-self.lookback:].flatten()
        
        predictions_scaled = []
        
        # Recursive prediction
        for _ in range(horizon):
            # Reshape for prediction
            X_pred = last_sequence.reshape(1, self.lookback, 1)
            
            # Predict next value
            pred_scaled = self.model.predict(X_pred, verbose=0)[0, 0]
            predictions_scaled.append(pred_scaled)
            
            # Update sequence
            last_sequence = np.append(last_sequence[1:], pred_scaled)
        
        # Inverse transform predictions
        predictions_scaled = np.array(predictions_scaled).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions_scaled).flatten()
        
        # Generate future dates
        last_date = self.train_data.index[-1]
        freq = pd.infer_freq(self.train_data.index)
        if freq is None:
            freq = 'H'
        
        future_dates = pd.date_range(
            start=last_date,
            periods=horizon + 1,
            freq=freq
        )[1:]
        
        # Simple confidence intervals (±2 std of training data)
        train_std = self.train_data[self.value_col].std()
        ci_width = 2 * train_std
        
        forecast_df = pd.DataFrame({
            'datetime': future_dates,
            'prediction': predictions,
            'lower_bound': predictions - ci_width,
            'upper_bound': predictions + ci_width
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
        
        # Combine train and test
        full_df = pd.concat([self.train_data, test_df])
        data = full_df[value_col].values.reshape(-1, 1)
        
        # Normalize
        data_scaled = self.scaler.transform(data)
        
        # Create sequences
        X, y = self.create_sequences(data_scaled.flatten())
        
        # Find test portion indices
        train_len = len(self.train_data)
        test_start_idx = train_len - self.lookback
        
        # Get test sequences
        if test_start_idx < len(X):
            X_test = X[test_start_idx:]
            y_test_scaled = y[test_start_idx:]
            
            # Predict
            y_pred_scaled = self.model.predict(X_test, verbose=0).flatten()
            
            # Inverse transform
            y_true = self.scaler.inverse_transform(
                y_test_scaled.reshape(-1, 1)
            ).flatten()
            y_pred = self.scaler.inverse_transform(
                y_pred_scaled.reshape(-1, 1)
            ).flatten()
            
            logger.info(f"Evaluation on {len(y_true)} test samples")
            logger.info(f"Mean actual: {np.mean(y_true):.2f}, Mean predicted: {np.mean(y_pred):.2f}")
            
            # Calculate metrics
            from forecasting.evaluation.metrics import calculate_metrics
            metrics = calculate_metrics(y_true, y_pred)
            
            return metrics
        else:
            logger.error("Insufficient test data for evaluation")
            raise ValueError("Test set too small for evaluation")
    
    def get_training_history(self) -> pd.DataFrame:
        """
        Get training history (loss curves)
        
        Returns:
            DataFrame with training history
        """
        if self.history is None:
            raise ValueError("Model must be trained first")
        
        return pd.DataFrame(self.history.history)
    
    def save_model(self, filename: Optional[str] = None) -> Path:
        """
        Save trained model to disk
        
        Args:
            filename: Optional filename (default: lstm_model.keras)
            
        Returns:
            Path to saved model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        if filename is None:
            filename = "lstm_model"
        
        # Save Keras model
        model_path = MODELS_DIR / f"{filename}.keras"
        self.model.save(model_path)
        
        # Save scaler and metadata
        metadata = {
            'scaler': self.scaler,
            'lookback': self.lookback,
            'units': self.units,
            'value_col': self.value_col,
            'train_data': self.train_data
        }
        metadata_path = MODELS_DIR / f"{filename}_metadata.pkl"
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Metadata saved to: {metadata_path}")
        
        return model_path
    
    @classmethod
    def load_model(cls, filename: str) -> 'LSTMForecaster':
        """
        Load trained model from disk
        
        Args:
            filename: Filename without extension
            
        Returns:
            LSTMForecaster instance with loaded model
        """
        model_path = MODELS_DIR / f"{filename}.keras"
        metadata_path = MODELS_DIR / f"{filename}_metadata.pkl"
        
        # Load metadata
        metadata = joblib.load(metadata_path)
        
        # Create forecaster instance
        forecaster = cls(lookback=metadata['lookback'], units=metadata['units'])
        
        # Load model
        forecaster.model = keras.models.load_model(model_path)
        forecaster.scaler = metadata['scaler']
        forecaster.value_col = metadata['value_col']
        forecaster.train_data = metadata['train_data']
        forecaster.is_fitted = True
        
        logger.info(f"Model loaded from: {model_path}")
        
        return forecaster
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        return {
            'lookback': self.lookback,
            'units': self.units,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'dropout': self.dropout
        }
    
    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return f"LSTMForecaster({status}, lookback={self.lookback})"