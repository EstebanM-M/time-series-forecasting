"""
LSTM Model Training and Evaluation Demo
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from forecasting.preprocessing.data_loader import DataLoader
from forecasting.preprocessing.cleaner import DataCleaner
from forecasting.models.lstm_model import LSTMForecaster
from forecasting.evaluation.metrics import calculate_metrics, print_metrics

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 6)


def train_lstm_demo():
    """Demo of LSTM model training and evaluation"""
    
    print("="*80)
    print("🧠 LSTM MODEL TRAINING DEMO")
    print("="*80)
    
    # Step 1: Load and prepare data
    print("\n1️⃣  Loading and preparing data...")
    loader = DataLoader()
    df = loader.load_pjm_sample()
    
    cleaner = DataCleaner()
    df_clean = cleaner.clean(df, 'datetime', 'consumption_mw', freq='H')
    
    print(f"   ✓ Loaded {len(df_clean)} hourly records")
    print(f"   ✓ Date range: {df_clean.index.min()} to {df_clean.index.max()}")
    
    # Step 2: Train/test split
    print("\n2️⃣  Splitting data (80% train, 20% test)...")
    train_size = int(len(df_clean) * 0.8)
    
    train_df = df_clean[:train_size]
    test_df = df_clean[train_size:]
    
    print(f"   ✓ Training set: {len(train_df)} records")
    print(f"   ✓ Test set: {len(test_df)} records ({len(test_df)/24:.0f} days)")
    
    # Step 3: Initialize and train LSTM
    print("\n3️⃣  Training LSTM model...")
    print("   ⏳ This may take 1-2 minutes (neural network training)...")
    
    forecaster = LSTMForecaster(
        lookback=30,  # Use last 30 hours
        units=50,
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
        dropout=0.2
    )
    
    forecaster.fit(train_df, value_col='consumption_mw', validation_split=0.2)
    
    print("   ✓ Model trained successfully!")
    
    # Step 4: Training history
    print("\n4️⃣  Analyzing training history...")
    history = forecaster.get_training_history()
    final_epoch = len(history)
    final_loss = history['loss'].iloc[-1]
    final_val_loss = history['val_loss'].iloc[-1]
    
    print(f"   ✓ Trained for {final_epoch} epochs")
    print(f"   ✓ Final training loss: {final_loss:.4f}")
    print(f"   ✓ Final validation loss: {final_val_loss:.4f}")
    
    # Step 5: Evaluate model
    print("\n5️⃣  Evaluating model performance...")
    
    metrics = forecaster.evaluate_on_test(test_df, value_col='consumption_mw')
    print_metrics(metrics, model_name="LSTM")
    
    # Step 6: Forecast future
    print("\n6️⃣  Generating 7-day future forecast...")
    
    # Train on all data
    forecaster_full = LSTMForecaster(
        lookback=30,
        units=50,
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
        dropout=0.2
    )
    
    print("   ⏳ Training on full dataset (1-2 minutes)...")
    forecaster_full.fit(df_clean, value_col='consumption_mw', validation_split=0.2)
    
    # Forecast 7 days (168 hours)
    print("   ⏳ Generating recursive forecast...")
    future_forecast = forecaster_full.predict(horizon=168)
    
    print("   ✓ Future forecast generated")
    print("\n   First 24 hours of forecast:")
    print(future_forecast.head(24))
    
    # Step 7: Save model
    print("\n7️⃣  Saving trained model...")
    model_path = forecaster_full.save_model('lstm_energy_model')
    print(f"   ✓ Model saved to: {model_path}")
    
    # Summary
    print("\n" + "="*80)
    print(" LSTM TRAINING COMPLETE")
    print("="*80)
    print(f"\n Key Results:")
    print(f"   • Test MAPE: {metrics['mape']:.2f}%")
    print(f"   • Test RMSE: {metrics['rmse']:.2f} MW")
    print(f"   • R² Score: {metrics['r2']:.4f}")
    print(f"   • Forecast Bias: {metrics['forecast_bias']:.2f}%")
    
    if 'improvement_vs_baseline' in metrics:
        print(f"   • Improvement vs Baseline: {metrics['improvement_vs_baseline']:.1f}%")
    
    print("\n Outputs:")
    print(f"   • Trained model: {model_path}")
    print(f"   • 7-day forecast: {len(future_forecast)} hourly predictions")
    print(f"   • Training history available")
    
    print("="*80 + "\n")
    
    return forecaster_full, metrics, future_forecast


if __name__ == "__main__":
    forecaster, metrics, forecast = train_lstm_demo()