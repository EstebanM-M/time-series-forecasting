"""
Prophet Model Training and Evaluation Demo
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from forecasting.preprocessing.data_loader import DataLoader
from forecasting.preprocessing.cleaner import DataCleaner
from forecasting.models.prophet_model import ProphetForecaster
from forecasting.evaluation.metrics import calculate_metrics, print_metrics

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 6)


def train_prophet_demo():
    """Demo of Prophet model training and evaluation"""
    
    print("="*80)
    print("🔮 PROPHET MODEL TRAINING DEMO")
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
    
    # Step 3: Initialize and train Prophet
    print("\n3️⃣  Training Prophet model...")
    print("   ⏳ This may take 15-30 seconds...")
    
    forecaster = ProphetForecaster(
        seasonality_mode='additive',
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        changepoint_prior_scale=0.05,
    )
    
    forecaster.fit(train_df, value_col='consumption_mw')
    
    print("   ✓ Model trained successfully!")
    
    # Step 4: Generate predictions on test set
    print("\n4️⃣  Generating predictions on test set...")
    
    test_horizon = len(test_df)
    forecast = forecaster.predict(horizon=test_horizon, freq='H')
    
    # Get predictions for test period
    test_predictions = forecast.tail(test_horizon)
    
    print(f"   ✓ Generated {test_horizon} predictions")
    
    # Step 5: Evaluate model
    print("\n5️⃣  Evaluating model performance...")
    
    metrics = forecaster.evaluate_on_test(test_df, value_col='consumption_mw')
    print_metrics(metrics, model_name="Prophet")
    
    # Step 6: Forecast future
    print("\n6️⃣  Generating 7-day future forecast...")
    
    # Train on all data
    forecaster_full = ProphetForecaster(
        seasonality_mode='additive',
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
    )
    forecaster_full.fit(df_clean, value_col='consumption_mw')
    
    # Forecast 7 days (168 hours)
    future_forecast = forecaster_full.predict(horizon=168, freq='H')
    future_summary = forecaster_full.get_forecast_summary(horizon=168)
    
    print("   ✓ Future forecast generated")
    print("\n   First 24 hours of forecast:")
    print(future_summary.head(24))
    
    # Step 7: Analyze components
    print("\n7️⃣  Analyzing forecast components...")
    components = forecaster_full.get_components()
    
    print("   ✓ Trend component extracted")
    print("   ✓ Weekly seasonality extracted")
    print("   ✓ Yearly seasonality extracted")
    
    # Step 8: Save model
    print("\n8️⃣  Saving trained model...")
    model_path = forecaster_full.save_model('prophet_energy_model.pkl')
    print(f"   ✓ Model saved to: {model_path}")
    
    # Summary
    print("\n" + "="*80)
    print("✅ PROPHET TRAINING COMPLETE")
    print("="*80)
    print(f"\n📊 Key Results:")
    print(f"   • Test MAPE: {metrics['mape']:.2f}%")
    print(f"   • Test RMSE: {metrics['rmse']:.2f} MW")
    print(f"   • R² Score: {metrics['r2']:.4f}")
    print(f"   • Forecast Bias: {metrics['forecast_bias']:.2f}%")
    
    if 'improvement_vs_baseline' in metrics:
        print(f"   • Improvement vs Baseline: {metrics['improvement_vs_baseline']:.1f}%")
    
    print("\n💾 Outputs:")
    print(f"   • Trained model: {model_path}")
    print(f"   • 7-day forecast: {len(future_summary)} hourly predictions")
    
    print("="*80 + "\n")
    
    return forecaster_full, metrics, future_summary


if __name__ == "__main__":
    forecaster, metrics, forecast = train_prophet_demo()