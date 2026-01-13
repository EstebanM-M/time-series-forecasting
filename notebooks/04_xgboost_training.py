"""
XGBoost Model Training and Evaluation Demo
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from forecasting.preprocessing.data_loader import DataLoader
from forecasting.preprocessing.cleaner import DataCleaner
from forecasting.models.xgboost_model import XGBoostForecaster
from forecasting.evaluation.metrics import calculate_metrics, print_metrics

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 6)


def train_xgboost_demo():
    """Demo of XGBoost model training and evaluation"""
    
    print("="*80)
    print(" XGBOOST MODEL TRAINING DEMO")
    print("="*80)
    
    # Step 1: Load and prepare data
    print("\n1️  Loading and preparing data...")
    loader = DataLoader()
    df = loader.load_pjm_sample()
    
    cleaner = DataCleaner()
    df_clean = cleaner.clean(df, 'datetime', 'consumption_mw', freq='H')
    
    print(f"   ✓ Loaded {len(df_clean)} hourly records")
    print(f"   ✓ Date range: {df_clean.index.min()} to {df_clean.index.max()}")
    
    # Step 2: Train/test split
    print("\n2️  Splitting data (80% train, 20% test)...")
    train_size = int(len(df_clean) * 0.8)
    
    train_df = df_clean[:train_size]
    test_df = df_clean[train_size:]
    
    print(f"   ✓ Training set: {len(train_df)} records")
    print(f"   ✓ Test set: {len(test_df)} records ({len(test_df)/24:.0f} days)")
    
    # Step 3: Initialize and train XGBoost
    print("\n3️  Training XGBoost model...")
    print("   ⏳ Creating features and training (10-20 seconds)...")
    
    forecaster = XGBoostForecaster(
        n_lags=24,  # Use last 24 hours
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
    )
    
    forecaster.fit(train_df, value_col='consumption_mw')
    
    print("   ✓ Model trained successfully!")
    
    # Step 4: Feature importance
    print("\n4️  Analyzing feature importance...")
    importance = forecaster.get_feature_importance()
    print("\n   Top 10 most important features:")
    print(importance.head(10).to_string(index=False))
    
    # Step 5: Evaluate model
    print("\n5️  Evaluating model performance...")
    
    metrics = forecaster.evaluate_on_test(test_df, value_col='consumption_mw')
    print_metrics(metrics, model_name="XGBoost")
    
    # Step 6: Forecast future
    print("\n6️  Generating 7-day future forecast...")
    
    # Train on all data
    forecaster_full = XGBoostForecaster(
        n_lags=24,
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=5,
    )
    forecaster_full.fit(df_clean, value_col='consumption_mw')
    
    # Forecast 7 days (168 hours)
    print("    Generating recursive forecast (this may take 10-20 seconds)...")
    future_forecast = forecaster_full.predict(horizon=168)
    
    print("   ✓ Future forecast generated")
    print("\n   First 24 hours of forecast:")
    print(future_forecast.head(24))
    
    # Step 7: Save model
    print("\n7  Saving trained model...")
    model_path = forecaster_full.save_model('xgboost_energy_model.pkl')
    print(f"   ✓ Model saved to: {model_path}")
    
    # Summary
    print("\n" + "="*80)
    print(" XGBOOST TRAINING COMPLETE")
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
    print(f"   • Feature importance available")
    
    print("="*80 + "\n")
    
    return forecaster_full, metrics, future_forecast


if __name__ == "__main__":
    forecaster, metrics, forecast = train_xgboost_demo()