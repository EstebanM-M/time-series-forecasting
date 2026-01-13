"""
Evaluation metrics for time series forecasting
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import logging

from forecasting.config import BUSINESS_METRICS_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error
    
    MAPE = (100/n) * Σ|y_true - y_pred| / |y_true|
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        MAPE percentage
    """
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error
    
    SMAPE = (100/n) * Σ|y_true - y_pred| / (|y_true| + |y_pred|)
    
    More stable than MAPE for values close to zero
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        SMAPE percentage
    """
    denominator = np.abs(y_true) + np.abs(y_pred)
    # Avoid division by zero
    mask = denominator != 0
    return np.mean(2.0 * np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_forecast_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate forecast bias (tendency to over or under-predict)
    
    Bias = mean(y_pred - y_true) / mean(y_true) * 100
    
    Positive: Over-forecasting
    Negative: Under-forecasting
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        Bias percentage
    """
    return (np.mean(y_pred - y_true) / np.mean(y_true)) * 100


def calculate_business_metrics(y_true: np.ndarray, 
                              y_pred: np.ndarray,
                              config: Optional[Dict] = None) -> Dict[str, float]:
    """
    Calculate business-focused metrics
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        config: Business metrics configuration
        
    Returns:
        Dictionary with business metrics
    """
    if config is None:
        config = BUSINESS_METRICS_CONFIG
    
    # Forecast errors
    errors = y_pred - y_true
    
    # Over-forecast (positive errors)
    over_forecast = errors[errors > 0]
    over_forecast_cost = len(over_forecast) * config['cost_per_unit_overforecast']
    
    # Under-forecast (negative errors)
    under_forecast = errors[errors < 0]
    under_forecast_cost = abs(len(under_forecast)) * config['cost_per_unit_underforecast']
    
    # Total cost
    total_cost = over_forecast_cost + under_forecast_cost
    
    # Baseline comparison
    if config['baseline_method'] == 'naive':
        # Naive forecast: use last value
        baseline_pred = np.full_like(y_true, y_true[0])
    elif config['baseline_method'] == 'seasonal_naive':
        # Use same value from same period last cycle
        baseline_pred = np.roll(y_true, 1)
    else:
        # Historical average
        baseline_pred = np.full_like(y_true, np.mean(y_true))
    
    baseline_mape = calculate_mape(y_true, baseline_pred)
    model_mape = calculate_mape(y_true, y_pred)
    
    improvement = ((baseline_mape - model_mape) / baseline_mape) * 100
    
    return {
        'over_forecast_count': len(over_forecast),
        'under_forecast_count': len(under_forecast),
        'over_forecast_cost': over_forecast_cost,
        'under_forecast_cost': under_forecast_cost,
        'total_forecast_cost': total_cost,
        'baseline_mape': baseline_mape,
        'model_mape': model_mape,
        'improvement_vs_baseline': improvement,
    }


def calculate_metrics(y_true: np.ndarray, 
                     y_pred: np.ndarray,
                     include_business: bool = True) -> Dict[str, float]:
    """
    Calculate comprehensive set of evaluation metrics
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        include_business: Whether to include business metrics
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {
        # Technical metrics
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': calculate_rmse(y_true, y_pred),
        'mape': calculate_mape(y_true, y_pred),
        'smape': calculate_smape(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'forecast_bias': calculate_forecast_bias(y_true, y_pred),
        
        # Additional stats
        'mean_actual': np.mean(y_true),
        'mean_predicted': np.mean(y_pred),
        'std_actual': np.std(y_true),
        'std_predicted': np.std(y_pred),
    }
    
    # Add business metrics if requested
    if include_business:
        business_metrics = calculate_business_metrics(y_true, y_pred)
        metrics.update(business_metrics)
    
    return metrics


def print_metrics(metrics: Dict[str, float], 
                 model_name: str = "Model") -> None:
    """
    Pretty print evaluation metrics
    
    Args:
        metrics: Dictionary of metrics
        model_name: Name of the model
    """
    print("\n" + "="*70)
    print(f"📊 {model_name} - EVALUATION METRICS")
    print("="*70)
    
    print("\n🔧 Technical Metrics:")
    print(f"  MAE (Mean Absolute Error):     {metrics['mae']:.2f}")
    print(f"  RMSE (Root Mean Squared Error): {metrics['rmse']:.2f}")
    print(f"  MAPE (Mean Abs % Error):        {metrics['mape']:.2f}%")
    print(f"  SMAPE (Symmetric MAPE):         {metrics['smape']:.2f}%")
    print(f"  R² Score:                        {metrics['r2']:.4f}")
    print(f"  Forecast Bias:                   {metrics['forecast_bias']:.2f}%")
    
    if 'improvement_vs_baseline' in metrics:
        print("\n💼 Business Metrics:")
        print(f"  Over-forecast instances:  {metrics['over_forecast_count']}")
        print(f"  Under-forecast instances: {metrics['under_forecast_count']}")
        print(f"  Total forecast cost:      ${metrics['total_forecast_cost']:,.2f}")
        print(f"  Baseline MAPE:            {metrics['baseline_mape']:.2f}%")
        print(f"  Model MAPE:               {metrics['model_mape']:.2f}%")
        print(f"  Improvement:              {metrics['improvement_vs_baseline']:.1f}%")
    
    print("\n📈 Data Statistics:")
    print(f"  Mean Actual:     {metrics['mean_actual']:.2f}")
    print(f"  Mean Predicted:  {metrics['mean_predicted']:.2f}")
    print(f"  Std Actual:      {metrics['std_actual']:.2f}")
    print(f"  Std Predicted:   {metrics['std_predicted']:.2f}")
    
    print("="*70 + "\n")


def compare_models(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Compare multiple models side by side
    
    Args:
        results: Dictionary mapping model names to their metrics
        
    Returns:
        DataFrame with model comparison
    """
    comparison = pd.DataFrame(results).T
    
    # Select key metrics for comparison
    key_metrics = ['mae', 'rmse', 'mape', 'smape', 'r2', 'forecast_bias']
    
    if all(key in comparison.columns for key in key_metrics):
        comparison = comparison[key_metrics]
    
    # Sort by MAPE (lower is better)
    comparison = comparison.sort_values('mape')
    
    return comparison