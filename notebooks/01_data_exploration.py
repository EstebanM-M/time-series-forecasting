"""
Data Exploration Script
Run this to explore the PJM Energy dataset
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from forecasting.preprocessing.data_loader import DataLoader
from forecasting.preprocessing.cleaner import DataCleaner

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 6)

def explore_data():
    """Explore PJM Energy data"""
    
    print("="*80)
    print("PJM ENERGY DATA EXPLORATION")
    print("="*80)
    
    # Load data
    print("\n1. Loading data...")
    loader = DataLoader()
    df = loader.load_pjm_sample()
    
    print(f"   ✓ Loaded {len(df):,} records")
    print(f"   ✓ Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"   ✓ Columns: {list(df.columns)}")
    
    # Basic statistics
    print("\n2. Basic Statistics:")
    print(df['consumption_mw'].describe())
    
    # Check for missing values
    print("\n3. Missing Values:")
    missing = df.isnull().sum()
    print(missing)
    
    # Clean data
    print("\n4. Cleaning data...")
    cleaner = DataCleaner()
    df_clean = cleaner.clean(df, 'datetime', 'consumption_mw', freq='H')
    
    print(f"   ✓ Cleaned data: {len(df_clean):,} records")
    
    # Add time features
    df_features = cleaner.add_time_features(df_clean)
    
    # Analyze patterns
    print("\n5. Analyzing Patterns:")
    
    # Average by hour
    hourly_avg = df_features.groupby('hour')['consumption_mw'].mean()
    print(f"   ✓ Peak hour: {hourly_avg.idxmax()}:00 ({hourly_avg.max():.0f} MW)")
    print(f"   ✓ Low hour: {hourly_avg.idxmin()}:00 ({hourly_avg.min():.0f} MW)")
    
    # Average by day of week
    daily_avg = df_features.groupby('day_of_week')['consumption_mw'].mean()
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    print(f"   ✓ Peak day: {days[daily_avg.idxmax()]} ({daily_avg.max():.0f} MW)")
    print(f"   ✓ Low day: {days[daily_avg.idxmin()]} ({daily_avg.min():.0f} MW)")
    
    # Seasonal patterns
    monthly_avg = df_features.groupby('month')['consumption_mw'].mean()
    print(f"   ✓ Peak month: {monthly_avg.idxmax()} ({monthly_avg.max():.0f} MW)")
    print(f"   ✓ Low month: {monthly_avg.idxmin()} ({monthly_avg.min():.0f} MW)")
    
    print("\n6. Data Quality:")
    print(f"   ✓ No missing values: {df_clean['consumption_mw'].notna().all()}")
    print(f"   ✓ Min value: {df_clean['consumption_mw'].min():.0f} MW")
    print(f"   ✓ Max value: {df_clean['consumption_mw'].max():.0f} MW")
    print(f"   ✓ Mean: {df_clean['consumption_mw'].mean():.0f} MW")
    print(f"   ✓ Std: {df_clean['consumption_mw'].std():.0f} MW")
    
    print("\n" + "="*80)
    print("EXPLORATION COMPLETE")
    print("="*80)
    
    return df_clean, df_features


if __name__ == "__main__":
    df_clean, df_features = explore_data()
    
    print("\nDataFrame info:")
    print(df_clean.head())
    print("\nFeatures added:")
    print(df_features.head())