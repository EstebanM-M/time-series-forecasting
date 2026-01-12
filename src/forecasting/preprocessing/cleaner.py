"""
Data cleaning and preprocessing utilities
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCleaner:
    """Clean and preprocess time series data"""
    
    def __init__(self):
        pass
    
    def clean(self, df: pd.DataFrame, 
              date_col: str, 
              value_col: str,
              freq: Optional[str] = None) -> pd.DataFrame:
        """
        Clean time series data
        
        Steps:
        1. Remove duplicates
        2. Handle missing values
        3. Remove outliers (optional)
        4. Set datetime index
        5. Ensure regular frequency
        
        Args:
            df: Input dataframe
            date_col: Name of date column
            value_col: Name of value column
            freq: Target frequency (e.g., 'D', 'H', 'W')
            
        Returns:
            Cleaned dataframe
        """
        logger.info(f"Cleaning data: {len(df)} records")
        
        df = df.copy()
        
        # Step 1: Convert datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Step 2: Remove duplicates
        before_count = len(df)
        df = df.drop_duplicates(subset=[date_col])
        duplicates_removed = before_count - len(df)
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate records")
        
        # Step 3: Sort by date
        df = df.sort_values(date_col).reset_index(drop=True)
        
        # Step 4: Handle missing values in value column
        missing_count = df[value_col].isna().sum()
        if missing_count > 0:
            logger.info(f"Handling {missing_count} missing values")
            # Forward fill then backward fill
            df[value_col] = df[value_col].fillna(method='ffill').fillna(method='bfill')
        
        # Step 5: Set datetime index
        df = df.set_index(date_col)
        
        # Step 6: Ensure regular frequency if specified
        if freq:
            df = self._ensure_regular_frequency(df, value_col, freq)
        
        logger.info(f"Cleaning complete: {len(df)} records")
        
        return df
    
    def _ensure_regular_frequency(self, df: pd.DataFrame, 
                                  value_col: str, 
                                  freq: str) -> pd.DataFrame:
        """
        Ensure dataframe has regular frequency by resampling
        
        Args:
            df: DataFrame with datetime index
            value_col: Name of value column
            freq: Target frequency
            
        Returns:
            Resampled dataframe
        """
        # Create full date range
        full_range = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq=freq
        )
        
        # Reindex to full range
        df = df.reindex(full_range)
        
        # Interpolate missing values
        df[value_col] = df[value_col].interpolate(method='time')
        
        return df
    
    def remove_outliers(self, df: pd.DataFrame, 
                       value_col: str,
                       method: str = 'iqr',
                       threshold: float = 1.5) -> pd.DataFrame:
        """
        Remove outliers from data
        
        Args:
            df: Input dataframe
            value_col: Name of value column
            method: 'iqr' or 'zscore'
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers removed
        """
        df = df.copy()
        
        if method == 'iqr':
            Q1 = df[value_col].quantile(0.25)
            Q3 = df[value_col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            mask = (df[value_col] >= lower_bound) & (df[value_col] <= upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((df[value_col] - df[value_col].mean()) / df[value_col].std())
            mask = z_scores < threshold
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        outliers_removed = (~mask).sum()
        logger.info(f"Removed {outliers_removed} outliers using {method} method")
        
        return df[mask]
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features for modeling
        
        Features added:
        - hour, day, month, year
        - day_of_week, day_of_year
        - is_weekend
        - quarter
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            DataFrame with additional features
        """
        df = df.copy()
        
        df['hour'] = df.index.hour
        df['day'] = df.index.day
        df['month'] = df.index.month
        df['year'] = df.index.year
        df['day_of_week'] = df.index.dayofweek
        df['day_of_year'] = df.index.dayofyear
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        df['quarter'] = df.index.quarter
        
        logger.info("Added time-based features")
        
        return df
    
    def get_cleaning_summary(self, df_before: pd.DataFrame, 
                            df_after: pd.DataFrame) -> Dict:
        """
        Get summary of cleaning operations
        
        Args:
            df_before: DataFrame before cleaning
            df_after: DataFrame after cleaning
            
        Returns:
            Dictionary with cleaning summary
        """
        summary = {
            'records_before': len(df_before),
            'records_after': len(df_after),
            'records_removed': len(df_before) - len(df_after),
            'date_range_start': str(df_after.index.min()),
            'date_range_end': str(df_after.index.max()),
            'missing_values_before': df_before.isna().sum().sum(),
            'missing_values_after': df_after.isna().sum().sum(),
        }
        
        return summary


def clean_dataframe(df: pd.DataFrame, 
                   date_col: str = 'datetime',
                   value_col: str = 'value',
                   freq: Optional[str] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to clean dataframe
    
    Args:
        df: Input dataframe
        date_col: Name of date column
        value_col: Name of value column
        freq: Target frequency
        
    Returns:
        (cleaned_df, summary)
    """
    cleaner = DataCleaner()
    
    df_before = df.copy()
    df_clean = cleaner.clean(df, date_col, value_col, freq)
    summary = cleaner.get_cleaning_summary(df_before, df_clean)
    
    return df_clean, summary