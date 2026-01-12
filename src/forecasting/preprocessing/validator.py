"""
Data validation utilities for time series data
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
from datetime import datetime
import logging

from forecasting.config import VALIDATION_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesValidator:
    """Validates uploaded time series data"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or VALIDATION_CONFIG
    
    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[str], Dict]:
        """
        Validate time series data
        
        Args:
            df: Input dataframe
            
        Returns:
            (is_valid, error_messages, metadata)
        """
        errors = []
        metadata = {}
        
        # Check 1: Minimum data points
        if len(df) < self.config["min_data_points"]:
            errors.append(
                f"Insufficient data: {len(df)} rows. "
                f"Minimum required: {self.config['min_data_points']}"
            )
        
        # Check 2: Maximum data points (for free tier)
        if len(df) > self.config["max_data_points"]:
            errors.append(
                f"Too much data: {len(df)} rows. "
                f"Maximum allowed: {self.config['max_data_points']}"
            )
        
        # Check 3: Detect date and value columns
        date_col, value_col, col_errors = self._detect_columns(df)
        errors.extend(col_errors)
        
        if date_col and value_col:
            metadata["date_column"] = date_col
            metadata["value_column"] = value_col
            
            # Check 4: Data types
            type_errors = self._check_data_types(df, date_col, value_col)
            errors.extend(type_errors)
            
            # Check 5: Missing values
            missing_info, missing_errors = self._check_missing_values(
                df, value_col
            )
            errors.extend(missing_errors)
            metadata.update(missing_info)
            
            # Check 6: Frequency detection
            freq_info, freq_errors = self._detect_frequency(df, date_col)
            errors.extend(freq_errors)
            metadata.update(freq_info)
            
            # Check 7: Outliers detection
            outlier_info = self._detect_outliers(df, value_col)
            metadata.update(outlier_info)
        
        is_valid = len(errors) == 0
        return is_valid, errors, metadata
    
    def _detect_columns(self, df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], List[str]]:
        """Detect date and value columns"""
        errors = []
        date_col = None
        value_col = None
        
        # Try to find date column
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        if len(datetime_cols) > 0:
            date_col = datetime_cols[0]
        else:
            # Try to parse columns as datetime
            for col in df.columns:
                try:
                    pd.to_datetime(df[col], errors='raise')
                    date_col = col
                    break
                except:
                    continue
        
        if not date_col:
            errors.append(
                "No date column found. Please ensure your data has a date/timestamp column."
            )
        
        # Try to find numeric value column
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 0:
            # Exclude the date column if it was detected as numeric
            numeric_cols = [col for col in numeric_cols if col != date_col]
            if len(numeric_cols) > 0:
                value_col = numeric_cols[0]  # Take first numeric column
        
        if not value_col:
            errors.append(
                "No numeric value column found. Please ensure your data has a numeric column."
            )
        
        return date_col, value_col, errors
    
    def _check_data_types(self, df: pd.DataFrame, date_col: str, 
                         value_col: str) -> List[str]:
        """Check if columns have correct data types"""
        errors = []
        
        # Check date column
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            try:
                pd.to_datetime(df[date_col], errors='raise')
            except:
                errors.append(
                    f"Column '{date_col}' cannot be converted to datetime."
                )
        
        # Check value column
        if not pd.api.types.is_numeric_dtype(df[value_col]):
            errors.append(
                f"Column '{value_col}' must be numeric."
            )
        
        return errors
    
    def _check_missing_values(self, df: pd.DataFrame, 
                             value_col: str) -> Tuple[Dict, List[str]]:
        """Check for missing values"""
        errors = []
        info = {}
        
        missing_count = df[value_col].isna().sum()
        missing_pct = (missing_count / len(df)) * 100
        
        info["missing_count"] = int(missing_count)
        info["missing_percentage"] = float(missing_pct)
        
        if missing_pct > self.config["max_missing_percentage"]:
            errors.append(
                f"Too many missing values: {missing_pct:.1f}%. "
                f"Maximum allowed: {self.config['max_missing_percentage']}%"
            )
        
        return info, errors
    
    def _detect_frequency(self, df: pd.DataFrame, 
                         date_col: str) -> Tuple[Dict, List[str]]:
        """Detect time series frequency"""
        errors = []
        info = {}
        
        try:
            # Convert to datetime if not already
            dates = pd.to_datetime(df[date_col])
            dates = dates.sort_values()
            
            # Calculate differences
            diffs = dates.diff().dropna()
            
            # Most common difference
            most_common_diff = diffs.mode()[0] if len(diffs) > 0 else None
            
            if most_common_diff:
                # Infer frequency
                if most_common_diff <= pd.Timedelta(hours=1):
                    freq = "Hourly"
                    freq_code = "H"
                elif most_common_diff <= pd.Timedelta(days=1):
                    freq = "Daily"
                    freq_code = "D"
                elif most_common_diff <= pd.Timedelta(days=7):
                    freq = "Weekly"
                    freq_code = "W"
                elif most_common_diff <= pd.Timedelta(days=31):
                    freq = "Monthly"
                    freq_code = "M"
                else:
                    freq = "Other"
                    freq_code = None
                
                info["frequency"] = freq
                info["frequency_code"] = freq_code
                info["date_range_start"] = str(dates.min())
                info["date_range_end"] = str(dates.max())
                info["total_periods"] = len(dates)
                
                if freq_code not in self.config["allowed_frequencies"]:
                    logger.warning(
                        f"Unusual frequency detected: {freq}. "
                        "Proceeding with caution."
                    )
            else:
                errors.append("Could not determine time series frequency.")
        
        except Exception as e:
            errors.append(f"Error detecting frequency: {str(e)}")
        
        return info, errors
    
    def _detect_outliers(self, df: pd.DataFrame, value_col: str) -> Dict:
        """Detect outliers using IQR method"""
        info = {}
        
        try:
            Q1 = df[value_col].quantile(0.25)
            Q3 = df[value_col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[
                (df[value_col] < lower_bound) | (df[value_col] > upper_bound)
            ]
            
            info["outlier_count"] = len(outliers)
            info["outlier_percentage"] = (len(outliers) / len(df)) * 100
            
        except Exception as e:
            logger.warning(f"Could not detect outliers: {str(e)}")
        
        return info


def validate_uploaded_data(df: pd.DataFrame) -> Tuple[bool, List[str], Dict]:
    """
    Convenience function to validate uploaded data
    
    Args:
        df: Uploaded dataframe
        
    Returns:
        (is_valid, error_messages, metadata)
    """
    validator = TimeSeriesValidator()
    return validator.validate(df)