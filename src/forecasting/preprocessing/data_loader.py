"""
Data loading and downloading utilities
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging
import warnings

from forecasting.config import RAW_DATA_DIR, SAMPLE_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Load and download time series datasets"""
    
    def __init__(self):
        self.raw_dir = RAW_DATA_DIR
        self.sample_dir = SAMPLE_DATA_DIR
        
        # Ensure directories exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)
    
    def download_from_kaggle(self) -> Optional[pd.DataFrame]:
        """
        Download real PJM Energy data from Kaggle
        
        Requires Kaggle API credentials to be configured:
        https://github.com/Kaggle/kaggle-api#api-credentials
        
        Returns:
            DataFrame if successful, None if failed
        """
        try:
            import kagglehub
            
            logger.info("Attempting to download from Kaggle...")
            
            # Download dataset
            path = kagglehub.dataset_download("robikscube/hourly-energy-consumption")
            logger.info(f"Dataset downloaded to: {path}")
            
            # Find the CSV file
            dataset_path = Path(path)
            csv_files = list(dataset_path.glob("*.csv"))
            
            if not csv_files:
                logger.warning("No CSV files found in downloaded dataset")
                return None
            
            # Load the main file (usually PJM_Load_hourly.csv)
            csv_file = csv_files[0]
            logger.info(f"Loading: {csv_file.name}")
            
            df = pd.read_csv(csv_file)
            
            # Standardize column names
            df.columns = ['datetime', 'consumption_mw']
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime').reset_index(drop=True)
            
            logger.info(f"✅ Successfully loaded {len(df):,} records from Kaggle")
            logger.info(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
            
            return df
            
        except ImportError:
            logger.warning("kagglehub not installed. Install with: pip install kagglehub")
            return None
        except Exception as e:
            logger.warning(f"Could not download from Kaggle: {str(e)}")
            logger.info("Tip: Configure Kaggle API credentials for real data")
            logger.info("See: https://github.com/Kaggle/kaggle-api#api-credentials")
            return None
    
    def generate_synthetic_energy_data(self, 
                                      start_date: str = '2021-01-01',
                                      periods: int = 17520,  # 2 years hourly
                                      freq: str = 'H') -> pd.DataFrame:
        """
        Generate realistic synthetic energy consumption data
        
        Simulates patterns observed in real energy data:
        - Daily seasonality (peaks during day, low at night)
        - Weekly seasonality (weekdays vs weekends)
        - Annual seasonality (higher in summer/winter)
        - Trend
        - Random noise and occasional spikes
        
        Args:
            start_date: Start date for time series
            periods: Number of periods
            freq: Frequency ('H' for hourly, 'D' for daily)
            
        Returns:
            DataFrame with datetime and consumption columns
        """
        logger.info(f"Generating synthetic energy data: {periods} periods...")
        
        # Create date range
        dates = pd.date_range(start=start_date, periods=periods, freq=freq)
        
        # Base load (MW)
        base_load = 25000
        
        # Components
        t = np.arange(periods)
        
        # 1. Annual trend (slight increase over time)
        trend = 500 * (t / periods)
        
        # 2. Annual seasonality (higher in summer and winter)
        annual_seasonality = 3000 * np.sin(2 * np.pi * t / (365 * 24)) + \
                           1500 * np.cos(2 * np.pi * t / (365 * 24))
        
        # 3. Weekly seasonality (lower on weekends)
        weekly_seasonality = -1000 * np.sin(2 * np.pi * t / (7 * 24))
        
        # 4. Daily seasonality (peak during day, low at night)
        daily_seasonality = 4000 * np.sin(2 * np.pi * t / 24 - np.pi/2)
        
        # 5. Random noise
        np.random.seed(42)  # For reproducibility
        noise = np.random.normal(0, 500, periods)
        
        # 6. Occasional spikes (simulating heat waves, cold snaps)
        spikes = np.zeros(periods)
        spike_indices = np.random.choice(periods, size=int(periods * 0.02), replace=False)
        spikes[spike_indices] = np.random.normal(3000, 1000, len(spike_indices))
        
        # Combine all components
        consumption = (base_load + trend + annual_seasonality + 
                      weekly_seasonality + daily_seasonality + 
                      noise + spikes)
        
        # Ensure no negative values
        consumption = np.maximum(consumption, 10000)
        
        # Create DataFrame
        df = pd.DataFrame({
            'datetime': dates,
            'consumption_mw': consumption
        })
        
        logger.info(f"✅ Generated {len(df)} records with realistic patterns")
        
        return df
    
    def download_pjm_energy_data(self, 
                                save_to_sample: bool = True,
                                use_kaggle: bool = True) -> pd.DataFrame:
        """
        Download or generate PJM Energy Consumption dataset
        
        Strategy:
        1. Try to download real data from Kaggle (if use_kaggle=True)
        2. Fallback to synthetic data if Kaggle fails
        
        Args:
            save_to_sample: If True, save to sample directory for app use
            use_kaggle: If True, attempt to download from Kaggle first
            
        Returns:
            DataFrame with datetime and consumption columns
        """
        df = None
        data_source = "unknown"
        
        # Try Kaggle first
        if use_kaggle:
            df = self.download_from_kaggle()
            if df is not None:
                data_source = "kaggle"
        
        # Fallback to synthetic data
        if df is None:
            logger.info("Using synthetic data as fallback...")
            df = self.generate_synthetic_energy_data(
                start_date='2022-01-01',
                periods=17520,  # 2 years hourly
                freq='H'
            )
            data_source = "synthetic"
        
        # Save to raw directory
        raw_path = self.raw_dir / f"pjm_energy_raw_{data_source}.csv"
        df.to_csv(raw_path, index=False)
        logger.info(f"Saved raw data to: {raw_path}")
        
        # Save to sample directory if requested
        if save_to_sample:
            # Take last 2 years if we have more data
            if len(df) > 17520:
                df_sample = df.tail(17520).copy()
            else:
                df_sample = df.copy()
            
            sample_path = self.sample_dir / "pjm_energy_sample.csv"
            df_sample.to_csv(sample_path, index=False)
            logger.info(f"Saved sample data to: {sample_path}")
            
            # Save metadata about data source
            metadata_path = self.sample_dir / "data_source.txt"
            with open(metadata_path, 'w') as f:
                f.write(f"Data source: {data_source}\n")
                f.write(f"Records: {len(df_sample)}\n")
                f.write(f"Date range: {df_sample['datetime'].min()} to {df_sample['datetime'].max()}\n")
            
            return df_sample
        
        return df
    
    def load_pjm_sample(self) -> pd.DataFrame:
        """Load PJM sample data from disk"""
        sample_path = self.sample_dir / "pjm_energy_sample.csv"
        
        if not sample_path.exists():
            logger.warning("Sample data not found. Downloading...")
            return self.download_pjm_energy_data(save_to_sample=True)
        
        df = pd.read_csv(sample_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Check if we have metadata about data source
        metadata_path = self.sample_dir / "data_source.txt"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                logger.info(f"Data info: {f.read().strip()}")
        
        return df
    
    def load_custom_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load custom CSV file
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame
        """
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} records from {filepath}")
        return df


def download_sample_data(use_kaggle: bool = True):
    """
    Convenience function to download sample data
    
    Args:
        use_kaggle: If True, try to download from Kaggle first
    """
    loader = DataLoader()
    df = loader.download_pjm_energy_data(save_to_sample=True, use_kaggle=use_kaggle)
    
    print("\n" + "="*70)
    print("✅ SAMPLE DATA READY")
    print("="*70)
    print(f"Total records: {len(df):,}")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Columns: {list(df.columns)}")
    print(f"Consumption range: {df['consumption_mw'].min():.0f} - {df['consumption_mw'].max():.0f} MW")
    print("\nFirst few rows:")
    print(df.head(10))
    print("\nBasic statistics:")
    print(df['consumption_mw'].describe())
    print("\nData saved to:")
    print(f"  - Raw: data/raw/pjm_energy_raw_*.csv")
    print(f"  - Sample: data/sample/pjm_energy_sample.csv")
    
    # Check data source
    metadata_path = Path("data/sample/data_source.txt")
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            print("\n" + f.read())
    
    print("\n" + "="*70)
    print("Note: Configure Kaggle API for real data, or use synthetic fallback")
    print("Kaggle setup: https://github.com/Kaggle/kaggle-api#api-credentials")
    print("="*70)
    
    return df


if __name__ == "__main__":
    download_sample_data()