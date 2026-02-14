import pandas as pd
import numpy as np

def preprocess_data(df):
    """
    Cleans and preprocesses the AQI data.
    - Handles missing values (interpolation/backfill)
    - Detects and handles outliers
    - Feature engineering
    """
    # 1. Handle Missing Values
    # Forward fill then backward fill for continuous time series data
    df = df.ffill().bfill()
    
    # 2. Outlier Handling (Clipping to 1st and 99th percentiles)
    # This prevents extreme measurement errors from skewing the model
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df[col] = df[col].clip(lower, upper)

    # 3. Feature Engineering
    # Date components
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    
    # Rolling Averages (Critical for AQI trends)
    # 24-hour moving average is a standard AQI metric
    for col in ['pm10', 'pm2_5', 'us_aqi']:
        df[f'{col}_rolling_24h'] = df[col].rolling(window=24, min_periods=1).mean()
        df[f'{col}_rolling_6h'] = df[col].rolling(window=6, min_periods=1).mean()
    
    # Lag features
    # What was the AQI 24 hours ago? 48 hours ago?
    df['lag_24h_aqi'] = df['us_aqi'].shift(24)
    df['lag_48h_aqi'] = df['us_aqi'].shift(48)
    
    # Drop rows with NaN created by lag/rolling
    df = df.dropna()
    
    return df


