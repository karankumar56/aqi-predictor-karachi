import pandas as pd
import numpy as np

def preprocess_data(df, is_training=True):
    """
    Cleans and preprocesses the AQI data.
    - Handles missing values (interpolation/backfill)
    - Feature engineering
    """
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # 1. Handle Missing Values
    # Forward fill then backward fill for continuous time series data
    df = df.ffill().bfill()
    
    # 2. Outlier Handling (Clipping to 1st and 99th percentiles) only during training
    # This keeps inference deterministic and based on actual recent data
    if is_training:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['date']:
                lower = df[col].quantile(0.01)
                upper = df[col].quantile(0.99)
                df[col] = df[col].clip(lower, upper)

    # 3. Feature Engineering
    # Date components (Cyclical encoding would be better, but start with simple)
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    
    # Rolling Averages (Critical for AQI trends)
    # Using a larger min_periods to ensure we have enough history
    for col in ['pm10', 'pm2_5', 'us_aqi']:
        df[f'{col}_rolling_24h'] = df[col].rolling(window=24, min_periods=1).mean()
        df[f'{col}_rolling_6h'] = df[col].rolling(window=6, min_periods=1).mean()
    
    # Lag features
    # These are the primary drivers for a persistence-based model
    df['lag_1h_aqi'] = df['us_aqi'].shift(1)
    df['lag_2h_aqi'] = df['us_aqi'].shift(2)
    df['lag_24h_aqi'] = df['us_aqi'].shift(24)
    
    # Note: We NO LONGER dropna() here. 
    # The calling function should handle NAs (e.g., during training) 
    # while the forecasting loop handles them by providing enough history.
    
    return df


