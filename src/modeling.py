import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import pickle
import os
from database import AQIDatabase # Assumes database.py is in the same folder or pythonpath
from preprocessing import preprocess_data

def train_and_evaluate():
    """
    Trains multiple models and selects the best one based on RMSE.
    """
    print("Fetching data from MongoDB...")
    db = AQIDatabase()
    df = db.fetch_data()
    
    if df.empty:
        print("No data found in database. Run data_ingestion.py first.")
        return

    print(f"Data fetched: {len(df)} records.")
    
    # Preprocess
    df = preprocess_data(df, is_training=True)
    
    # Shift target logic: 
    # Current row features -> Next hour AQI
    df['target'] = df['us_aqi'].shift(-1)
    
    # Drop rows with NaN (early history for lags/rolling, and the last row for target)
    df = df.dropna()
    print(f"Data after preprocessing and alignment: {len(df)} records.")
    
    # Feature selection (focus on lags and rolling averages)
    features = [c for c in df.columns if 'lag' in c or 'rolling' in c or c in ['hour', 'day_of_week', 'month']]
    print(f"Features used: {features}")
    
    X = df[features]
    y = df['target']
    
    # Time-based train/test split (last 20% for testing)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42),
        "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    }
    
    best_model = None
    best_score = float('inf')
    best_name = ""
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        try:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            print(f"{name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
            results[name] = rmse
            
            if rmse < best_score:
                best_score = rmse
                best_model = model
                best_name = name
        except Exception as e:
            print(f"Failed to train {name}: {e}")
            
    if best_model is None:
        print("\n❌ Error: No models were successfully trained. Check the logs above for specific errors.")
        import sys
        sys.exit(1)

    print(f"\nBest model: {best_name} with RMSE: {best_score:.4f}")
    
    # Save best model
    with open('model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    # Also save the list of features to ensure consistency during inference
    with open('features.pkl', 'wb') as f:
        pickle.dump(features, f)
        
    print("Model and features saved.")
    return best_model, features

def predict_next_72_hours(model, features, recent_data):
    """
    Generates a 72-hour AQI forecast using recursive multi-step forecasting.
    Maintains a history buffer and applies robust clipping to prevent drift/explosion.
    """
    predictions = []
    # We need enough history for the longest lag window (24h)
    history_df = recent_data.tail(100).copy()
    
    # Estimate a reasonable upper bound based on recent history + safety margin
    last_known_aqi = history_df['us_aqi'].iloc[-1]
    max_physical_aqi = 500 # Absolute maximum based on EPA scales
    
    for i in range(72):
        # 1. Preprocess: Generate features based on current history
        df_processed = preprocess_data(history_df.copy(), is_training=False)
        
        # 2. Get Input: Use the VERY LAST row to predict the next step
        input_row = df_processed.iloc[-1:]
        X_input = input_row[features]
        
        # 3. Predict & Clip
        # We apply two-stage clipping:
        # - Hard floor at 0 (physical limit)
        # - Hard ceiling at 500 (physical limit)
        # - Smoothing: Prevent massive jumps between steps (max 20% change per hour)
        raw_pred = model.predict(X_input)[0]
        
        # Apply smoothing relative to previous value in history
        prev_aqi = history_df['us_aqi'].iloc[-1]
        
        # Allow more flexibility for the first step, then dampen the feedback loop
        dampening = 0.15 # Max 15% change per step to prevent explosion
        lower_bound = prev_aqi * (1 - dampening)
        upper_bound = prev_aqi * (1 + dampening)
        
        pred_aqi = np.clip(raw_pred, lower_bound, upper_bound)
        pred_aqi = np.clip(pred_aqi, 0, max_physical_aqi)
        
        # 4. Update History
        last_date = history_df['date'].iloc[-1]
        next_date = last_date + pd.Timedelta(hours=1)
        
        # Create a new row with the predicted AQI
        new_row = pd.DataFrame({'date': [next_date], 'us_aqi': [pred_aqi]})
        
        # Carry forward external/weather features
        last_known = history_df.iloc[-1].to_dict()
        for col, val in last_known.items():
            if col not in ['date', 'us_aqi', 'target'] and col not in new_row.columns and 'lag' not in col and 'rolling' not in col:
                 new_row[col] = val
                 
        history_df = pd.concat([history_df, new_row], ignore_index=True)
        
        predictions.append({'date': next_date, 'predicted_aqi': float(pred_aqi)})
        
    return pd.DataFrame(predictions)

if __name__ == "__main__":
    train_and_evaluate()
