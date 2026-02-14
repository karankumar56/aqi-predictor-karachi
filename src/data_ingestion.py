import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import os
from datetime import datetime, timedelta

def fetch_historical_data():
    """
    Fetches historical AQI data for Karachi from Open Meteo API for the last 6 months.
    """
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Coordinates for Karachi
    latitude = 24.8607
    longitude = 67.0011

    # Calculate date range (last 6 months)
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=6*30)

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone", "us_aqi"]
    }

    print(f"Fetching data from {start_date} to {end_date}...")
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    # Process hourly data
    hourly = response.Hourly()
    hourly_pm10 = hourly.Variables(0).ValuesAsNumpy()
    hourly_pm2_5 = hourly.Variables(1).ValuesAsNumpy()
    hourly_co = hourly.Variables(2).ValuesAsNumpy()
    hourly_no2 = hourly.Variables(3).ValuesAsNumpy()
    hourly_so2 = hourly.Variables(4).ValuesAsNumpy()
    hourly_o3 = hourly.Variables(5).ValuesAsNumpy()
    hourly_us_aqi = hourly.Variables(6).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}
    hourly_data["pm10"] = hourly_pm10
    hourly_data["pm2_5"] = hourly_pm2_5
    hourly_data["carbon_monoxide"] = hourly_co
    hourly_data["nitrogen_dioxide"] = hourly_no2
    hourly_data["sulphur_dioxide"] = hourly_so2
    hourly_data["ozone"] = hourly_o3
    hourly_data["us_aqi"] = hourly_us_aqi

    df = pd.DataFrame(data = hourly_data)
    return df

if __name__ == "__main__":
    from database import AQIDatabase
    
    try:
        print("Starting data ingestion process...")
        
        # 1. Fetch data from Open-Meteo
        df = fetch_historical_data()
        print(f"✅ Fetched {len(df)} records from Open-Meteo API.")
        
        # 2. Store in MongoDB
        if not df.empty:
            print("Connecting to MongoDB Atlas...")
            db = AQIDatabase()
            db.insert_data(df)
            print("✅ Data successfully ingested into MongoDB.")
        else:
            print("⚠️ No data fetched.")

    except Exception as e:
        print(f"❌ Data ingestion failed: {e}")
