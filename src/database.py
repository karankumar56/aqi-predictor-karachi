import os
import time
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ConfigurationError
from dotenv import load_dotenv
import pandas as pd
import streamlit as st

load_dotenv()

class AQIDatabase:
    def __init__(self):
        self.uri = os.getenv("MONGO_URI")
        self.db_name = os.getenv("DB_NAME", "aqi_db")
        self.collection_name = "hourly_aqi"
        self.client = None
        self.db = None
        self.collection = None
        
        if not self.uri:
            raise ValueError("MONGO_URI environment variable not set")
            
        self._connect()

    def _connect(self, retries=3, delay=2):
        """Establishes connection with retry logic."""
        for attempt in range(retries):
            try:
                # Add connection timeout options
                self.client = MongoClient(
                    self.uri, 
                    serverSelectionTimeoutMS=5000,
                    connectTimeoutMS=5000,
                    socketTimeoutMS=5000
                )
                self.db = self.client[self.db_name]
                self.collection = self.db[self.collection_name]
                
                self.client.admin.command('ping')
                print("Connected to MongoDB.")
                return
            except (ConnectionFailure, ConfigurationError) as e:
                print(f"Connection attempt {attempt+1} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(delay)
                else:
                    print("Max retries reached. Connection failed.")
                    raise e

    def insert_data(self, df):
        """
        Inserts DataFrame into MongoDB efficiently.
        """
        if df.empty:
            return

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        records = df.to_dict('records')
        
        try:
            # Upsert logic using bulk_write is most efficient
            from pymongo import UpdateOne
            operations = [
                UpdateOne({'date': r['date']}, {'$set': r}, upsert=True)
                for r in records
            ]
            
            if operations:
                self.collection.bulk_write(operations)
                print(f"Upserted/Updated {len(operations)} records.")
                
        except Exception as e:
            print(f"Error inserting data: {e}")
            raise

    def fetch_data(self):
        """
        Fetches all data from MongoDB and returns as DataFrame.
        """
        try:
            if self.collection is None:
                 self._connect()
                 
            cursor = self.collection.find().sort("date", 1)
            df = pd.DataFrame(list(cursor))
            if not df.empty and '_id' in df.columns:
                df = df.drop(columns=['_id'])
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            # Return empty DF on failure to allow UI to render empty state
            return pd.DataFrame()

if __name__ == "__main__":
    db = AQIDatabase()
