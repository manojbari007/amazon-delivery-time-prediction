import pandas as pd
import numpy as np
from geopy.distance import geodesic

def calculate_distance(row):
    """Calculate geodesic distance between store and drop location."""
    try:
        store_loc = (row['Store_Latitude'], row['Store_Longitude'])
        drop_loc = (row['Drop_Latitude'], row['Drop_Longitude'])
        return geodesic(store_loc, drop_loc).km
    except ValueError:
        return np.nan

def extract_time_features(df):
    """Extract features from date and time columns."""
    if 'Order_Date' in df.columns:
        df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors='coerce')
        df['Order_Day_Of_Week'] = df['Order_Date'].dt.dayofweek
        df['Order_Month'] = df['Order_Date'].dt.month
        df['Is_Weekend'] = df['Order_Day_Of_Week'].apply(lambda x: 1 if x >= 5 else 0)
        
    if 'Order_Time' in df.columns:
        # Check if Order_Time is full datetime or just time
        df['Order_Hour'] = pd.to_datetime(df['Order_Time'], format='%H:%M:%S', errors='coerce').dt.hour
        mask = df['Order_Hour'].isna()
        if mask.any():
             df.loc[mask, 'Order_Hour'] = pd.to_datetime(df.loc[mask, 'Order_Time'], errors='coerce').dt.hour
        
        # Fill NaN hours with median if any remain
        df['Order_Hour'] = df['Order_Hour'].fillna(df['Order_Hour'].median())
        
        # New Feature: Is Peak Hour (e.g., lunch and dinner times)
        df['Is_Peak_Hour'] = df['Order_Hour'].apply(lambda x: 1 if (12 <= x <= 14) or (18 <= x <= 22) else 0)

    # Time taken for pickup if both Order_Time and Pickup_Time exist
    if 'Order_Time' in df.columns and 'Pickup_Time' in df.columns:
        try:
            ot = pd.to_datetime(df['Order_Time'], format='%H:%M:%S', errors='coerce')
            pt = pd.to_datetime(df['Pickup_Time'], format='%H:%M:%S', errors='coerce')
            df['Pickup_Wait_Min'] = (pt - ot).dt.total_seconds() / 60.0
            # Handle negative wait times (day rollover)
            df.loc[df['Pickup_Wait_Min'] < 0, 'Pickup_Wait_Min'] += 1440
            df['Pickup_Wait_Min'] = df['Pickup_Wait_Min'].fillna(df['Pickup_Wait_Min'].median())
        except:
            pass

    return df

def feature_engineering_pipeline(df):
    """Run all feature engineering steps."""
    print("Calculating distances... this might take a while.")
    df['Distance_km'] = df.apply(calculate_distance, axis=1)
    
    print("Extracting time features...")
    df = extract_time_features(df)
    
    # Drop rows where distance could not be calculated
    df = df.dropna(subset=['Distance_km'])

    # --- ADVANCED ACCURACY BOOSTERS ---
    print("Creating interaction features for higher accuracy...")
    
    # 1. Distance and Traffic Interaction (Multiplicative impact)
    traffic_map = {'low': 1, 'medium': 2, 'high': 3, 'jam': 4}
    df['Traffic_Numeric'] = df['Traffic'].map(traffic_map).fillna(2)
    df['Dist_Traffic_Interaction'] = df['Distance_km'] * df['Traffic_Numeric']
    
    # 2. Weather and Traffic Interaction
    weather_map = {'sunny': 1, 'cloudy': 2, 'windy': 3, 'fog': 4, 'sandstorms': 5, 'stormy': 6}
    df['Weather_Numeric'] = df['Weather'].map(weather_map).fillna(2)
    df['Weather_Traffic_Factor'] = df['Weather_Numeric'] * df['Traffic_Numeric']

    # 3. Efficiency Score (Agent Rating weighted by Distance)
    df['Agent_Efficiency_Index'] = (df['Agent_Rating'] / (df['Distance_km'] + 1))
    
    # 4. Log Distance (Address skewness if any)
    df['Log_Distance'] = np.log1p(df['Distance_km'])

    return df

if __name__ == "__main__":
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, "..", "dataset", "amazon_delivery_cleaned.csv")
    
    try:
        df = pd.read_csv(filepath)
        df = feature_engineering_pipeline(df)
        
        # Save processed data
        output_path = os.path.join(script_dir, "..", "dataset", "amazon_delivery_processed.csv")
        df.to_csv(output_path, index=False)
        print(f"Feature engineering complete. Saved to {output_path}")
        print(df.head())
    except FileNotFoundError:
        print("Cleaned data not found. Run data_preprocessing.py first.")
    except Exception as e:
        print(f"Error: {e}")
