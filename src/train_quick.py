import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import joblib
import os

def train_quick():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "dataset", "amazon_delivery_processed.csv")
    
    if not os.path.exists(data_path):
        print("Data not found!")
        return

    df = pd.read_csv(data_path)
    
    exclude_cols = ['Order_ID', 'Order_Date', 'Order_Time', 'Pickup_Time', 
                    'Store_Latitude', 'Store_Longitude', 'Drop_Latitude', 'Drop_Longitude', 
                    'Delivery_Time']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    target_col = 'Delivery_Time'
    
    X = df[feature_cols]
    y = df[target_col]
    
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ])
        
    # Quick model
    model = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=-1)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])
    
    print("Training quick model...")
    pipeline.fit(X, y) # Train on full data for simplicity or split? Train on full for final model.
    
    model_path = os.path.join(script_dir, "..", "models", "best_model.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Quick model saved to {model_path}")

if __name__ == "__main__":
    train_quick()
