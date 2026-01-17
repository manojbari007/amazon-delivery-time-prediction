import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import mlflow
import mlflow.sklearn
import joblib
from sklearn.impute import SimpleImputer
import os
import json

def load_data(filepath):
    return pd.read_csv(filepath)

def train_and_visualize():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "dataset", "amazon_delivery_processed.csv")
    
    df = load_data(data_path)
    
    # Define features and target
    # Exclude IDs, original date/times, and target
    exclude_cols = ['Order_ID', 'Order_Date', 'Order_Time', 'Pickup_Time', 
                    'Store_Latitude', 'Store_Longitude', 'Drop_Latitude', 'Drop_Longitude', 
                    'Delivery_Time']
    
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    target_col = 'Delivery_Time'
    
    # Filter outliers in target - common in delivery data
    # Some values might be garbage. Let's keep it between 5 and 300 mins.
    df = df[(df[target_col] >= 5) & (df[target_col] <= 240)]
    
    X = df[feature_cols]
    y = df[target_col]
    
    print(f"Features used: {feature_cols}")
    print(f"Total samples after filtering: {len(df)}")
    
    # Preprocessing
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
        
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Base models for stacking
    base_estimators = [
        ('xgb', xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=8, random_state=42)),
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42))
    ]

    # Models to train
    models = {
        'Optimized XGBoost': xgb.XGBRegressor(
            objective='reg:absoluteerror', 
            n_estimators=800, 
            learning_rate=0.05, 
            max_depth=8, 
            random_state=42
        ),
        'Tuned Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200, 
            learning_rate=0.05, 
            max_depth=8, 
            loss='huber', 
            random_state=42
        ),
        'Elite Stacking Ensemble': StackingRegressor(
            estimators=base_estimators,
            final_estimator=xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
            n_jobs=-1
        )
    }
    
    best_model = None
    best_score = -float('inf')
    results = []
    best_metrics = {}
    
    # Setup MLflow experiment
    mlflow.set_experiment("Amazon_Delivery_Prediction_Enhanced")
    
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('model', model)])
            
            print(f"Training {name}...")
            pipeline.fit(X_train, y_train)
            
            preds = pipeline.predict(X_test)
            
            mae = mean_absolute_error(y_test, preds)
            mse = mean_squared_error(y_test, preds)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, preds)
            
            print(f"{name} Results: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
            
            # Log metrics
            mlflow.log_param("model_type", name)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            
            # Log model
            mlflow.sklearn.log_model(pipeline, "model")
            
            results.append({'Model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2})
            
            if r2 > best_score:
                best_score = r2
                best_model = pipeline
                best_metrics = {
                    'model_name': name,
                    'mae': round(mae, 4),
                    'rmse': round(rmse, 4),
                    'r2': round(r2, 4)
                }
                
    # Display comparison table
    results_df = pd.DataFrame(results).sort_values(by='R2', ascending=False)
    print("\nModel Comparison Table:")
    print(results_df)
    
    # Save best model and metrics
    if best_model:
        model_dir = os.path.join(script_dir, "..", "models")
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, "best_model.pkl")
        joblib.dump(best_model, model_path)
        
        metrics_path = os.path.join(model_dir, "best_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(best_metrics, f)
            
        print(f"\nBest model saved to {model_path} with R2: {best_score}")
        print(f"Best metrics saved to {metrics_path}")

if __name__ == "__main__":
    train_and_visualize()
