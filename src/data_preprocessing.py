import pandas as pd
import numpy as np

def load_data(filepath):
    """Load dataset from csv file."""
    return pd.read_csv(filepath)

def clean_data(df):
    """Perform data cleaning operations."""
    # Drop duplicates
    df = df.drop_duplicates()

    # Handle coordinates that are 0.0
    cols_to_fix = ['Store_Latitude', 'Store_Longitude', 'Drop_Latitude', 'Drop_Longitude']
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = df[col].replace(0.0, np.nan)

    # Handle missing values
    # Agent_Age: fill with median
    if 'Agent_Age' in df.columns:
        df['Agent_Age'] = pd.to_numeric(df['Agent_Age'], errors='coerce')
        df['Agent_Age'] = df['Agent_Age'].fillna(df['Agent_Age'].median())
    
    # Agent_Rating: fill with median
    if 'Agent_Rating' in df.columns:
        df['Agent_Rating'] = pd.to_numeric(df['Agent_Rating'], errors='coerce')
        df['Agent_Rating'] = df['Agent_Rating'].fillna(df['Agent_Rating'].median())
        
    # Weather and Traffic: fill with mode
    if 'Weather' in df.columns:
        df['Weather'] = df['Weather'].fillna(df['Weather'].mode()[0])
    
    if 'Traffic' in df.columns:
        df['Traffic'] = df['Traffic'].fillna(df['Traffic'].mode()[0])

    # Drop rows where critical locations are missing
    df = df.dropna(subset=['Store_Latitude', 'Store_Longitude', 'Drop_Latitude', 'Drop_Longitude'])

    return df

def preprocess_categoricals(df):
    """Standardize categorical string formats."""
    # Apply to all object columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()
    return df

if __name__ == "__main__":
    import os
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct path to the dataset
    filepath = os.path.join(script_dir, "..", "dataset", "amazon_delivery.csv")
    try:
        df = load_data(filepath)
        print(f"Original shape: {df.shape}")
        
        df = clean_data(df)
        df = preprocess_categoricals(df)
        
        print(f"Cleaned shape: {df.shape}")
        
        # Save cleaned data
        output_path = os.path.join(script_dir, "..", "dataset", "amazon_delivery_cleaned.csv")
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
    except Exception as e:
        print(f"Error: {e}")
