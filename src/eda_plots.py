import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_eda_reports():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "dataset", "amazon_delivery_processed.csv")
    
    if not os.path.exists(data_path):
        print("Processed data not found.")
        return

    df = pd.read_csv(data_path)
    
    # Set style
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Distribution of Delivery Times
    plt.figure()
    sns.histplot(df['Delivery_Time'], kde=True, color='orange')
    plt.title('Distribution of Delivery Times (Minutes)')
    plt.xlabel('Delivery Time (min)')
    plt.savefig(os.path.join(script_dir, "..", "delivery_time_dist.png"))
    
    # 2. Impact of Traffic on Delivery Time
    plt.figure()
    sns.boxplot(x='Traffic', y='Delivery_Time', data=df, palette='viridis')
    plt.title('Impact of Traffic on Delivery Time')
    plt.savefig(os.path.join(script_dir, "..", "traffic_impact.png"))
    
    # 3. Impact of Weather on Delivery Time
    plt.figure()
    sns.boxplot(x='Weather', y='Delivery_Time', data=df, palette='magma')
    plt.title('Impact of Weather on Delivery Time')
    plt.savefig(os.path.join(script_dir, "..", "weather_impact.png"))
    
    # 4. Distance vs Delivery Time
    plt.figure()
    sns.scatterplot(x='Distance_km', y='Delivery_Time', data=df.sample(2000), alpha=0.5, hue='Traffic')
    plt.title('Distance vs Delivery Time')
    plt.savefig(os.path.join(script_dir, "..", "distance_vs_time.png"))
    
    # 5. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    plt.savefig(os.path.join(script_dir, "..", "correlation_heatmap.png"))
    
    print("EDA Visualizations generated successfully.")

if __name__ == "__main__":
    generate_eda_reports()
