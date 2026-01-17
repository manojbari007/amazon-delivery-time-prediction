# 📦 Amazon Delivery Time Prediction Pro

An end-to-end Machine Learning solution to predict delivery arrival times with high accuracy using historical logistics data.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://amazon-delivery-time-prediction-besx8j26hpnjxyqc8nas75.streamlit.app/)

## 🚀 Key Features

- **Geospatial Analytics:** Geodesic distance calculations between fulfillment centers and customers.
- **Advanced Feature Engineering:** Time-of-day peak analysis, weekend effects, and pickup latency factors.
- **Multi-Model Pipeline:** Automated training and comparison of Linear Regression, Random Forest, and XGBoost.
- **Experimental Tracking:** Full integration with **MLflow** for hyperparameter and metric logging.
- **Premium Dashboard:** High-fidelity Streamlit interface with interactive analytics and real-time predictions.

## 🛠️ Tech Stack

- **Languages:** Python
- **Data:** Pandas, NumPy, Geopy
- **ML:** Scikit-Learn, XGBoost
- **Tracking:** MLflow
- **UI:** Streamlit, Seaborn, Matplotlib

## 📋 Project Structure

```text
├── dataset/             # Raw and processed CSV files
├── models/              # Serialized best models (.pkl)
├── mlruns/              # MLflow experiment tracking logs
├── src/
│   ├── data_preprocessing.py  # Cleaning and outlier handling
│   ├── feature_engineering.py # Geospatial & temporal features
│   ├── eda_plots.py           # Automated insight generation
│   ├── model_training.py      # ML pipeline & model selection
│   └── app.py                 # Streamlit application
└── requirements.txt     # Dependency list
```

## ⚙️ Execution Guide

1. **Environment Setup**

   ```bash
   pip install -r requirements.txt
   ```

2. **Data Pipeline (Run in sequence)**

   ```powershell
   python src/data_preprocessing.py  # Clean raw data
   python src/feature_engineering.py # Generate features
   python src/eda_plots.py           # Generate visual reports
   ```

3. **Machine Learning & Tracking**

   ```bash
   python src/model_training.py
   ```

   _The script selects the best performing model (highest R²) and saves it to `models/best_model.pkl`._

4. **Launch Dashboard**
   ```bash
   streamlit run src/app.py
   ```

## 📊 Business Impact

By accurately predicting delivery windows, this solution helps:

- Improve **Customer Satisfaction** via precise ETAs.
- Optimize **Fleet Management** by identifying bottleneck traffic paths.
- Enhance **Agent Evaluation** based on normalized rating benchmarks.
