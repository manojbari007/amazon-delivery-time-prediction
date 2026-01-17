import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import json
st.set_page_config(page_title="Amazon Delivery Prediction Pro", layout="wide", page_icon="📦")

# Custom CSS for premium aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .stApp {
        background: transparent;
    }
    
    .header-style {
        background: linear-gradient(90deg, #232f3e 0%, #37475a 100%);
        padding: 40px;
        border-radius: 15px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    
    .card {
        background: rgba(255, 255, 255, 0.95);
        padding: 25px;
        border-radius: 15px;
        border-left: 5px solid #ff9900;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    .predict-btn button {
        background: linear-gradient(90deg, #ff9900 0%, #e68a00 100%) !important;
        border: none !important;
        color: white !important;
        font-size: 1.2rem !important;
        padding: 15px !important;
        border-radius: 10px !important;
        transition: transform 0.3s ease !important;
    }
    
    .predict-btn button:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 15px rgba(255, 153, 0, 0.4);
    }
    
    .result-card {
        background: #232f3e;
        color: #ff9900;
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        border: 2px solid #ff9900;
    }
    
    .metric-val {
        font-size: 3.5rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header-style">
    <h1>📦 Amazon Delivery Time Prediction Pro</h1>
    <p>Predicting minutes that matter using Advanced Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    script_dir = os.path.dirname(__file__)
    model_path = os.path.join(script_dir, "..", "models", "best_model.pkl")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

# Options
@st.cache_data
def get_options():
    return {
        'Weather': ['sunny', 'cloudy', 'fog', 'sandstorms', 'stormy', 'windy'],
        'Traffic': ['low', 'medium', 'high', 'jam'],
        'Vehicle': ['bicycle', 'electric_scooter', 'motorcycle', 'scooter', 'van'],
        'Area': ['urban', 'metropolitian', 'semi-urban', 'other'],
        'Category': ['clothing', 'electronics', 'toys', 'watch', 'books', 'cosmetics', 'snacks', 'shoes', 'apparel', 'jewelry', 'outdoors', 'grocery', 'kitchen', 'pet supplies', 'skincare', 'home']
    }

options = get_options()

if model is None:
    st.error("❌ Model not found. Please run Training first.")
else:
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📍 Logistics Details")
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            distance = st.number_input("Distance (km)", 0.1, 100.0, 5.0)
            area = st.selectbox("Area Type", options['Area'])
            category = st.selectbox("Category", options['Category'])
        with d_col2:
            vehicle = st.selectbox("Vehicle", options['Vehicle'])
            traffic = st.selectbox("Traffic", options['Traffic'])
            weather = st.selectbox("Weather", options['Weather'])
        
        st.subheader("👤 Agent Profiles")
        a_col1, a_col2 = st.columns(2)
        with a_col1:
            agent_age = st.slider("Agent Age", 18, 60, 30)
        with a_col2:
            agent_rating = st.slider("Agent Rating", 1.0, 5.0, 4.5, 0.1)
        
        st.subheader("⏰ Timing & Delays")
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            order_date = st.date_input("Order Date")
            pickup_wait = st.slider("Pickup Delay (mins)", 0, 60, 15)
        with t_col2:
            order_time = st.time_input("Order Time")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # Advanced Feature Engineering for Input (Must match Training)
        order_day_of_week = order_date.weekday()
        order_month = order_date.month
        is_weekend = 1 if order_day_of_week >= 5 else 0
        order_hour = order_time.hour
        is_peak_hour = 1 if (12 <= order_hour <= 14) or (18 <= order_hour <= 22) else 0
        
        # New interaction features
        traffic_map = {'low': 1, 'medium': 2, 'high': 3, 'jam': 4}
        traffic_num = traffic_map.get(traffic, 2)
        weather_map = {'sunny': 1, 'cloudy': 2, 'windy': 3, 'fog': 4, 'sandstorms': 5, 'stormy': 6}
        weather_num = weather_map.get(weather, 2)
        
        input_df = pd.DataFrame({
            'Agent_Age': [agent_age],
            'Agent_Rating': [agent_rating],
            'Weather': [weather],
            'Traffic': [traffic],
            'Vehicle': [vehicle],
            'Area': [area],
            'Category': [category],
            'Distance_km': [distance],
            'Order_Day_Of_Week': [order_day_of_week],
            'Order_Month': [order_month],
            'Is_Weekend': [is_weekend],
            'Order_Hour': [order_hour],
            'Is_Peak_Hour': [is_peak_hour],
            'Pickup_Wait_Min': [float(pickup_wait)],
            # Efficiency Enhancers
            'Traffic_Numeric': [traffic_num],
            'Dist_Traffic_Interaction': [distance * traffic_num],
            'Weather_Numeric': [weather_num],
            'Weather_Traffic_Factor': [weather_num * traffic_num],
            'Agent_Efficiency_Index': [agent_rating / (distance + 1)],
            'Log_Distance': [np.log1p(distance)]
        })
        
        st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
        if st.button("🚀 CALCULATE DELIVERY TIME", use_container_width=True):
            with st.spinner("Analyzing parameters..."):
                prediction = model.predict(input_df)[0]
                
                st.markdown(f"""
                <div class="result-card">
                    <p style="font-size: 1.2rem; margin-bottom: 0;">Estimated Arrival In</p>
                    <div class="metric-val">{prediction:.0f} mins</div>
                    <p style="margin-top: 10px;">Predicted using {model.named_steps['model'].__class__.__name__}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.balloons()
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Insights or Model Info
        st.markdown('<div class="card" style="margin-top: 40px; border-left-color: #232f3e;">', unsafe_allow_html=True)
        st.subheader("💡 Prediction Insights")
        st.write(f"• **Weekend impact:** {'Active' if is_weekend else 'Inactive'}")
        st.write(f"• **Peak hour pressure:** {'Detected' if is_peak_hour else 'Normal'}")
        st.write(f"• **Weather conditions:** {weather.capitalize()}")
        st.markdown('</div>', unsafe_allow_html=True)

# Tabs for EDA and Info
st.markdown("---")
tab1, tab2, tab3 = st.tabs(["📊 Market Analytics", "🧠 Model performance", "ℹ️ System Info"])

with tab1:
    st.subheader("Data-Driven Logistics Insights")
    c1, c2 = st.columns(2)
    
    script_dir = os.path.dirname(__file__)
    traffic_img = os.path.join(script_dir, "..", "traffic_impact.png")
    weather_img = os.path.join(script_dir, "..", "weather_impact.png")
    dist_img = os.path.join(script_dir, "..", "delivery_time_dist.png")
    heatmap_img = os.path.join(script_dir, "..", "correlation_heatmap.png")
    
    if os.path.exists(traffic_img):
        with c1:
            st.image(traffic_img, caption="Traffic Impact Analysis")
            st.image(dist_img, caption="Delivery Time Distribution")
        with c2:
            st.image(weather_img, caption="Weather Impact Analysis")
            st.image(heatmap_img, caption="Feature Correlation Heatmap")
    else:
        st.info("Run `python src/eda_plots.py` to see advanced analytics.")

with tab2:
    st.subheader("🚀 Model Training Accuracy")
    
    script_dir = os.path.dirname(__file__)
    metrics_path = os.path.join(script_dir, "..", "models", "best_metrics.json")
    
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("R-Squared (Accuracy)", f"{metrics['r2']*100:.2f}%")
        with m2:
            st.metric("MAE (Error)", f"{metrics['mae']:.2f} min")
        with m3:
            st.metric("RMSE", f"{metrics['rmse']:.2f}")
            
        st.write(f"**Best Performing Model:** {metrics['model_name']}")
        st.info("The model was trained on 40,000+ records with cross-validation for maximum robustness.")
    else:
        st.write("Current model utilizes an ensemble approach with XGBoost for maximum accuracy.")
        st.warning("Metrics file not found. Please re-run training.")

with tab3:
    st.markdown("""
    **Amazon Delivery Time Prediction System**
    - **Dataset:** Amazon Logistics Data (40k+ records)
    - **Models:** Linear Regression, RandomForest, XGBoost
    - **Target Accuracy:** >80% R-squared
    - **Stack:** Python, Scikit-learn, XGBoost, MLflow, Streamlit
    """)
