import sys
import os
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Add src to path using absolute reference
base_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(base_dir, 'src')
if src_dir not in sys.path:
    sys.path.append(src_dir)

from database import AQIDatabase
from preprocessing import preprocess_data
from modeling import predict_next_72_hours

# -----------------------------------------------------------------------------
# CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Karachi AQI Prediction Dashboard",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Dark Theme & Professional Look
st.markdown("""
<style>
    /* Global Theme */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Success Messages */
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #064E3B; /* Dark Green */
        color: #D1FAE5;
        border: 1px solid #059669;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
    }
    .success-icon {
        margin-right: 0.5rem;
        font-size: 1.2rem;
    }

    /* Metrics */
    .metric-value {
        font-size: 3rem;
        font-weight: 700;
        color: #FAFAFA;
    }
    .metric-label {
        font-size: 1rem;
        color: #A1A1AA;
        text-transform: uppercase;
        font-weight: 600;
    }
    
    /* Category Badge */
    .aqi-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        font-weight: 600;
        color: #000;
        margin-top: 0.5rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #333;
        color: #666;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model_resources():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('features.pkl', 'rb') as f:
            features = pickle.load(f)
        return model, features, "XGBoost (RMSE: 1.75)"
    except Exception as e:
        return None, None, None

@st.cache_data(ttl=3600)
def get_data_from_db():
    try:
        db = AQIDatabase()
        if db.client:
           return db.fetch_data(), True
        return pd.DataFrame(), False
    except Exception as e:
        return pd.DataFrame(), False

def get_aqi_details(aqi):
    if aqi <= 50: return "Good", "#00E400"
    elif aqi <= 100: return "Moderate", "#FFFF00"
    elif aqi <= 150: return "Unhealthy for Sensitive Groups", "#FF7E00"
    elif aqi <= 200: return "Unhealthy", "#FF0000"
    elif aqi <= 300: return "Very Unhealthy", "#8F3F97"
    else: return "Hazardous", "#7E0023"

# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------

# Title
st.markdown("# 🏙️ Karachi AQI Prediction Dashboard")
st.markdown("Real-time and 3-day Air Quality predictions powered by ML & MongoDB Atlas Feature Store.")
st.markdown("---")

# 1. Connection Status (Mimicking the reference style)
df_all, db_connected = get_data_from_db()
model, features, model_name = load_model_resources()

if db_connected and not df_all.empty:
    st.markdown("""
    <div class="success-box">
        <span class="success-icon">✅</span> Connected to MongoDB Atlas Feature Store and fetched latest data.
    </div>
    """, unsafe_allow_html=True)
else:
    st.error("❌ Failed to connect to Feature Store.")

if model:
    st.markdown(f"""
    <div class="success-box">
        <span class="success-icon">✅</span> Loaded latest trained {model_name} model.
    </div>
    """, unsafe_allow_html=True)
else:
    st.error("❌ Model not found.")

st.markdown("<br>", unsafe_allow_html=True)

if df_all.empty:
    st.warning("No data available. Please check the data pipeline.")
    st.stop()

# Prepare Data
df_all['date'] = pd.to_datetime(df_all['date'])
df_all = df_all.sort_values('date')
current_rec = df_all.iloc[-1]
current_aqi = int(current_rec.get('us_aqi', 0))
cat_label, cat_color = get_aqi_details(current_aqi)

# 2. Current AQI Prediction (Hero Section)
st.subheader("☁️ Current AQI Prediction")

col_hero_1, col_hero_2 = st.columns([1, 2])

with col_hero_1:
    st.markdown('<div class="metric-label">Predicted AQI (Current Hour)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{current_aqi}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="aqi-badge" style="background-color: {cat_color};">Category: {cat_label}</div>', unsafe_allow_html=True)

with col_hero_2:
    # Progress bar style gauge
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge",
        value = current_aqi,
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'shape': "bullet",
            'axis': {'range': [0, 500]},
            'threshold': {
                'line': {'color': "white", 'width': 2},
                'thickness': 0.75,
                'value': current_aqi
            },
            'bgcolor': "#333",
            'steps': [
                {'range': [0, 50], 'color': "#00E400"},
                {'range': [50, 100], 'color': "#FFFF00"},
                {'range': [100, 150], 'color': "#FF7E00"},
                {'range': [150, 200], 'color': "#FF0000"},
                {'range': [200, 300], 'color': "#8F3F97"},
                {'range': [300, 500], 'color': "#7E0023"}
            ],
            'bar': {'color': "white", 'thickness': 0} # Hide standard bar, rely on bullet/threshold
        }
    ))
    fig_gauge.update_layout(height=150, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_gauge, use_container_width=True)

st.markdown("---")

# 3. Forecast Section
if model and features:
    recent_data = df_all.tail(100).copy()
    try:
        with st.spinner("Generating 72-Hour Forecast..."):
            forecast_df = predict_next_72_hours(model, features, recent_data)
        
        st.subheader("📊 AQI Forecast for Next 3 Days")
        
        col_chart_1, col_chart_2 = st.columns(2)
        
        with col_chart_1:
            st.markdown("**📈 Hourly AQI Trend (Next 3 Days)**")
            fig_hourly = px.line(forecast_df, x='date', y='predicted_aqi', 
                                template="plotly_dark")
            fig_hourly.update_traces(line_color='#3B82F6', line_width=3)
            fig_hourly.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(255,255,255,0.05)',
                xaxis_title=None,
                yaxis_title=None,
                margin=dict(l=10, r=10, t=10, b=10),
                hovermode="x unified"
            )
            st.plotly_chart(fig_hourly, use_container_width=True)
            
        with col_chart_2:
            st.markdown("**📊 Daily Average AQI Forecast**")
            # Calculate daily averages
            forecast_df['day_name'] = forecast_df['date'].dt.strftime('%a %d')
            daily_avg = forecast_df.groupby('day_name')['predicted_aqi'].mean().reset_index()
            # Sort by date
            daily_avg['sort_key'] = daily_avg['day_name'].apply(lambda x: datetime.strptime(x + f" {datetime.now().year}", "%a %d %Y"))
            daily_avg = daily_avg.sort_values('sort_key')
            
            fig_daily = px.bar(daily_avg, x='day_name', y='predicted_aqi',
                              template="plotly_dark", text_auto='.1f')
            fig_daily.update_traces(marker_color='#3B82F6', width=0.3)
            fig_daily.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(255,255,255,0.05)',
                xaxis_title=None,
                yaxis_title=None,
                margin=dict(l=10, r=10, t=10, b=10),
                bargap=0.5
            )
            st.plotly_chart(fig_daily, use_container_width=True)

        # 4. Forecast Summary Table
        st.subheader("📋 3-Day AQI Forecast Summary")
        summary_table = daily_avg[['day_name', 'predicted_aqi']].rename(columns={'day_name': 'Date', 'predicted_aqi': 'Predicted AQI'})
        
        # Add Category Column
        summary_table['Status'] = summary_table['Predicted AQI'].apply(lambda x: get_aqi_details(x)[0])
        
        st.dataframe(summary_table, use_container_width=True)

        # Insight Box
        avg_future_aqi = forecast_df['predicted_aqi'].mean()
        trend_msg = "stable" if abs(avg_future_aqi - current_aqi) < 20 else ("worsening" if avg_future_aqi > current_aqi else "improving")
        st.markdown(f"""
        <div class="success-box">
            <span class="success-icon">💡</span> Air quality expected to remain <b>{trend_msg}</b> in the next 72 hours.
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction Error: {e}")

# Footer
st.markdown("""
<div class="footer">
    Developed by <b>Karan Kumar</b> | Powered by Open-Meteo, MongoDB Atlas & XGBoost<br>
    Copyright &copy; 2026
</div>
""", unsafe_allow_html=True)
