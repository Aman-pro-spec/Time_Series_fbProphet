import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px
import plotly.graph_objs as go
from prophet.plot import plot_plotly, plot_components_plotly, plot_cross_validation_metric
from prophet.diagnostics import cross_validation, performance_metrics
import os

st.set_page_config(
    page_title="Time Series Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #1E3A8A;
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3203/3203071.png", width=100)
st.sidebar.title("Data Upload")
st.sidebar.markdown("Upload your time series dataset or use the default `AIR_PASSENGERS.csv`.")

data_source = st.sidebar.radio("Select Data Source", ["Default (Air Passengers)", "Upload Custom CSV"])

uploaded_file = None
if data_source == "Upload Custom CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

@st.cache_data
def load_data(source, uploaded_file_obj=None):
    if source == "Default (Air Passengers)":
        default_path = "AIR_PASSENGERS.csv"
        if os.path.exists(default_path):
            df = pd.read_csv(default_path)
            return df
        else:
            st.error(f"Default file {default_path} not found in the current directory.")
            return None
    else:
        if uploaded_file_obj is not None:
            df = pd.read_csv(uploaded_file_obj)
            return df
        return None

df = load_data(data_source, uploaded_file)

# --- Main App ---
st.title("📈 Time Series Forecasting Dashboard")
st.markdown("A dynamic, interactive web app for visualizing and predicting time series data using **Facebook Prophet**.")

if df is not None:
    st.markdown("---")
    st.header("1. Exploratory Data Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("Dataset Shape")
        st.info(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    with col2:
        st.subheader("Data Configuration")
        # Let user select date and target columns
        columns = df.columns.tolist()
        date_col = st.selectbox("Select Date Column", columns, index=0)
        target_col = st.selectbox("Select Target Variable (Y)", [c for c in columns if c != date_col], index=0 if len(columns) > 1 else None)
        
    if date_col and target_col:
        # Plotting the raw data
        st.subheader("Historical Time Series Plot")
        fig = px.line(df, x=date_col, y=target_col, title=f"{target_col} over time", template="plotly_white")
        fig.update_traces(line=dict(color='#2563EB', width=2))
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
        # --- Forecasting Section ---
        st.markdown("---")
        st.header("2. Prophet Forecasting")
        
        forecast_horizon = st.slider("Select Forecast Horizon (Periods)", min_value=1, max_value=365, value=12, step=1)
        freq = st.selectbox("Select Data Frequency", ["D", "W", "M", "Y"], index=2, help="D: Daily, W: Weekly, M: Monthly, Y: Yearly")
        
        if st.button("🚀 Generate Forecast"):
            with st.spinner("Training Prophet model and predicting..."):
                try:
                    # Prepare dataframe for Prophet
                    df_prophet = df[[date_col, target_col]].rename(columns={date_col: 'ds', target_col: 'y'})
                    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
                    
                    # Initialize and fit model
                    m = Prophet(seasonality_mode='multiplicative') # Often good for air passengers
                    m.fit(df_prophet)
                    
                    # Create future dataframe
                    future = m.make_future_dataframe(periods=forecast_horizon, freq=freq)
                    forecast = m.predict(future)
                    
                    st.success("Forecast generated successfully!")
                    
                    # Tabs for results
                    tab1, tab2, tab3, tab4 = st.tabs(["Interactive Forecast Plot", "Forecast Components", "Forecast Data", "Model Accuracy"])
                    
                    with tab1:
                        st.subheader("Forecast Plot")
                        fig_forecast = plot_plotly(m, forecast)
                        fig_forecast.update_layout(template="plotly_white", hovermode="x unified")
                        st.plotly_chart(fig_forecast, use_container_width=True)
                        
                    with tab2:
                        st.subheader("Trend & Seasonality Components")
                        fig_components = plot_components_plotly(m, forecast)
                        fig_components.update_layout(template="plotly_white")
                        st.plotly_chart(fig_components, use_container_width=True)
                        
                    with tab3:
                        st.subheader("Raw Forecast Values")
                        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_horizon))
                        
                    with tab4:
                        st.subheader("Cross-Validation & Metrics")
                        st.markdown("Run cross-validation to calculate accuracy metrics (RMSE, MAE, MAPE). *Note: This may take a minute.*")
                        
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            cv_initial = st.number_input("Initial Training Period (days)", min_value=30, value=730, step=30)
                        with col_b:
                            cv_period = st.number_input("Period between Cutoffs (days)", min_value=10, value=180, step=30)
                        with col_c:
                            cv_horizon = st.number_input("Forecast Horizon (days)", min_value=10, value=365, step=30)
                            
                        if st.button("📊 Evaluate Model Accuracy"):
                            with st.spinner("Running cross-validation..."):
                                try:
                                    df_cv = cross_validation(m, initial=f'{cv_initial} days', period=f'{cv_period} days', horizon=f'{cv_horizon} days', parallel="threads")
                                    df_p = performance_metrics(df_cv)
                                    
                                    st.success("Cross-validation completed!")
                                    
                                    st.markdown("#### Average Metrics")
                                    # Show average metrics over the horizon
                                    metrics_summary = df_p[['rmse', 'mae', 'mape']].mean().to_frame(name="Average Value").T
                                    st.dataframe(metrics_summary, use_container_width=True)
                                    
                                    st.markdown("#### MAPE Plot")
                                    fig_cv = plot_cross_validation_metric(df_cv, metric='mape')
                                    st.pyplot(fig_cv)
                                    
                                    st.markdown("#### Detailed Performance Metrics")
                                    st.dataframe(df_p, use_container_width=True)
                                except Exception as cv_e:
                                    st.error(f"Error during cross-validation: {cv_e}")
                                    st.info("Tip: Try reducing the initial, period, or horizon days. Your dataset must be large enough to support the chosen parameters.")
                except Exception as e:
                    st.error(f"An error occurred during forecasting: {e}")
else:
    st.info("👆 Please upload a CSV file or use the default dataset to begin.")
