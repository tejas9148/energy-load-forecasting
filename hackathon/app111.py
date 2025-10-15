import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
import warnings
import numpy as np
# Suppress warnings to keep the app looking clean
warnings.filterwarnings("ignore")

# --- Page Configuration ---
st.set_page_config(
    page_title="Energy Demand Forecast Dashboard",
    page_icon="⚡",
    layout="wide"
)

# --- Data Loading and Caching ---
@st.cache_data
def load_data(file_path):
    """
    Loads, cleans, and resamples the energy consumption data from a CSV file.
    This function is cached to run only once.
    """
    file_path= r"C:\Users\tejas\Documents\hackthon_project1\AEP_hourly.csv.zip"
    df = pd.read_csv(file_path)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime')
    # Resample from hourly to daily by summing up the consumption
    daily_df = df['AEP_MW'].resample('D').sum().to_frame()
    return daily_df

# --- Model Training and Caching ---
@st.cache_resource
def train_and_predict(_data):
    """
    Trains the SARIMA model and generates predictions and a future forecast.
    This function is cached to run only once.
    """
    train_size = int(len(_data) * 0.9)
    train_data = _data.iloc[:train_size]
    test_data = _data.iloc[train_size:].copy()

    # Define and fit the model
    model = SARIMAX(train_data['AEP_MW'],
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, 7))
    results = model.fit(disp=False)

    # Get predictions and forecast
    predictions = results.predict(start=test_data.index[0], end=test_data.index[-1])
    test_data['Predicted_MW'] = predictions

    future_forecast = results.get_forecast(steps=30)
    future_index = pd.date_range(start=_data.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')
    future_df = pd.DataFrame({'Forecast_MW': future_forecast.predicted_mean.values}, index=future_index)
    
    return test_data, future_df, _data

# --- Main App ---
st.title("⚡ Energy Demand Forecast Dashboard")
st.markdown("An interactive dashboard to visualize historical energy usage, model predictions, and future demand.")

# Define the path to your data file. 
# IMPORTANT: This assumes 'AEP_hourly.csv' is in the SAME FOLDER as your 'app.py' script.
daily_data = load_data(r'C:\Users\tejas\Documents\hackthon_project1\AEP_hourly.csv.zip')
data_file= r"C:\Users\tejas\Documents\hackthon_project1\AEP_hourly.csv.zip"


# Load data and train model (with a loading spinner)
with st.spinner("Loading data and training the forecast model... This may take a moment."):
    try:
        daily_data = load_data(data_file)
        predictions_df, forecast_df, full_daily_data = train_and_predict(daily_data)
        st.success("Data loaded and model trained successfully!")

        # --- Display Model Performance ---
        # --- Display Model Performance ---
       # --- Display Model Performance ---
        st.subheader("Model Performance Evaluation")

# Calculate Mean Absolute Error (MAE)
        mae = mean_absolute_error(predictions_df['AEP_MW'], predictions_df['Predicted_MW'])

# Calculate Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((predictions_df['AEP_MW'] - predictions_df['Predicted_MW']) / predictions_df['AEP_MW'])) * 100

# Calculate Root Mean Squared Error (RMSE)
        rmse = np.sqrt(np.mean((predictions_df['AEP_MW'] - predictions_df['Predicted_MW'])**2))

# Create three columns to display the metrics
        col1, col2, col3 = st.columns(3)
        col1.metric(label="Mean Absolute Error (MAE)", value=f"{mae:,.2f} MW")
        col2.metric(label="Mean Absolute Percentage Error (MAPE)", value=f"{mape:.2f}%")
        col3.metric(label="Root Mean Squared Error (RMSE)", value=f"{rmse:,.2f} MW")

        st.markdown("(These metrics represent the average error of the model's predictions.)")

        # --- Dashboard Visualizations ---
        st.subheader("Interactive Forecast Visualizations")

        # Chart 1: Actual vs. Predicted
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=predictions_df.index, y=predictions_df['AEP_MW'], mode='lines', name='Actual Usage', line=dict(color='deepskyblue')))
        fig1.add_trace(go.Scatter(x=predictions_df.index, y=predictions_df['Predicted_MW'], mode='lines', name='Predicted Usage', line=dict(color='red', dash='dash')))
        fig1.update_layout(title='<b>Actual vs. Predicted Energy Demand</b>', xaxis_title='Date', yaxis_title='Energy (MW)', template='plotly_white')
        st.plotly_chart(fig1, use_container_width=True)

        # Chart 2: Future Forecast
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=full_daily_data.tail(100).index, y=full_daily_data.tail(100)['AEP_MW'], mode='lines', name='Historical Usage', line=dict(color='gray')))
        fig2.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecast_MW'], mode='lines', name='30-Day Forecast', line=dict(color='crimson', dash='dot', width=3)))
        fig2.update_layout(title='<b>30-Day Future Demand Forecast</b>', xaxis_title='Date', yaxis_title='Forecasted Energy (MW)', template='plotly_white')
        st.plotly_chart(fig2, use_container_width=True)

        # Chart 3: Heatmap
        with st.expander("Explore Seasonal Consumption Patterns"):
            daily_df_viz = full_daily_data.copy()
            daily_df_viz['year'] = daily_df_viz.index.year
            daily_df_viz['day_of_week'] = daily_df_viz.index.day_name()
            daily_df_viz['month'] = daily_df_viz.index.month_name()
            last_year_data = daily_df_viz[daily_df_viz['year'] == daily_df_viz['year'].max() - 1]
            
            fig3 = px.density_heatmap(last_year_data, x='month', y='day_of_week', z='AEP_MW',
                                      category_orders={"month": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
                                                       "day_of_week": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]},
                                      color_continuous_scale="Viridis",
                                      title=f'<b>Energy Consumption Heatmap for {last_year_data.year.max()}</b>')
            st.plotly_chart(fig3, use_container_width=True)

    except FileNotFoundError:
        st.error(f"Error: The data file '{data_file}' was not found. Please make sure it's in the same folder as your app.py script.")