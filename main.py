import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
from keras.layers import InputLayer
from keras import mixed_precision
from tensorflow.keras import mixed_precision
from tensorflow.keras.mixed_precision import Policy, set_global_policy, global_policy
from keras.src.mixed_precision import policy as policy_module 
from streamlit.runtime.scriptrunner import RerunException, get_script_run_ctx
import time
import os
import warnings
import logging

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")

# Suppress Streamlit context warnings
logging.getLogger('streamlit.runtime.scriptrunner.script_runner').setLevel(logging.ERROR)

warnings.filterwarnings('ignore', category=DeprecationWarning)

# TensorFlow + scaler for LSTM (if needed later)
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from auth.user_auth import add_user, login_user 


# --- Initialize login session state ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ''


# --- Sidebar Login/Register ---
st.sidebar.title("🔒 User Login")
menu = st.sidebar.selectbox("Menu", ["Login", "Register"])

if menu == "Login":
    st.sidebar.subheader("Login to Dashboard")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type='password')

    if st.sidebar.button("Login"):
        result = login_user(username, password)
        if result:
            st.sidebar.success(f"Welcome To The Market Global {username}!")
            st.session_state.logged_in = True
            st.session_state.username = username
        else:
            st.sidebar.error("Incorrect Username/Password")

elif menu == "Register":
    st.sidebar.subheader("Create New Account")
    new_username = st.sidebar.text_input("New Username")
    new_password = st.sidebar.text_input("New Password", type='password')

    if st.sidebar.button("Register"):
        add_user(new_username, new_password)
        st.sidebar.success("Account created successfully! Now click login.")


# --- Main Dashboard (only if logged in) ---
if st.session_state.logged_in:

    st.title("📊 Stock Price Forecasting & Live Dashboard")

    # Load data
    data = pd.read_csv('stock_data.csv', parse_dates=['Date'])
    data.set_index('Date', inplace=True)

    # Column list
    all_columns = [
        'AMZN', 'DPZ', 'BTC', 'NFLX', 'Natural_Gas_Price', 'Crude_oil_Price', 'Copper_Price',
        'Bitcoin_Price', 'Platinum_Price', 'Ethereum_Price', 'S&P_500_Price', 'Nasdaq_100_Price',
        'Apple_Price', 'Tesla_Price', 'Microsoft_Price', 'Silver_Price', 'Google_Price',
        'Nvidia_Price', 'Berkshire_Price', 'Netflix_Price', 'Amazon_Price', 'Meta_Price', 'Gold_Price'
    ]

    # Clean numeric columns
    for col in all_columns:
        if col in data.columns:
            data[col] = data[col].astype(str).str.replace(',', '').astype(float)

    # Select target column for prediction
    target_column = st.selectbox("Select the target column for prediction", options=all_columns)

    feature_columns = [col for col in all_columns if col != target_column]

    X = data[feature_columns]
    y = data[target_column]

    # Train/test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train RandomForest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    val_predictions = model.predict(X_val)
    mse = mean_squared_error(y_val, val_predictions)
    st.success(f"Validation Mean Squared Error: {mse:.4f}")

    # --- LSTM MODEL ---

    # Prepare data for LSTM (only using target column values for univariate time series prediction)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(y.values.reshape(-1, 1))

    def create_sequences(data, seq_length=10):
        Xs, ys = [], []
        for i in range(len(data) - seq_length):
            Xs.append(data[i:i+seq_length])
            ys.append(data[i+seq_length])
        return np.array(Xs), np.array(ys)

    seq_length = 10
    X_seq, y_seq = create_sequences(scaled_data, seq_length)

    # Split train/val for LSTM
    split = int(len(X_seq)*0.8)
    X_train_lstm, X_val_lstm = X_seq[:split], X_seq[split:]
    y_train_lstm, y_val_lstm = y_seq[:split], y_seq[split:]

    # Build LSTM model
    lstm_model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, activation='relu', input_shape=(seq_length, 1)),
        tf.keras.layers.Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')

    # Train LSTM
    lstm_model.fit(X_train_lstm, y_train_lstm, epochs=20, batch_size=16, verbose=0)

    # Predict with LSTM
    lstm_val_pred = lstm_model.predict(X_val_lstm)
    mse_lstm = mean_squared_error(y_val_lstm, lstm_val_pred)
    st.success(f"Validation Mean Squared Error (LSTM): {mse_lstm:.4f}")

    # Inverse scale for plotting
    y_val_lstm_rescaled = scaler.inverse_transform(y_val_lstm)
    lstm_val_pred_rescaled = scaler.inverse_transform(lstm_val_pred)

    # Plot LSTM Validation predictions
    st.header("📊 LSTM Validation Predictions")
    fig_lstm = go.Figure()
    fig_lstm.add_trace(go.Scatter(
        y=y_val_lstm_rescaled.flatten(),
        mode='lines+markers',
        name='Actual',
        line=dict(dash='dot', color='RoyalBlue')
    ))
    fig_lstm.add_trace(go.Scatter(
        y=lstm_val_pred_rescaled.flatten(),
        mode='lines+markers',
        name='Predicted',
        line=dict(color='Orange', width=1)
    ))
    fig_lstm.update_layout(
        title=f"{target_column} LSTM Validation Predictions",
        xaxis_title="Time Step",
        yaxis_title="Price",
        template='plotly_white'
    )
    st.plotly_chart(fig_lstm, use_container_width=True)

    # Future prediction settings
    st.sidebar.header("Forecast Settings")
    future_days = st.sidebar.slider("Future Prediction Days (1-100)", 1, 100, 7)

    last_known_date = data.index[-1]
    future_dates = [last_known_date + timedelta(days=i) for i in range(1, future_days + 1)]

    future_features = X.tail(future_days)
    if len(future_features) < future_days:
        last_row = future_features.iloc[-1]
        rows_to_add = future_days - len(future_features)
        future_features = pd.concat([future_features, pd.DataFrame([last_row]*rows_to_add, columns=feature_columns)])

    future_predictions = model.predict(future_features)

    # 1️⃣ Historical Prices Graph
    st.header("📈 1. Historical Prices")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=data.index,
        y=data[target_column],
        mode='lines+markers',
        name='Historical Prices',
        line=dict(color='Blue', width=1)
    ))
    fig_hist.update_layout(title=f"{target_column} Historical Prices",
                           xaxis_title="Date", yaxis_title="Price",
                           template='plotly_white')
    st.plotly_chart(fig_hist, use_container_width=True)

    # 2️⃣ Validation Predictions Graph
    st.header("📊 2. Validation Predictions")
    val_dates = y_val.index
    fig_val = go.Figure()
    fig_val.add_trace(go.Scatter(
        x=val_dates,
        y=y_val,
        mode='lines+markers',
        name='Actual',
        line=dict(dash='dot', color='Gold')
    ))
    fig_val.add_trace(go.Scatter(
        x=val_dates,
        y=val_predictions,
        mode='lines+markers',
        name='Predicted',
        line=dict(color='Salmon', width=1)
    ))
    fig_val.update_layout(title=f"{target_column} Validation Predictions",
                          xaxis_title="Date", yaxis_title="Price",
                          template='plotly_white')
    st.plotly_chart(fig_val, use_container_width=True)

    # 3️⃣ Future Predictions Graph
    st.header("🔮 3. Future Predictions")
    fig_future = go.Figure()
    fig_future.add_trace(go.Scatter(
        x=future_dates,
        y=future_predictions,
        mode='lines+markers',
        name='Future Prediction',
        line=dict(color='green', width=1.5)
    ))
    fig_future.update_layout(title=f"{target_column} Future {future_days} Days Predictions",
                             xaxis_title="Date", yaxis_title="Predicted Price",
                             template='plotly_white')
    st.plotly_chart(fig_future, use_container_width=True)



    # --- Real-Time OHLC Live Data ---

    def generate_ohlc(prev_close):
        open_price = prev_close + np.random.uniform(-5, 5)
        high_price = open_price + np.random.uniform(5, 5)
        low_price = open_price - np.random.uniform(5, 5)
        close_price = low_price + np.random.uniform(5, high_price - low_price)
        return round(open_price, 2), round(high_price, 2), round(low_price, 2), round(close_price, 2)

    def style_live_data(df):
        return df.style.format("{:.2f}", subset=['Open', 'High', 'Low', 'Close'])\
                   .background_gradient(cmap='viridis', subset=['Close'])

    # Initialize session_state for live data
    if 'live_data' not in st.session_state:
        initial_timestamp = datetime.now()
        initial_close = data[target_column].iloc[-1]
        open_p, high_p, low_p, close_p = generate_ohlc(initial_close)
        st.session_state.live_data = pd.DataFrame([{
            'Timestamp': initial_timestamp,
            'Open': open_p,
            'High': high_p,
            'Low': low_p,
            'Close': close_p
        }])

    st.header("📡 Real-Time OHLC Live Data")
    realtime_chart = st.empty()
    live_metrics_table = st.empty()

    # Run real-time updates (50 steps)
    for _ in range(50):
        new_timestamp = datetime.now()
        prev_close = st.session_state.live_data["Close"].iloc[-1]
        open_p, high_p, low_p, close_p = generate_ohlc(prev_close)

        new_row = {
            "Timestamp": new_timestamp,
            "Open": open_p,
            "High": high_p,
            "Low": low_p,
            "Close": close_p
        }

        # Append new row to live_data DataFrame
        st.session_state.live_data = pd.concat([st.session_state.live_data, pd.DataFrame([new_row])], ignore_index=True)

        # Show styled live data table (last 10 rows)
        live_metrics_table.dataframe(style_live_data(st.session_state.live_data.tail(10)))

        # Plot last 30 Close prices live
        fig_live = go.Figure()
        fig_live.add_trace(go.Scatter(
            x=st.session_state.live_data["Timestamp"].tail(30),
            y=st.session_state.live_data["Close"].tail(30),
            mode='lines+markers',
            name='Live Close Price',
            line=dict(color='MediumPurple')
        ))
        fig_live.update_layout(
            title=f"Live Close Price for {target_column}",
            xaxis_title="Time",
            yaxis_title="Price",
            template='plotly_white',
            height=400
        )
        realtime_chart.plotly_chart(fig_live, use_container_width=True)

        time.sleep(1) 

else:
        st.markdown(
        """
        <div style="display:flex; justify-content:center; align-items:center; height:80vh; font-size:35px; font-weight:bold;">
            🔒 Welcome! Please login to access The Stock Market Dashboard.
        </div>
        """,
        unsafe_allow_html=True
    )
