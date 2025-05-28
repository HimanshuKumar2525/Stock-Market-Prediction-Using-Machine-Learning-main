# model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
from model import load_data, preprocess_data, train_model, predict_future_prices, calculate_rmse

# Load your data
data = load_data('data/stock_data.csv')

# Preprocess data for target column, e.g. 'Close'
X, y, scaler = preprocess_data(data, target_column='Close')

# Split into train and validation sets (keep your old code logic)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train LSTM model using the imported function
model = train_model(X_train, y_train, X_val, y_val)

# Predict future prices, for example next 7 days
last_60_days = X[-1].flatten()  # get the last sequence from data
future_preds = predict_future_prices(model, last_60_days, days=7)

# Calculate validation RMSE
val_preds = model.predict(X_val).flatten()
rmse_val = calculate_rmse(y_val, val_preds)
print(f"Validation RMSE: {rmse_val}")


# Load data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Preprocess data
def preprocess_data(data, target_column):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[[target_column]])
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X, y, scaler

# Build LSTM model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train model
def train_model(X_train, y_train, X_val, y_val):
    model = build_lstm_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
    return model

# Predict future prices
def predict_future_prices(model, last_60_days, days=7):
    predictions = []
    for _ in range(days):
        pred_input = last_60_days.reshape((1, last_60_days.shape[0], 1))
        pred_price = model.predict(pred_input)[0]
        predictions.append(pred_price)
        last_60_days = np.append(last_60_days[1:], pred_price)
    return np.array(predictions)

# Calculate RMSE
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))













