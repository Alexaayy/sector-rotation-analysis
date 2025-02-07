import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import pandas_datareader.data as web
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Set up Streamlit app
st.title("📈 Sector Rotation Analysis Dashboard with ML Predictions")
st.sidebar.header("Sector Selection")

# Define sector ETFs
sectors = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Consumer Staples": "XLP"
}

# User selects a sector
selected_sector = st.sidebar.selectbox("Choose a Sector", list(sectors.keys()))
sector_ticker = sectors[selected_sector]

# Fetch historical data
@st.cache_data
def get_sector_data(ticker):
    try:
        data = yf.Ticker(ticker).history(period="5y")
        data.index = data.index.tz_localize(None)  # Convert timezone-aware index to naive
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

sector_data = get_sector_data(sector_ticker)

if not sector_data.empty:
    # Calculate daily returns
    sector_data['Returns'] = sector_data['Close'].pct_change()

    # Plot sector performance
    st.subheader(f"{selected_sector} Sector Performance")
    fig = px.line(sector_data, x=sector_data.index, y='Close', title=f"{selected_sector} Closing Prices")
    st.plotly_chart(fig)

    # Prepare data for ML model
    def prepare_data(data, lookback=30):
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
        
        X, Y = [], []
        for i in range(len(scaled_data) - lookback):
            X.append(scaled_data[i:i+lookback])
            Y.append(scaled_data[i+lookback])
        
        return np.array(X), np.array(Y), scaler

    X_train, Y_train, scaler = prepare_data(sector_data['Close'].dropna())

    # Train LSTM Model
    def train_lstm(X_train, Y_train):
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, Y_train, epochs=5, batch_size=16, verbose=1)
        return model

    model = train_lstm(X_train, Y_train)

    # Predict future prices
    def predict_future(model, last_data, scaler, days=30):
        predictions = []
        last_window = last_data[-30:].reshape(1, 30, 1)

        for _ in range(days):
            pred = model.predict(last_window)
            predictions.append(pred[0, 0])
            last_window = np.roll(last_window, -1, axis=1)
            last_window[0, -1, 0] = pred

        return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    future_prices = predict_future(model, X_train, scaler, days=30)
    
    # Create future dates
    future_dates = pd.date_range(start=sector_data.index[-1], periods=30, freq='D')

    # Display predictions
    st.subheader(f"📊 Predicted Prices for {selected_sector} (Next 30 Days)")
    pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_prices.flatten()})
    fig_pred = px.line(pred_df, x='Date', y='Predicted Price', title="ML-Based Sector Price Prediction")
    st.plotly_chart(fig_pred)
