import os
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pandas_datareader.data as web
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import time
from nsepython import fnolist

# Disable GPU to prevent CUDA errors on Streamlit Cloud
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Set up Streamlit app with improved layout
st.set_page_config(page_title="Sector Rotation Dashboard", layout="wide")
st.markdown("""
    <style>
    body {
        background-color: #f5f5f5;
    }
    .stApp {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        font-family: Arial, sans-serif;
    }
    h1, h2, h3 {
        text-align: center;
    }
    .stDataFrame {
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 Sector Rotation Analysis Dashboard with AI-Based Stock Recommendations")
st.markdown("### A beginner-friendly dashboard to understand sector performance and stock trends")

st.sidebar.header("Sector Selection")


# Fetch live stock data from NSE
@st.cache_data
def fetch_nse_stocks():
    try:
        stock_symbols = fnolist()
        stock_list = pd.DataFrame({"Ticker": stock_symbols})
        return stock_list
    except Exception as e:
        st.error(f"Error fetching NSE stock data: {e}")
        return pd.DataFrame()


stock_list = fetch_nse_stocks()

# Ensure stock list is not empty before proceeding
if stock_list.empty:
    st.error("⚠️ No stock data available. Please check NSE API connectivity.")
    st.stop()

# User selects a stock
time_range = st.sidebar.selectbox("Select Time Range", ["1y", "5y", "10y", "max"])
selected_stock = st.sidebar.selectbox("Choose a Stock", stock_list["Ticker"].unique())


# Fetch historical data for selected stock
@st.cache_data
def get_stock_data(ticker, period):
    try:
        data = yf.Ticker(ticker).history(period=period)
        if data.empty:
            data = yf.Ticker(ticker + ".BO").history(period=period)  # Try BSE if NSE fails
        if data.empty:
            raise ValueError(f"No historical data found for {ticker}")
        data.index = data.index.tz_localize(None)
        return data
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return pd.DataFrame()


stock_data = get_stock_data(selected_stock, time_range)

# Compute Technical Indicators
if not stock_data.empty:
    stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['SMA_200'] = stock_data['Close'].rolling(window=200).mean()
    stock_data['Upper_BB'] = stock_data['Close'].rolling(20).mean() + (2 * stock_data['Close'].rolling(20).std())
    stock_data['Lower_BB'] = stock_data['Close'].rolling(20).mean() - (2 * stock_data['Close'].rolling(20).std())

    # Compute RSI
    delta = stock_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    stock_data['RSI'] = 100 - (100 / (1 + rs))

    # Compute MACD
    short_ema = stock_data['Close'].ewm(span=12, adjust=False).mean()
    long_ema = stock_data['Close'].ewm(span=26, adjust=False).mean()
    stock_data['MACD'] = short_ema - long_ema
    stock_data['Signal_Line'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()

# AI-Based Stock Recommendations
st.subheader("📊 AI-Based Stock Recommendation for Selected Stock")


def stock_recommendation(stock_data):
    try:
        if stock_data.empty:
            return "⚠️ No Data Available", []

        recent_trend = stock_data['Close'].pct_change().rolling(window=20).mean().iloc[-1]
        latest_rsi = stock_data['RSI'].iloc[-1]
        latest_macd = stock_data['MACD'].iloc[-1]
        latest_signal = stock_data['Signal_Line'].iloc[-1]
        alerts = []

        if recent_trend > 0 and latest_rsi < 70 and latest_macd > latest_signal:
            recommendation = "📈 Strong Buy - Positive Trend & Momentum"
        elif recent_trend > 0 and latest_macd < latest_signal:
            recommendation = "🔄 Hold - Uptrend but Weak Momentum"
        elif recent_trend < 0 and latest_rsi > 70:
            recommendation = "📉 Sell - Overbought & Downtrend Detected"
        else:
            recommendation = "⚠️ Neutral - No Clear Signal"

        if latest_rsi > 70:
            alerts.append("⚠️ RSI Alert: Overbought Condition Detected (RSI > 70)")
        elif latest_rsi < 30:
            alerts.append("⚠️ RSI Alert: Oversold Condition Detected (RSI < 30)")

        if latest_macd > latest_signal:
            alerts.append("📊 MACD Alert: Bullish Signal - MACD above Signal Line")
        elif latest_macd < latest_signal:
            alerts.append("📊 MACD Alert: Bearish Signal - MACD below Signal Line")

        return recommendation, alerts
    except Exception as e:
        return f"Error in recommendation: {e}", []


if not stock_data.empty:
    recommendation, alerts = stock_recommendation(stock_data)
    st.markdown(
        f"<div style='padding:10px; border-radius:5px; background-color:#f0f0f0; font-size:18px;'>{recommendation}</div>",
        unsafe_allow_html=True)

    if alerts:
        st.subheader("🚨 Stock Alerts")
        for alert in alerts:
            st.write(alert)
