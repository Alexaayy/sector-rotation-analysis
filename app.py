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
from transformers import pipeline
import requests
import time

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


# Load full stock list for each sector
@st.cache_data
def load_sector_stocks():
    return pd.read_csv("sector_stocks.csv")  # Ensure this CSV has sector-wise stock data


stock_list = load_sector_stocks()

# User selects a sector, time range, and technical indicators
selected_sector = st.sidebar.selectbox("Choose a Sector", list(sectors.keys()))
sector_ticker = sectors[selected_sector]
time_range = st.sidebar.selectbox("Select Time Range", ["1y", "5y", "10y", "max"])
show_sma = st.sidebar.checkbox("Show Moving Averages (SMA 50 & SMA 200)")
show_volatility = st.sidebar.checkbox("Show Volatility (Bollinger Bands)")
show_rsi = st.sidebar.checkbox("Show Relative Strength Index (RSI)")
show_macd = st.sidebar.checkbox("Show MACD (Moving Average Convergence Divergence)")

# Display list of stocks in selected sector
st.subheader(f"🏢 Stocks in {selected_sector} Sector")
sector_stocks = stock_list[stock_list['Sector'] == selected_sector]
st.dataframe(sector_stocks[['Company Name', 'Ticker']])


# Fetch historical data with user-selected time range
@st.cache_data
def get_sector_data(ticker, period):
    try:
        data = yf.Ticker(ticker).history(period=period)
        data.index = data.index.tz_localize(None)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()


sector_data = get_sector_data(sector_ticker, time_range)

# Compute Technical Indicators
if not sector_data.empty:
    sector_data['SMA_50'] = sector_data['Close'].rolling(window=50).mean()
    sector_data['SMA_200'] = sector_data['Close'].rolling(window=200).mean()
    sector_data['Upper_BB'] = sector_data['Close'].rolling(20).mean() + (2 * sector_data['Close'].rolling(20).std())
    sector_data['Lower_BB'] = sector_data['Close'].rolling(20).mean() - (2 * sector_data['Close'].rolling(20).std())

    # Compute RSI
    delta = sector_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    sector_data['RSI'] = 100 - (100 / (1 + rs))

    # Compute MACD
    short_ema = sector_data['Close'].ewm(span=12, adjust=False).mean()
    long_ema = sector_data['Close'].ewm(span=26, adjust=False).mean()
    sector_data['MACD'] = short_ema - long_ema
    sector_data['Signal_Line'] = sector_data['MACD'].ewm(span=9, adjust=False).mean()

# AI-Based Stock Recommendations
st.subheader("📊 AI-Based Stock Recommendation for Selected Sector")


def stock_recommendation(sector_data):
    try:
        recent_trend = sector_data['Close'].pct_change().rolling(window=20).mean().iloc[-1]
        latest_rsi = sector_data['RSI'].iloc[-1]
        latest_macd = sector_data['MACD'].iloc[-1]
        latest_signal = sector_data['Signal_Line'].iloc[-1]

        if recent_trend > 0 and latest_rsi < 70 and latest_macd > latest_signal:
            return "📈 Strong Buy - Positive Trend & Momentum"
        elif recent_trend > 0 and latest_macd < latest_signal:
            return "🔄 Hold - Uptrend but Weak Momentum"
        elif recent_trend < 0 and latest_rsi > 70:
            return "📉 Sell - Overbought & Downtrend Detected"
        else:
            return "⚠️ Neutral - No Clear Signal"
    except Exception as e:
        return f"Error in recommendation: {e}"


if not sector_data.empty:
    recommendation = stock_recommendation(sector_data)
    st.markdown(
        f"<div style='padding:10px; border-radius:5px; background-color:#f0f0f0; font-size:18px;'>{recommendation}</div>",
        unsafe_allow_html=True)
