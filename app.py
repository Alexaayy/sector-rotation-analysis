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
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 Sector Rotation Analysis Dashboard with AI-Based Stock Recommendations")
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

# User selects a sector, time range, and technical indicators
selected_sector = st.sidebar.selectbox("Choose a Sector", list(sectors.keys()))
sector_ticker = sectors[selected_sector]
time_range = st.sidebar.selectbox("Select Time Range", ["1y", "5y", "10y", "max"])
show_sma = st.sidebar.checkbox("Show Moving Averages (SMA 50 & SMA 200)")
show_volatility = st.sidebar.checkbox("Show Volatility (Bollinger Bands)")
show_rsi = st.sidebar.checkbox("Show Relative Strength Index (RSI)")
show_macd = st.sidebar.checkbox("Show MACD (Moving Average Convergence Divergence)")

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

# Display Sector Performance with Interactive Chart
st.subheader(f"📊 {selected_sector} Performance ({time_range})")
if not sector_data.empty:
    fig_sector = go.Figure()
    fig_sector.add_trace(go.Scatter(x=sector_data.index, y=sector_data['Close'], mode='lines', name='Closing Price'))
    if show_sma:
        fig_sector.add_trace(go.Scatter(x=sector_data.index, y=sector_data['SMA_50'], mode='lines', name='SMA 50'))
        fig_sector.add_trace(go.Scatter(x=sector_data.index, y=sector_data['SMA_200'], mode='lines', name='SMA 200'))
    if show_volatility:
        fig_sector.add_trace(go.Scatter(x=sector_data.index, y=sector_data['Upper_BB'], mode='lines', name='Upper Bollinger Band', line=dict(dash='dot')))
        fig_sector.add_trace(go.Scatter(x=sector_data.index, y=sector_data['Lower_BB'], mode='lines', name='Lower Bollinger Band', line=dict(dash='dot')))
    fig_sector.update_layout(title=f"{selected_sector} Sector Performance", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig_sector, use_container_width=True)
    
    # RSI Chart
    if show_rsi:
        st.subheader("📊 Relative Strength Index (RSI)")
        fig_rsi = px.line(sector_data, x=sector_data.index, y='RSI', title="Relative Strength Index (RSI)")
        st.plotly_chart(fig_rsi, use_container_width=True)
    
    # MACD Chart
    if show_macd:
        st.subheader("📊 MACD (Moving Average Convergence Divergence)")
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=sector_data.index, y=sector_data['MACD'], mode='lines', name='MACD'))
        fig_macd.add_trace(go.Scatter(x=sector_data.index, y=sector_data['Signal_Line'], mode='lines', name='Signal Line'))
        fig_macd.update_layout(title="MACD & Signal Line", xaxis_title="Date", yaxis_title="Value")
        st.plotly_chart(fig_macd, use_container_width=True)
