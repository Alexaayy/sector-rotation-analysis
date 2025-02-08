import os
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import datetime
import numpy as np
from nsepython import fnolist

# Set up Streamlit app
st.set_page_config(page_title="Sector Rotation Dashboard", layout="wide")
st.title("📈 Sector Rotation Analysis Dashboard")
st.markdown("### A user-friendly dashboard to analyze sector performance and stock trends in India")


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

# Sidebar for user selection
time_range = st.sidebar.selectbox("Select Time Range", ["1y", "5y", "10y", "max"])
selected_stock = st.sidebar.selectbox("Choose a Stock", stock_list["Ticker"].unique())


# Fetch historical data for selected stock
@st.cache_data
def get_stock_data(ticker, period):
    try:
        data = yf.Ticker(ticker + ".NS").history(period=period)
        if data.empty:
            data = yf.Ticker(ticker + ".BO").history(period=period)  # Try BSE if NSE fails
        if data.empty:
            raise ValueError(f"No historical data found for {ticker}")
        return data
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return pd.DataFrame()


stock_data = get_stock_data(selected_stock, time_range)

# Display stock chart
if not stock_data.empty:
    st.subheader(f"📊 {selected_stock} Stock Performance")
    fig = px.line(stock_data, x=stock_data.index, y='Close', title=f"{selected_stock} Closing Prices")
    st.plotly_chart(fig)

    # Compute Technical Indicators
    stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['SMA_200'] = stock_data['Close'].rolling(window=200).mean()
    stock_data['RSI'] = 100 - (100 / (1 + (
                stock_data['Close'].diff().where(stock_data['Close'].diff() > 0, 0).rolling(14).mean() / (
            -stock_data['Close'].diff().where(stock_data['Close'].diff() < 0, 0).rolling(14).mean()))))

    # Show additional charts
    st.subheader("📈 Moving Averages")
    fig_ma = px.line(stock_data, x=stock_data.index, y=['Close', 'SMA_50', 'SMA_200'],
                     title=f"{selected_stock} with Moving Averages")
    st.plotly_chart(fig_ma)

    st.subheader("📊 Relative Strength Index (RSI)")
    fig_rsi = px.line(stock_data, x=stock_data.index, y='RSI', title=f"{selected_stock} RSI Indicator")
    st.plotly_chart(fig_rsi)

# AI-Based Stock Recommendations
st.subheader("📊 AI-Based Stock Recommendation")


def stock_recommendation(stock_data):
    try:
        if stock_data.empty:
            return "⚠️ No Data Available"
        latest_rsi = stock_data['RSI'].iloc[-1]
        if latest_rsi > 70:
            return "📉 Sell - Overbought Condition Detected (RSI > 70)"
        elif latest_rsi < 30:
            return "📈 Buy - Oversold Condition Detected (RSI < 30)"
        else:
            return "🔄 Hold - No Strong Signal"
    except Exception as e:
        return f"Error in recommendation: {e}"


if not stock_data.empty:
    recommendation = stock_recommendation(stock_data)
    st.markdown(
        f"<div style='padding:10px; border-radius:5px; background-color:#f0f0f0; font-size:18px;'>{recommendation}</div>",
        unsafe_allow_html=True)
