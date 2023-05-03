import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
import hsml
import hopsworks

def get_stock_data(ticker):
    stock_data = yf.download(ticker, period='100d')
    return stock_data

def calculate_indicators(stock_data):
    sma = SMAIndicator(stock_data['Close'], 50).sma_indicator()
    ema = EMAIndicator(stock_data['Close'], 21).ema_indicator()
    rsi = RSIIndicator(stock_data['Close'], 21).rsi()
    
    stock_data['SMA'] = sma
    stock_data['EMA'] = ema
    stock_data['RSI'] = rsi
    
    return stock_data.tail(21)

def get_model():
    conn = hsml.connection()
    mr = conn.get_model_registry(project="workshop")
    model = mr.get_model("spy_forecasting_model_21", version=1)
    return model

def execute_model(model, data):
    prediction = model.predict(data)
    return prediction

# Streamlit app
st.title("Stock Forecasting App")

ticker = st.selectbox("Select a ticker symbol:", ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"])
forecast_button = st.button("Forecast")

if forecast_button:
    stock_data = get_stock_data(ticker)
    stock_data_indicators = calculate_indicators(stock_data)
    model = get_model()
    prediction = execute_model(model, stock_data_indicators)
    
    if prediction == 0:
        st.write("Down Day")
        st.write("🔻")
        st.write("🔴")
    else:
        st.write("Up Day")
        st.write("🔺")
        st.write("🟢")
