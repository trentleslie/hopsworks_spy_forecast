import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
import hsml
import hopsworks
from joblib import load

def get_model_input(ticker, window_size=21):
    def calculate_indicators(stock_data, window_size=21):
        sma = SMAIndicator(stock_data['close'], 50).sma_indicator() / stock_data['close']
        ema = EMAIndicator(stock_data['close'], 21).ema_indicator() / stock_data['close']
        rsi = RSIIndicator(stock_data['close'], 21).rsi()
        
        stock_data['sma'] = sma
        stock_data['ema'] = ema
        stock_data['rsi'] = rsi
        
        return stock_data.tail(2 * window_size - 1)
    
    def zscore(x, window):
        r = x.rolling(window=window)
        m = r.mean().shift(1)
        s = r.std(ddof=0).shift(1)
        z = (x - m) / s
        return z
    
    def create_flattened_dataframe(df, window_size=21):
        flattened_df = pd.DataFrame()
        
        for i in range(df.shape[0] - window_size + 1):
            df_window = df.iloc[i:i + window_size,].reset_index(drop=True)
            df_window.index = df_window.index.map(str)
            df_window = df_window.unstack().to_frame().sort_index(level=1).T
            df_window.columns = df_window.columns.map('_'.join)
            flattened_df = pd.concat([flattened_df, df_window], axis=0)

        return flattened_df
    
    days_back = 2 * window_size + 50
    stock_data = yf.download(ticker, period=f'{days_back}d')
    
    #stock_data.sort_index(inplace=True, ascending=False)
    
    # Check if stock_data is a pandas DataFrame and convert it if necessary
    if not isinstance(stock_data, pd.DataFrame):
        stock_data = pd.DataFrame(stock_data)
        
    # Drop the 'Adj Close' column if it exists
    if 'Adj Close' in stock_data.columns:
        stock_data = stock_data.drop(columns=['Adj Close'])
        
    # Convert all column names to lowercase
    stock_data = stock_data.rename(columns={col: col.lower() for col in stock_data.columns})
    
    stock_data = stock_data.pct_change().replace([np.inf, -np.inf], np.nan) #.dropna() ???
    #stock_data = zscore(stock_data, window=window_size) #.dropna() ???
    
    return create_flattened_dataframe(zscore(calculate_indicators(stock_data, window_size), 
                  window=window_size).dropna().sort_index(inplace=False, ascending=False),
                                      window_size=window_size-1), stock_data.index.max().date()

def get_hopsworks_model(): #not working
    conn = hsml.connection()
    mr = conn.get_model_registry(project="workshop")
    model = mr.get_model("spy_forecasting_model_5", version=1)
    return model

def get_local_model(model_path):
    return load(model_path)

def execute_model(model, data):
    prediction = model.predict(data)
    if len(prediction) == 1:
        return prediction[0]
    else:
        return prediction

# Streamlit app
st.title("Stock Forecasting App")

ticker = st.selectbox("Select a ticker symbol:", ["SPY", "AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"])
forecast_button = st.button("Forecast")

if forecast_button:
    model_input, last_date = get_model_input(ticker, 5)
    model = get_local_model("/home/trent/github/hopsworks_spy_forecast/spy_forecasting_model/spy_forecasting_model.pkl")
    prediction = execute_model(model, model_input)
    
    st.write(f"Last date of available data: {last_date}, this forecast is for the next trading day.")
    
    if prediction == 0:
        st.write("🔴 The next trading day is forecasted to be a DOWN day.")
    else:
        st.write("🟢 The next trading day is forecasted to be an UP day.")
