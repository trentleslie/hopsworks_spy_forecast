import requests
import pandas as pd
import time
from api_key import alpha_api_key
import json

def fetch_price_data(symbol):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize=full&apikey={alpha_api_key}"
    r = requests.get(url)
    data = r.json()
    df_price = pd.DataFrame(data['Time Series (Daily)']).T
    return df_price

def fetch_technical_data(symbol, tech):
    url = f"https://www.alphavantage.co/query?function={tech[0]}&symbol={symbol}&interval=daily&time_period={tech[1]}&series_type=close&apikey={alpha_api_key}"
    r = requests.get(url)
    #print(f"Response content for {symbol} and {tech}:")
    #print(r.content)
    
    try:
        data = r.json()
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON for {symbol} and {tech}: {e}")
        return None
    
    df_tech = pd.DataFrame(data[tech[2]]).T
    
    return df_tech

def main():
    symbol_list = ['SPY','QQQ','XLF','EEM','XLE','SLV','FXI','GDX','EFA','TLT','LQD','XLU','XLV','XLI','IEMG','VWO','XLK','IEF','XLB','JETS','BND']
    tech_list = [['SMA',50,'Technical Analysis: SMA'],
                 ['EMA',21,'Technical Analysis: EMA'],
                 ['RSI',14,'Technical Analysis: RSI']]

    for symbol in symbol_list:
        df_price = fetch_price_data(symbol)
        time.sleep(1)

        for tech in tech_list:
            df_tech = fetch_technical_data(symbol, tech)
            df_price = df_price.merge(df_tech, how='inner', left_index=True, right_index=True)
            time.sleep(1)

        df_price.to_csv(f"../data/raw/{symbol}_daily.csv")
        print(f"{symbol} saved")

if __name__ == "__main__":
    main()
