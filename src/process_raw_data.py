import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join

window_sizes = [21]

def zscore(x, window):
    r = x.rolling(window=window)
    m = r.mean().shift(1)
    s = r.std(ddof=0).shift(1)
    z = (x - m) / s
    return z

def process_dataframe(df, window_size):
    df.sort_index(inplace=True, ascending=False)
    columns_to_drop = ['Unnamed: 0', '5. adjusted close', '7. dividend amount', '8. split coefficient', 'SMA', 'EMA']
    df.drop([col for col in columns_to_drop if col in df.columns], axis=1, inplace=True)
    df.columns = ['open', 'high', 'low', 'close', 'volume', 'rsi']
    df['sma'] = df.iloc[:, 3].rolling(window=50).mean() / df.iloc[:, 3]
    df['ema'] = df.iloc[:, 3].ewm(span=21).mean() / df.iloc[:, 3]
    df = df.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    df = zscore(df, window=window_size).dropna()
    return df

def process_files(window_sizes):
    raw_files = [f for f in listdir('../data/raw/') if isfile(join('../data/raw/', f)) and '_daily.csv' in f]
    ticker_list = [filename.split('_')[0] for filename in raw_files]
    
    for window_size in window_sizes:
        ticker_stats_mean = pd.DataFrame()
        ticker_stats_std = pd.DataFrame()

        for filename in raw_files:
            ticker = filename.split('_')[0]
            df = pd.read_csv(f"../data/raw/{ticker}_daily.csv")
            df = process_dataframe(df, window_size)
            ticker_stats_mean, ticker_stats_std = update_ticker_stats(df, ticker, ticker_stats_mean, ticker_stats_std, window_size)
            save_processed_files(df, ticker, window_size)

        plot_histograms(ticker_stats_mean, ticker_stats_std, window_size)
        ticker_list = set(ticker_list)

    return ticker_list

def update_ticker_stats(df, ticker, ticker_stats_mean, ticker_stats_std, window_size):
    temp_mean = pd.DataFrame(df.describe()).iloc[1:2,]
    temp_mean['ticker'] = ticker
    ticker_stats_mean = pd.concat([ticker_stats_mean, temp_mean])
    ticker_stats_mean.to_csv(f"../data/interim/ticker_stats_mean_{window_size}.csv")

    temp_std = pd.DataFrame(df.describe()).iloc[2:3,]
    temp_std['ticker'] = ticker
    ticker_stats_std = pd.concat([ticker_stats_std, temp_std])
    ticker_stats_std.to_csv(f"../data/interim/ticker_stats_std_{window_size}.csv")

    return ticker_stats_mean, ticker_stats_std

def save_processed_files(df, ticker, window_size):
    df.to_csv(f"../data/interim/{ticker}_{window_size}_processed.csv")
    df.describe().to_csv(f"../data/interim/{ticker}_{window_size}_describe.csv")

def plot_histograms(ticker_stats_mean, ticker_stats_std, window_size):
    print(f"{window_size} z-score mean histograms")
    ticker_stats_mean.hist(column='open', bins=10)
    ticker_stats_mean.hist(column='high', bins=10)
    ticker_stats_mean.hist(column='low', bins=10)
    ticker_stats_mean.hist(column='close', bins=10)
    ticker_stats_mean.hist(column='volume', bins=10)
    ticker_stats_mean.hist(column='rsi', bins=10)
    ticker_stats_mean.hist(column='sma', bins=10)
    ticker_stats_mean.hist(column='ema', bins=10)

    print(f"{window_size} z-score std histograms")
    ticker_stats_std.hist(column='open', bins=10)
    ticker_stats_std.hist(column='high', bins=10)
    ticker_stats_std.hist(column='low', bins=10)
    ticker_stats_std.hist(column='close', bins=10)
    ticker_stats_std.hist(column='volume', bins=10)
    ticker_stats_std.hist(column='rsi', bins=10)
    ticker_stats_std.hist(column='sma', bins=10)
    ticker_stats_std.hist(column='ema', bins=10)

# Main execution
ticker_list = process_files(window_sizes)
