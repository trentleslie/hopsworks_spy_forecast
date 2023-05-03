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

def process_window_sizes(window_sizes, ticker_list):
    for window_size in window_sizes:
        print(f"Starting window size: {window_size}")

        for ticker in ticker_list:
            flattened_df = create_flattened_dataframe(ticker, window_size)
            flattened_df.to_csv(f"../data/interim/{ticker}_{window_size}_flattened.csv")
            print(f"{ticker} {window_size} flattened")

        all_data_combined = combine_flattened_data(window_size)
        all_data_combined.to_csv(f"../data/processed/all_processed_{window_size}.csv")
        print(f"Window {window_size} data combined")
        print(all_data_combined.shape)


def create_flattened_dataframe(ticker, window_size):
    flattened_df = pd.DataFrame()
    df = pd.read_csv(f"../data/interim/{ticker}_{window_size}_processed.csv").drop(['Unnamed: 0'], axis=1)

    for i in range(df.shape[0] - window_size + 1):
        df_window = df.iloc[i:i + window_size,].reset_index(drop=True)
        df_window.index = df_window.index.map(str)
        df_window = df_window.unstack().to_frame().sort_index(level=1).T
        df_window.columns = df_window.columns.map('_'.join)
        flattened_df = pd.concat([flattened_df, df_window], axis=0)

    return flattened_df

def combine_flattened_data(window_size):
    onlyfiles_flattened = [f for f in listdir('../data/interim/') if isfile(join('../data/interim/', f))]
    onlyfiles_flattened = list(filter(lambda thisfilename: f"{window_size}_flattened.csv" in thisfilename, onlyfiles_flattened))

    all_data_combined = pd.DataFrame()

    for filename in onlyfiles_flattened:
        ticker = filename.split('_')[0]
        all_data_combined = pd.concat([all_data_combined, pd.read_csv(f"../data/interim/{ticker}_{window_size}_flattened.csv").drop(['Unnamed: 0'], axis=1)])

    all_data_combined = all_data_combined.drop([f"volume_{window_size-1}", f"rsi_{window_size-1}", f"sma_{window_size-1}", f"ema_{window_size-1}"], axis=1)

    return all_data_combined

# Main execution
ticker_list = process_files(window_sizes)
process_window_sizes(window_sizes, ticker_list)

