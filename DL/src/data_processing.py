import pandas as pd
import datetime
import numpy as np
import yfinance as yf

def download_data(ticker):
    df = yf.download(ticker)
    return df

def preprocess_data(df):
    df = df[['Close']]
    df['Date'] = pd.to_datetime(df.index)
    df.set_index('Date', inplace=True)
    return df

def df_to_windowed_df(dataframe, first_date_str, last_date_str, n=20):
    first_date = pd.to_datetime(first_date_str)
    last_date = pd.to_datetime(last_date_str)
    target_date = first_date
    dates = []
    X, Y = [], []

    while target_date <= last_date:
        df_subset = dataframe.loc[:target_date].tail(n+1)
        if len(df_subset) != n+1:
            target_date += datetime.timedelta(days=1)
            continue

        values = df_subset['Close'].to_numpy()
        x, y = values[:-1], values[-1]
        dates.append(target_date)
        X.append(x)
        Y.append(y)
        target_date += datetime.timedelta(days=1)

    ret_df = pd.DataFrame({})
    ret_df['Target Date'] = dates
    X = np.array(X)
    for i in range(n):
        ret_df[f'Target-{n-i}'] = X[:, i]
    ret_df['Target'] = Y

    return ret_df

def windowed_df_to_date_X_y(windowed_dataframe):
    df_as_np = windowed_dataframe.to_numpy()
    dates = df_as_np[:, 0]
    middle_matrix = df_as_np[:, 1:-1]
    X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))
    Y = df_as_np[:, -1]
    return dates, X.astype(np.float32), Y.astype(np.float32)
