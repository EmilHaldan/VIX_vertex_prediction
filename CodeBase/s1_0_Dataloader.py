# Package import
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import talib
import pickle
from tqdm import tqdm
import copy
import warnings

warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', None)

from functions_TA_indicator import *


def load_pickled_dict(version = 4):
    with open(f"Data/Transformed_Data/datasets_v{version}.pickle", "rb") as f:
        ml_dataset = pickle.load(f)
    return ml_dataset

def load_specific_data(range_type = "all",version = 4):
    if range_type in ["all", "train", "test", "val"]:
        return load_pickled_dict(version)[f"{range_type}"]
    else: 
        raise ValueError("range_type must be 'all', 'train', 'val' or 'test'")

def load_features(data):
    return data["all"]

def load_target(data):
    return data["target"][["Date", "High_Target", "Low_Target"]]

def load_dates(range_type = "all"):
    data = load_specific_data(range_type = range_type, version = 4)
    
    X_dates = data["all"]["Date"].to_list()
    y_dates = data["target"]["Date"].to_list()

    if X_dates != y_dates:
        raise ValueError("Dates of X and y are different")
    return X_dates


def get_window_and_feature_space(LSTM_X_data, LSTM_y_data):
    window_size = len(LSTM_X_data[0])
    feature_space = len(LSTM_X_data[0][0])
    target_length = len(LSTM_y_data[0])
    return window_size, feature_space, target_length


def load_lstm_data(window_size = 20, range_type = "train", version = 4, verbose = False, features = "all"):
    """

    Parameters
    ----------
    window_size : int, optional
        DESCRIPTION. The default is 20.
    range_type : str, optional
        DESCRIPTION. The default is "train".
    version : int, optional
        DESCRIPTION. The default and maximum is 4.
    verbose : bool, optional
        DESCRIPTION. The default is False.
    features : str, optional
        DESCRIPTION. The default is "all". Other option is "OHLC".

    Returns 
    -------
    LSTM_X_data : numpy array
        DESCRIPTION. list for feature data in the shape of (len(LSTM_X_data), window_size, feature_space)
    LSTM_y_data : numpy array
        DESCRIPTION. list for target data
    y_dates : list 
        DESCRIPTION. list of dates corresponding to the y data
    """
    data = load_specific_data(range_type = range_type, version = version)

    feature_data = load_features(data).drop(columns = ["Date"])
    target_data = load_target(data).drop(columns = ["Date"])
    date_data = load_dates(range_type = range_type)

    if features == "OHLC":
        feature_data = feature_data[["Open", "High", "Low", "Close"]]

    if verbose:
        print("feature_data[0:2]: "  , feature_data[0:2])
        print("feature_data[-3:]: ", feature_data[-3:])
        print("")
        print("target_data[0:2]: "  , target_data[0:2])
        print("target_data[-3:]: ", target_data[-3:])
        print("")
        print("date_date[0:2]: "  , date_data[0:2])
        print("date_date[-3:]: ", date_data[-3:])
        print("\n"*2)

    LSTM_X_list = []
    LSTM_y_list =  []
    y_dates = []

    for i in tqdm(range(window_size-1, len(feature_data))):
        LSTM_X_list.append(feature_data.loc[i-window_size+1:i])
        LSTM_y_list.append(target_data.loc[i])
        y_dates.append(date_data[i])

    LSTM_X_data = np.array(LSTM_X_list)
    LSTM_y_data = np.array(LSTM_y_list)

    if verbose:
        print("date_data len: ", len(y_dates))
        print("LSTM_X    len: ", len(LSTM_X_data))
        print("LSTM_y    len: ", len(LSTM_y_data))
        print("\n")

        for i in range(3):
            print(f"date_data[{i}]: ", y_dates[i])
            print(f"LSTM_X[{i}]: ", LSTM_X_data[i],"\n")
            print(f"LSTM_y[{i}]: ", LSTM_y_data[i])
            print("\n")

    return LSTM_X_data, LSTM_y_data, y_dates



if __name__ == "__main__":
    # X, y, dates = load_lstm_data(window_size=1, range_type = "val" ,version = 4, verbose=True)
    
    # print("X.shape: ", X.shape)
    # print("y.shape: ", y.shape)
    # print("dates.shape: ", len(dates))
    # print("dates start: ", dates[0])
    # print("dates end  : ", dates[-1])

    # pd.set_option('display.float_format', lambda x: '%.2f' % x)
    df = load_specific_data(range_type = "test", version = 3)["all"]
    # df = df[["Date", "Open", "High", "Low", "Close"]]
    print("shape df: ", df.shape)
    
    # df["Date"] = pd.to_datetime(df["Date"])

    # fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    # fig.add_trace(go.Candlestick(x=df['Date'],
    #                 open=df['Open'],
    #                 high=df['High'],
    #                 low=df['Low'],
    #                 close=df['Close']), row=1, col=1)

    
    # fig.update_layout(
    #     xaxis_rangeslider_visible=False,
    #     xaxis=dict(
    #         type='date',
    #         tick0="2003-01-01",
    #         dtick="M12",
    #         tickformat="%Y",
    #         tickangle=45,
    #     )
    # )
    # fig.show()