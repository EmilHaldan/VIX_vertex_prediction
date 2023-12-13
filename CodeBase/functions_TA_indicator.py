from datetime import datetime
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import os
import talib


def aroon(df, timeperiod=14):
    aroon_down, aroon_up = talib.AROON(df['High'], df['Low'], timeperiod=timeperiod)
    new_df = pd.DataFrame()
    new_df['Date'] = df['Date']
    new_df[f'AROON_down_{timeperiod}'] = aroon_down
    new_df[f'AROON_up_{timeperiod}'] = aroon_up
    
    return new_df

def bollinger_bands(df, timeperiod=20, nbdevup=2, nbdevdn=2):
    upper, middle, lower = talib.BBANDS(df['Close'], timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn)
    new_df = pd.DataFrame()
    new_df['Date'] = df['Date']
    new_df[f'BB_upper_{timeperiod}'] = upper
    new_df[f'BB_middle_{timeperiod}'] = middle
    new_df[f'BB_lower_{timeperiod}'] = lower
    return new_df

def ema(df, timeperiod=14):
    new_df = pd.DataFrame()
    new_df['Date'] = df['Date']
    new_df[f'EMA_{timeperiod}'] = talib.EMA(df['Close'], timeperiod=timeperiod)
    return new_df

def macd(df, timeperiod=9):

    fastperiod=round(timeperiod*1.333) # usually static at 12
    slowperiod=round(timeperiod*2.888) # usually static at 26
    signalperiod=timeperiod          # usually static at 9

    macd, signal, hist = talib.MACD(df['Close'], fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
    new_df = pd.DataFrame()
    new_df['Date'] = df['Date']
    new_df[f'MACD_{timeperiod}'] = macd
    new_df[f'MACD_signal_{timeperiod}'] = signal
    new_df[f'MACD_hist_{timeperiod}'] = hist
    return new_df

def roc(df, timeperiod=10):
    new_df = pd.DataFrame()
    new_df['Date'] = df['Date']
    new_df[f'ROC_{timeperiod}'] = talib.ROC(df['Close'], timeperiod=timeperiod)

    return new_df

def rsi(df, timeperiod=14):
    new_df = pd.DataFrame()
    new_df['Date'] = df['Date']
    new_df[f'RSI_{timeperiod}'] = talib.RSI(df['Close'], timeperiod=timeperiod)
    return new_df

def max_high(df, timeperiod=14):
    new_df = pd.DataFrame()
    new_df['Date'] = df['Date']
    new_df[f'Max_High_{timeperiod}'] = talib.MAX(df['High'], timeperiod=timeperiod)

    return new_df

def min_low(df, timeperiod=14):
    new_df = pd.DataFrame()
    new_df['Date'] = df['Date']
    new_df[f'Min_Low_{timeperiod}'] = talib.MIN(df['Low'], timeperiod=timeperiod)

    return new_df

def high_low_diff(df, timeperiod=14):
    new_df = pd.DataFrame()
    new_df['Date'] = df['Date']
    new_df[f'High_Low_Diff_{timeperiod}'] = talib.MAX(df['High'], timeperiod=timeperiod) - talib.MIN(df['Low'], timeperiod=timeperiod)

    return new_df

def high_low_mean(df, timeperiod=14):
    new_df = pd.DataFrame()
    new_df['Date'] = df['Date']
    new_df[f'High_Low_Mean_{timeperiod}'] = (talib.MAX(df['High'], timeperiod=timeperiod) + talib.MIN(df['Low'], timeperiod=timeperiod))/2

    return new_df




if __name__ == '__main__':
    
    # create tests of functions
    df = pd.read_csv('Data/VIX_OHLC_2003_2023.csv')
    print("head of original df")
    print(df.head())
    print("\n tail of original df")
    print(df.tail())
    print("\n"*2)

    tmp_df = aroon(df)
    print("head of aroon")
    print(tmp_df.head())
    print("\n tail of aroon")
    print(tmp_df.tail())
    print("\n"*2)

    tmp_df = bollinger_bands(df)
    print("head of bollinger_bands")
    print(tmp_df.head())
    print("\n tail of bollinger_bands")
    print(tmp_df.tail())
    print("\n"*2)

    tmp_df = ema(df)
    print("head of ema")
    print(tmp_df.head())
    print("\n tail of ema")
    print(tmp_df.tail())
    print("\n"*2)
    
    tmp_df = macd(df)
    print("head of macd")
    print(tmp_df.head())
    print("\n tail of macd")
    print(tmp_df.tail())
    print("\n"*2)

    tmp_df = roc(df)
    print("head of roc")
    print(tmp_df.head())
    print("\n tail of roc")
    print(tmp_df.tail())
    print("\n"*2)

    tmp_df = rsi(df)
    print("head of rsi")
    print(tmp_df.head())
    print("\n tail of rsi")
    print(tmp_df.tail())
    print("\n"*2)
