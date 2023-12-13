
import numpy as np
import pandas as pd
import os
import sys
import random as rn
from collections import OrderedDict
import time
from datetime import datetime
import shutil

from s1_0_Dataloader import load_specific_data



def process_df(df, high_threshold, low_threshold, predictions_df, range_type):
    global val_target, test_target

    target_df = val_target if range_type == "val" else test_target

    for index, row in df.iterrows():
        high_y_pred = 1 if row['Close'] > high_threshold else 0
        low_y_pred = 1 if row['Close'] < low_threshold else 0
        new_row = {
            'date': row['Date'],
            'high_y_target': target_df.loc[index,'High_Target'],
            'high_y_pred': high_y_pred,
            'low_y_target': target_df.loc[index,'Low_Target'],
            'low_y_pred': low_y_pred
        }
        predictions_df.loc[index] = new_row

    return predictions_df


def create_predictions(val_df, test_df, high_threshold, low_threshold):
    # This function creates the predictions for the baseline model in a similar format to that of the LSTM models, 
    # so that the evaluation functions can be reused.

    columns = ['date', 'high_y_target', 'high_y_pred', 'low_y_target', 'low_y_pred']
    predictions_df = pd.DataFrame(columns=columns)
    predictions_df = process_df(val_df, high_threshold, low_threshold, predictions_df, "val")
    predictions_df.to_csv("ML_saved_models/Baseline/Baseline/Baseline_val_y_pred.csv", index=False)

    columns = ['date', 'high_y_target', 'high_y_pred', 'low_y_target', 'low_y_pred']
    predictions_df = pd.DataFrame(columns=columns)
    predictions_df = process_df(test_df, high_threshold, low_threshold, predictions_df, "test")
    predictions_df.to_csv("ML_saved_models/Baseline/Baseline/Baseline_y_pred.csv", index=False)


if __name__ == "__main__":
    
    train_df = load_specific_data(range_type = "train", version = 3)
    val_df   = load_specific_data(range_type = "val", version = 3)
    test_df  = load_specific_data(range_type = "test", version = 3)

    val_target  = val_df["target"]
    test_target = test_df["target"]

    train_OHLC = train_df["all"][["Date","Open", "High", "Low", "Close"]]
    val_OHLC   = val_df["all"][["Date","Open", "High", "Low", "Close"]]
    test_OHLC  = test_df["all"][["Date","Open", "High", "Low", "Close"]]


    print("val_target.describe():", val_target.describe())
    print("")
    print("test_target.describe():", test_target.describe())
    print("")
    print("train_OHLC.describe():", train_OHLC.describe())
    print("")
    print("val_OHLC.describe():", val_OHLC.describe())
    print("")
    print("test_OHLC.describe():", test_OHLC.describe())
    print("")

    high_threshold = val_OHLC.describe()["High"]["75%"]
    low_threshold  = val_OHLC.describe()["Low"]["25%"]

    create_predictions(val_OHLC, test_OHLC, high_threshold, low_threshold)

    