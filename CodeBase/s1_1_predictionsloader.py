import numpy as np
import pandas as pd
import os
import sys
import random as rn
import time
from datetime import datetime
import shutil


def load_target_and_pred(model_name, specific_model_name, range_type = "test"):
    
    specific_model_path = f"ML_saved_models/{model_name}/{specific_model_name}"
    if range_type == "val":
        df = pd.read_csv(f"{specific_model_path}/{specific_model_name}_val_y_pred.csv")
    elif range_type == "test":
        df = pd.read_csv(f"{specific_model_path}/{specific_model_name}_y_pred.csv")
    else: 
        raise ValueError("range_type must be 'val' or 'test'")

    return df


if __name__ == "__main__":
    model_name = "LSTM_all_features_MSE"
    specific_model_name = "LSTM_4" 

    load_target_and_pred(model_name, specific_model_name, range_type = "val")