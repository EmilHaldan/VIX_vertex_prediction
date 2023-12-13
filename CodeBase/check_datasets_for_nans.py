# Package import
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import os
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import talib
from plotly.subplots import make_subplots
import pickle
from tqdm import tqdm
import copy
import warnings

warnings.filterwarnings("ignore")


pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', None)

from functions_TA_indicator import *


if __name__ == "__main__":

    
    i = 4
    with open(f"Data/Transformed_Data/datasets_v{i}.pickle", "rb") as f:
        ml_dataset = pickle.load(f)

    print(f" ### Dataset version {i} ###")
    for key in ml_dataset.keys():
        print(f"Dataset {key}    keys: {ml_dataset[key].keys()}")

        print("key: ", key)
        
        for key_2 in ml_dataset[key]:
            if ml_dataset[key][key_2].isna().sum().sum() > 0:
                print("")
                print(f"NaN values found in {key}{key_2}")
                print(ml_dataset[key][key_2].isna().sum())
            else: 
                print(f"No NaN values found in {key} {key_2}")
            total_len = len(ml_dataset[key][key_2])
            print(f"First date: {ml_dataset[key][key_2].loc[0,'Date']}")
            print(f"Last date: {ml_dataset[key][key_2].loc[total_len-1,'Date']}")
            print("")
                





   

