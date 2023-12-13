import numpy as np
import pandas as pd
import os
import sys
import random as rn
import time
from datetime import datetime
import shutil

from s1_0_Dataloader import load_specific_data
from s1_1_predictionsloader import load_target_and_pred

import plotly.graph_objs as go
from plotly.subplots import make_subplots


def visualize_targets(df, range_type):
    if range_type not in ["test", "val"]:
        raise ValueError("range_type must be 'test' or 'val'")

    fig = make_subplots(rows=3, cols=1, 
                        shared_xaxes=True, vertical_spacing=0.01, 
                        subplot_titles=('High Target and Prediction', 'Low Target and Prediction', "OHLC Values"))
    
    fig.add_trace(go.Scatter(x=df['date'], y=df['high_y_target'], mode='lines', name='High Target', line=dict(color='#798ffc')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['high_y_pred'], mode='lines', name='High Prediction', line=dict(color='#6af774')), row=1, col=1)

    fig.add_trace(go.Scatter(x=df['date'], y=df['low_y_target'], mode='lines', name='Low Target', line=dict(color='#7993fc')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['low_y_pred'], mode='lines', name='Low Prediction', line=dict(color='#f77168')), row=2, col=1)
    
    ohlc_df = load_specific_data(range_type = range_type, version = 1)["all"]
    ohlc_df = ohlc_df[["Date", "Open", "High", "Low", "Close"]]
    ohlc_df = ohlc_df[ohlc_df["Date"].isin(df["date"])]

    fig.add_trace(go.Candlestick(x=ohlc_df['Date'],
                                open=ohlc_df['Open'],
                                high=ohlc_df['High'],
                                low=ohlc_df['Low'],
                                close=ohlc_df['Close']) , row=3, col=1)

    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    
    fig.update_yaxes(title_text="High Value", row=1, col=1)
    fig.update_yaxes(title_text="Low Value", row=2, col=1)
    fig.update_yaxes(title_text="VIX Price", row=3, col=1, autorange=True, fixedrange=False)
    
    fig.update_layout(xaxis1=dict(showticklabels=False),
                        xaxis2=dict(showticklabels=False),
                        template="plotly_white",
                        title_text=f"Targets and Predictions Over Time    {specific_model_name}    {model_name} ",
                        xaxis3_rangeslider_visible=True, 
                        xaxis3_rangeslider_thickness=0.05)

    if not os.path.exists(f"Visualizations/Predictions_vs_targets"):
        os.makedirs(f"Visualizations/Predictions_vs_targets")

    fig.write_html(f"Visualizations/Predictions_vs_targets/{model_name}_{specific_model_name}_{range_type}_targets_and_predictions.html")

    fig.show()


if __name__ == "__main__":
        
    for range_type in ["val", "test"]:
        for i in range(4):
            model_name = model_names[i]
            specific_model_name = specific_model_names[i]


            df = load_target_and_pred(model_name, specific_model_name, range_type = range_type)

            visualize_targets(df, range_type = range_type)