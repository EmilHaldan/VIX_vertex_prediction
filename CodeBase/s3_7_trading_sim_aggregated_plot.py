import numpy as np
import pandas as pd
import os
import sys
import shutil
from tqdm import tqdm
import sqlite3
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

pio.templates.default = "plotly_white"



def load_data(data_entries):
    """
    Check if the simulation was already made
    """

    conn = sqlite3.connect('Trading_simulations/Trading_simulations.db')
    c = conn.cursor()

    c.execute("""SELECT * 
                FROM Simulations AS ts  
                WHERE ts.range_type = ?
            """, (data_entries))

    result = c.fetchall()
    result = pd.DataFrame(result, columns = ["model_name", "specific_model_name", "range_type", 
                                            "compounded_returns", "avg_returns","std_returns", "win_rate", "trades_placed", 
                                            "place_trade_diff_threshold", "place_trade_high_threshold", 
                                            "place_trade_low_threshold", "close_trade_take_profit", "score" ])

    return result




def create_violin_plot(results, range_type, y_axis_1, y_axis_2):
    filtered_data = results[results['range_type'] == range_type]

    fig = make_subplots(rows=1, cols=3, subplot_titles=(y_axis_1, y_axis_2))

    # Adding violin plot for y_axis_1
    for model_name in filtered_data['Model Name'].unique():
        fig.add_trace(go.Violin(x=filtered_data[filtered_data['Model Name'] == model_name]['Model Name'],
                                y=filtered_data[filtered_data['Model Name'] == model_name][y_axis_1],
                                name=model_name,
                                line_color='blue',
                                opacity=0.6,
                                points = False,
                                box_visible=True), row=1, col=1)

    # Adding violin plot for y_axis_2
    for model_name in filtered_data['Model Name'].unique():
        fig.add_trace(go.Violin(x=filtered_data[filtered_data['Model Name'] == model_name]['Model Name'],
                                y=filtered_data[filtered_data['Model Name'] == model_name][y_axis_2],
                                name=model_name,
                                line_color='green',
                                opacity=0.6,
                                points = False,
                                box_visible=True), row=1, col=2)

    # Update layout
    fig.update_traces(meanline_visible=True, points=False)
    fig.update_layout(violingap=0, violinmode='overlay', showlegend=False)
    fig.update_layout(title_text=f'Violin plot for {y_axis_1} and {y_axis_2} by Model Name ({range_type})')

    # setting second plot y-axis be on the right side 
    fig.update_layout(yaxis2=dict(overlaying='y', side='right'))

    # Add axis titles
    fig.update_yaxes(title_text="Cumulative Returns (%)", row=1, col=1)
    fig.update_yaxes(title_text="Average Returns (%)", row=1, col=2)

    fig.write_html(f"Visualizations/Trading_simulations/Violin_plot/{range_type}_violin_plot.html")

    fig.show()


if __name__ == "__main__":
    
        
    for range_type in ["val","test"]:
        data_entries = [range_type]
        results = load_data(data_entries)

        print("results: \n", results)
        print("")
        print("results.columns: \n", results.columns)
        print("")
        print("results.describe(): \n", results.describe())

        results["compounded_returns"] = results["compounded_returns"]*100
        results["avg_returns"] = results["avg_returns"]*100
        results["std_returns"] = results["std_returns"]*100
        results["win_rate"] = results["win_rate"]*100

        # Polishing model names for plot
        results["model_name"] = results["model_name"].replace("LSTM_all_features_MSE", "All, MSE")
        results["model_name"] = results["model_name"].replace("LSTM_OHLC_MSE", "OHLC, MSE")
        results["model_name"] = results["model_name"].replace("LSTM_all_features_custom_loss", "All, Custom Loss")
        results["model_name"] = results["model_name"].replace("LSTM_OHLC_custom_loss", "OHLC, Custom Loss")
        results.rename(columns = {"model_name": "Model Name",
                                  "compounded_returns": "Cumulative Returns",
                                  "avg_returns": "Avg. Returns"}, inplace = True)

        create_violin_plot(results = results, 
                    y_axis_1 = "Cumulative Returns", 
                    y_axis_2 = "Avg. Returns", 
                    range_type = range_type)
        
