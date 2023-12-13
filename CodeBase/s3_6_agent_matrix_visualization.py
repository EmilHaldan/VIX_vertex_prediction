import numpy as np
import pandas as pd
import os
import sys
import shutil
from tqdm import tqdm
import sqlite3
import matplotlib.pyplot as plt



def load_data(data_entries):
    """
    Check if the simulation was already made
    """

    conn = sqlite3.connect('Trading_simulations/Trading_simulations.db')
    c = conn.cursor()

    c.execute("""SELECT * 
                FROM Simulations AS ts  
                WHERE ts.model_name = ? 
                AND ts.range_type = ?
                AND ts.place_trade_high_threshold > 0.5
                AND ts.place_trade_high_threshold = ts.place_trade_low_threshold 
            """, (data_entries))

    # c.execute("""SELECT 
    #             model_name, specific_model_name, range_type, total_returns,
    #             avg_returns, trades_placed, place_trade_diff_threshold,
    #             place_trade_high_threshold, place_trade_low_threshold,close_trade_take_profit 
    #         FROM 
    #             (SELECT 
    #                 *, MAX(total_returns) AS best_returns
    #             FROM 
    #                 Simulations
    #             WHERE model_name = ?  
    #             AND range_type = ?
    #             GROUP BY 
    #                 place_trade_high_threshold, 
    #                 place_trade_low_threshold
    #             ) AS outerQuery
    #         ORDER BY 
    #             place_trade_high_threshold, place_trade_low_threshold;
    #         """, (data_entries))


    result = c.fetchall()
    result = pd.DataFrame(result, columns = ["model_name", "specific_model_name", "range_type", 
                                            "compounded_returns", "avg_returns","std_returns", "win_rate", "trades_placed", 
                                            "place_trade_diff_threshold", "place_trade_high_threshold", 
                                            "place_trade_low_threshold", "close_trade_take_profit", "score" ])

    return result


def create_matrix(results, x_axis, y_axis, color_axis, model_name, range_type):

    x_values = results[x_axis]
    y_values = results[y_axis]
    colors = results[color_axis]

    matrix_data = pd.pivot_table(results, values=color_axis, index=y_axis, columns=x_axis)

    fig, ax = plt.subplots()
    if color_axis == "avg_returns":
        matrix_data = matrix_data*100
        cax = ax.matshow(matrix_data, cmap='RdYlGn', vmin=-30, vmax=30)
    elif color_axis == "compounded_returns":
        matrix_data = matrix_data*100
        cax = ax.matshow(matrix_data, cmap='RdYlGn', vmin=-100, vmax=100)
    elif color_axis == "trades_placed":
        cax = ax.matshow(matrix_data, cmap='Blues', vmin=0, vmax=20)
    elif color_axis == "std_returns":
        cax = ax.matshow(matrix_data, cmap='Reds', vmin=0, vmax=3)
    elif color_axis == "win_rate":
        cax = ax.matshow(matrix_data, cmap='RdYlGn', vmin=0, vmax=1)
    else:
        raise ValueError("color_axis must be either avg_returns or trades_placed")

    ax.set_title(f"{model_name}")

    ax.set_xlabel("Diff Threshold")
    ax.set_ylabel("High & Low Threshold")

    ax.set_xticks(np.arange(len(matrix_data.columns)))
    ax.set_yticks(np.arange(len(matrix_data.index)))

    ax.set_xticklabels([f"{label:.2f}" for label in matrix_data.columns], rotation=90)
    ax.set_yticklabels([f"{label:.2f}" for label in matrix_data.index])

    for i in range(len(matrix_data.index)):
        for j in range(len(matrix_data.columns)):
            value = matrix_data.iloc[i, j]
            if color_axis == "avg_returns":
                if value >= 25:
                    text = ax.text(j, i, f"{round(value)}%", ha="center", va="center", color="w", size =9)
                else: 
                    text = ax.text(j, i, f"{round(value)}%", ha="center", va="center", color="k", size =9)
            if color_axis == "compounded_returns":
                if value >= 80:
                    text = ax.text(j, i, f"{round(value)}%", ha="center", va="center", color="w", size =9)
                else: 
                    text = ax.text(j, i, f"{round(value)}%", ha="center", va="center", color="k", size =9)
            elif color_axis == "trades_placed":
                if value >= 13:
                    text = ax.text(j, i, int(value), ha="center", va="center", color="w", size =9)
                else: 
                    text = ax.text(j, i, int(value), ha="center", va="center", color="k", size = 9)
            elif color_axis == "std_returns":
                if value >= 10:
                    text = ax.text(j, i, round(value,2), ha="center", va="center", color="w", size =9)
                else: 
                    text = ax.text(j, i, round(value,2), ha="center", va="center", color="k", size =9)
            elif color_axis == "win_rate":
                if value >= 10:
                    text = ax.text(j, i, round(value,2), ha="center", va="center", color="w", size =9)
                else: 
                    text = ax.text(j, i, round(value,2), ha="center", va="center", color="k", size =9)

    if color_axis == "avg_returns":
        cbar = fig.colorbar(cax, label = "Avg Returns pr. Trade")
        fig.suptitle(f"Average Returns pr. Trade    {model_name}", fontsize=14)
    elif color_axis == "trades_placed":
        cbar = fig.colorbar(cax, label = "Trades Placed")
        fig.suptitle(f"Trades Placed    {model_name}", fontsize=14)
    elif color_axis == "std_returns":
        cbar = fig.colorbar(cax, label = "Std Returns pr. Trade")
        fig.suptitle(f"Std Returns pr. Trade    {model_name}", fontsize=14)
    elif color_axis == "compounded_returns":
        cbar = fig.colorbar(cax, label = "Cumulative Returns (%)")
        fig.suptitle(f"Cumulative Returns (%) {model_name}", fontsize=14)

    ax.xaxis.set_ticks_position('bottom')
    ax.invert_yaxis()
    fig.set_size_inches(7, 5)

    plt.savefig(f"Visualizations\Trading_simulations\Matrices\{range_type}\{model_name}_{color_axis}_{range_type}.png", bbox_inches='tight')


if __name__ == "__main__":
    
    for model_name in ["LSTM_OHLC_MSE", "LSTM_all_features_MSE", "LSTM_OHLC_custom_loss", "LSTM_all_features_custom_loss"]:
        
        for range_type in ["val", "test"]:
            data_entries = (model_name, range_type)
            results = load_data(data_entries)
            
            create_matrix(results = results, 
                        y_axis = "place_trade_high_threshold", 
                        x_axis = "place_trade_diff_threshold", 
                        color_axis = "avg_returns",
                        model_name = model_name,
                        range_type = range_type)

            create_matrix(results = results, 
                        y_axis = "place_trade_high_threshold", 
                        x_axis = "place_trade_diff_threshold", 
                        color_axis = "trades_placed",
                        model_name = model_name,
                        range_type = range_type)

            create_matrix(results = results, 
                        y_axis = "place_trade_high_threshold", 
                        x_axis = "place_trade_diff_threshold", 
                        color_axis = "compounded_returns",
                        model_name = model_name,
                        range_type = range_type)
        

        