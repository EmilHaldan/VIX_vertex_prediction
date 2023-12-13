import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import time
import sqlite3

from s0_5_synthesize_VIX_option_prices import get_wednesdays_closest_to_the_20th, get_options_prices
from s1_0_Dataloader import *
from s1_1_predictionsloader import *
from s0_6_options_util import *
# from s3_2_1_simulate_options_trades import *
from s3_2_0_simulate_trades import *


def bulk_load_simulations(data_entries):
    conn = sqlite3.connect('Trading_simulations/Trading_simulations.db')
    c = conn.cursor()

    # Create table if it doesn't exist
    c.execute('''CREATE TABLE IF NOT EXISTS Simulations 
                 (model_name TEXT, 
                specific_model_name TEXT, 
                range_type TEXT, 
                compounded_returns REAL, 
                avg_returns REAL, 
                std_returns REAL,
                win_rate REAL,
                trades_placed REAL,
                place_trade_diff_threshold REAL,
                place_trade_high_threshold REAL,
                place_trade_low_threshold REAL,
                close_trade_take_profit REAL,
                score REAL)''')

    # Bulk insert
    c.executemany("INSERT INTO Simulations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", data_entries)

    conn.commit()
    conn.close()



def check_if_sim_was_made(data_entries):
    """
    Check if the simulation was already made
    """

    try:
        conn = sqlite3.connect('Trading_simulations/Trading_simulations.db')
        c = conn.cursor()

        c.execute("""SELECT * 
                    FROM Simulations AS ts  
                    WHERE ts.model_name = ? 
                    AND ts.specific_model_name = ? 
                    AND ts.range_type = ? 
                    AND ts.place_trade_diff_threshold = ? 
                    AND ts.place_trade_high_threshold = ? 
                    AND ts.place_trade_low_threshold = ? 
                    AND ts.close_trade_take_profit = ? ;
                """, (data_entries))

    except sqlite3.OperationalError:
        # Database or table doesn't exist
        return False

    result = c.fetchall()
    conn.close()

    if len(result) == 0:
        return False
    else:
        return True


def calc_time_remaining(start_time, current_job_count, total_jobs_to_do, length_of_combinations):
    """
    Calculate the time remaining for the simulation to finish
    """
    time_elapsed = time.time() - start_time
    time_per_index = time_elapsed/current_job_count
    time_remaining = time_per_index*(total_jobs_to_do-current_job_count)

    print(f"Time elapsed     : {round(time_elapsed%216000//3600)} hours  {round(time_elapsed%3600//60)} minutes {round(time_elapsed%60)} seconds")
    print(f"Time remaining   : {round(time_remaining%216000//3600)} hours  {round(time_remaining%3600//60)} minutes {round(time_remaining%60)} seconds")
    print(f"Jobs remaining   : {total_jobs_to_do-current_job_count} out of {length_of_combinations}")
    print(f"Job Average time : {round(time_per_index%60,3)} seconds\n")



if __name__ == "__main__":
    
    # model_name = "LSTM_all_features_custom_loss"
    # specific_model_name = "LSTM_78"
    model_names = ["LSTM_all_features_custom_loss","LSTM_OHLC_custom_loss","LSTM_all_features_MSE", "LSTM_OHLC_MSE"]
    specific_model_names = [ "LSTM_78","LSTM_17", "LSTM_70" , "LSTM_46" ]
    for range_type in ["val", "test"]:
            for i in range(4):

                model_name = model_names[i]
                specific_model_name = specific_model_names[i]
                try:
                    merged_df = prepare_dataframe(model_name, specific_model_name, range_type)
                except FileNotFoundError:
                    continue

                place_trade_diff_thresholds = list(np.arange(0, 0.5, 0.05))
                place_trade_high_thresholds = list(np.arange(0.3, 0.96, 0.05))
                place_trade_low_thresholds =  list(np.arange(0.3, 0.96, 0.05))
                close_trade_take_profits =    [0.5]

                # create a list of dicts with all the combinations of hyperparameters
                hyperparameter_combinations = []
                for place_trade_diff_threshold in place_trade_diff_thresholds:
                    for place_trade_high_threshold in place_trade_high_thresholds:
                        for place_trade_low_threshold in place_trade_low_thresholds:
                            for close_trade_take_profit in close_trade_take_profits:
                                # if place_trade_high_threshold == place_trade_low_threshold:  # Temporary

                                hyperparameter_combinations.append({'place_trade_diff_threshold': place_trade_diff_threshold,
                                                                        'place_trade_high_threshold': place_trade_high_threshold,
                                                                        'place_trade_low_threshold': place_trade_low_threshold,
                                                                        'close_trade_take_profit': close_trade_take_profit})


                data_entries = []
                start_time = time.time()
                current_job_count = 1
                total_jobs_to_do = len(hyperparameter_combinations)

                for idx,hyperparameters in enumerate(hyperparameter_combinations):
                    
                    check_parameters = (model_name,
                                        specific_model_name, 
                                        range_type, 
                                        hyperparameters['place_trade_diff_threshold'],
                                        hyperparameters['place_trade_high_threshold'],
                                        hyperparameters['place_trade_low_threshold'],
                                        hyperparameters['close_trade_take_profit']
                                        )

                    if check_if_sim_was_made(check_parameters):
                        total_jobs_to_do -= 1
                        continue

                    else:
                        trade_history_df = trading_simulation(merged_df = merged_df, 
                                                        place_trade_diff_threshold = hyperparameters['place_trade_diff_threshold'], 
                                                        place_trade_high_threshold = hyperparameters['place_trade_high_threshold'], 
                                                        place_trade_low_threshold = hyperparameters['place_trade_low_threshold'], 
                                                        close_trade_take_profit = hyperparameters['close_trade_take_profit'])
                        
                    total_returns = 0
                    wins = 0
                    losses = 0
                    compounded_returns = 1
                    for row in trade_history_df.iterrows():
                        total_returns += row[1]['Returns']
                        if row[1]['Returns'] > 0:
                            wins += 1
                        elif row[1]['Returns'] < 0:
                            losses += 1
                        compounded_returns = compounded_returns*(1+(row[1]['Returns']*0.20))
                    trades_placed = len(trade_history_df)
                    if trades_placed == 0:
                        std_returns = 0
                        avg_returns = 0
                        win_rate = 0.5
                    else:
                        avg_returns = total_returns/trades_placed
                        std_returns = np.std(trade_history_df['Returns'])
                        win_rate = wins/trades_placed
                    compounded_returns -= 1

                    score = compounded_returns*avg_returns

                    data_entries.append((model_name, specific_model_name, range_type, 
                                            compounded_returns, avg_returns, std_returns, win_rate, trades_placed,
                                            hyperparameters['place_trade_diff_threshold'],
                                            hyperparameters['place_trade_high_threshold'],
                                            hyperparameters['place_trade_low_threshold'],
                                            hyperparameters['close_trade_take_profit'],
                                            score ))

                    current_job_count += 1

                    if (len(data_entries) >= 40) or (idx == len(hyperparameter_combinations)-1):
                        bulk_load_simulations(data_entries)
                        data_entries = []
                        calc_time_remaining(start_time, current_job_count, 
                                            total_jobs_to_do, 
                                            len(hyperparameter_combinations))




                
            

