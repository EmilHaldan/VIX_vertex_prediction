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
from s3_2_1_simulate_options_trades import *


    #NOTE: This script was discarded in the end as it was decided to not includes options in the final model.
    #NOTE: This script was discarded in the end as it was decided to not includes options in the final model.
    #NOTE: This script was discarded in the end as it was decided to not includes options in the final model.
    #NOTE: This script was discarded in the end as it was decided to not includes options in the final model.



def bulk_load_simulations(data_entries):
    conn = sqlite3.connect('Trading_simulations/Trading_simulations.db')
    c = conn.cursor()

    # Create table if it doesn't exist
    c.execute('''CREATE TABLE IF NOT EXISTS Simulations 
                 (model_name TEXT, 
                specific_model_name TEXT, 
                range_type TEXT, 
                total_returns REAL, 
                avg_returns REAL, 
                trades_placed REAL,
                place_trade_diff_threshold REAL,
                place_trade_high_threshold REAL,
                place_trade_low_threshold REAL,
                close_trade_take_profit REAL,
                close_trade_stop_loss REAL)''')

    # Bulk insert
    c.executemany("INSERT INTO Simulations VALUES (?,?,?,?,?,?,?,?,?,?,?)", data_entries)

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
                    AND ts.close_trade_take_profit = ? 
                    AND ts.close_trade_stop_loss = ? ;
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
    

    #NOTE: This script was discarded in the end as it was decided to not includes options in the final model.
    #NOTE: This script was discarded in the end as it was decided to not includes options in the final model.
    #NOTE: This script was discarded in the end as it was decided to not includes options in the final model.
    #NOTE: This script was discarded in the end as it was decided to not includes options in the final model.


    # model_name = "LSTM_all_features_custom_loss"
    # specific_model_name = "LSTM_78"
    model_names = ["LSTM_all_features_MSE", "LSTM_all_features_custom_loss", "LSTM_OHLC_MSE", "LSTM_OHLC_custom_loss"]
    specific_model_names = ["LSTM_36" , "LSTM_78" , "LSTM_64" , "LSTM_68" ]

    for i in range(4):
        model_name = model_names[i]
        specific_model_name = specific_model_names[i]

    # for model_name in os.listdir("ML_saved_models"):
    #     for specific_model_name in os.listdir(f"ML_saved_models/{model_name}"):
        range_type = "val"

        merged_df = prepare_dataframe(model_name, specific_model_name, range_type)

        place_trade_diff_thresholds = list(np.arange(0, 0.5, 0.1))
        place_trade_high_thresholds = list(np.arange(0.6, 0.91, 0.1))
        place_trade_low_thresholds =  list(np.arange(0.6, 0.91, 0.1))

        close_trade_take_profits =    list(np.arange(0.5, 1, 0.1)) + [1.5, 2, 2.5, 3, 3.5, 4]
        close_trade_stop_losses =     list(np.arange(0.5, 1, 0.1))

        # create a list of dicts with all the combinations of hyperparameters
        hyperparameter_combinations = []
        for place_trade_diff_threshold in place_trade_diff_thresholds:
            for place_trade_high_threshold in place_trade_high_thresholds:
                for place_trade_low_threshold in place_trade_low_thresholds:
                    for close_trade_take_profit in close_trade_take_profits:
                        for close_trade_stop_loss in close_trade_stop_losses:
                            hyperparameter_combinations.append({'place_trade_diff_threshold': place_trade_diff_threshold,
                                                                'place_trade_high_threshold': place_trade_high_threshold,
                                                                'place_trade_low_threshold': place_trade_low_threshold,
                                                                'close_trade_take_profit': close_trade_take_profit,
                                                                'close_trade_stop_loss': close_trade_stop_loss})

        data_entries = []
        start_time = time.time()
        current_job_count = 1
        total_jobs_to_do = len(hyperparameter_combinations)

        for hyperparameters in hyperparameter_combinations:
            
            check_parameters = (model_name,
                                specific_model_name, 
                                range_type, 
                                hyperparameters['place_trade_diff_threshold'],
                                hyperparameters['place_trade_high_threshold'],
                                hyperparameters['place_trade_low_threshold'],
                                hyperparameters['close_trade_take_profit'],
                                hyperparameters['close_trade_stop_loss']
                                )

            if check_if_sim_was_made(check_parameters):
                total_jobs_to_do -= 1
                continue

            else:
                trade_history_df = trading_simulation(merged_df = merged_df, 
                                                place_trade_diff_threshold = hyperparameters['place_trade_diff_threshold'], 
                                                place_trade_high_threshold = hyperparameters['place_trade_high_threshold'], 
                                                place_trade_low_threshold = hyperparameters['place_trade_low_threshold'], 
                                                close_trade_take_profit = hyperparameters['close_trade_take_profit'], 
                                                close_trade_stop_loss = hyperparameters['close_trade_stop_loss'])
                
            total_returns = 0
            for row in trade_history_df.iterrows():
                total_returns += row[1]['Returns']
            trades_placed = len(trade_history_df)
            if trades_placed == 0:
                avg_returns = 0
            else:
                avg_returns = total_returns/trades_placed

            data_entries.append((model_name, specific_model_name, range_type, 
                                    total_returns, avg_returns, trades_placed,
                                    hyperparameters['place_trade_diff_threshold'],
                                    hyperparameters['place_trade_high_threshold'],
                                    hyperparameters['place_trade_low_threshold'],
                                    hyperparameters['close_trade_take_profit'],
                                    hyperparameters['close_trade_stop_loss']
                                    ))

            current_job_count += 1

            if len(data_entries) >= 10:
                bulk_load_simulations(data_entries)
                data_entries = []
                calc_time_remaining(start_time, current_job_count, 
                                    total_jobs_to_do, 
                                    len(hyperparameter_combinations))

            


            
        

