import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from s0_5_synthesize_VIX_option_prices import get_wednesdays_closest_to_the_20th, get_options_prices
from s1_0_Dataloader import *
from s1_1_predictionsloader import *
from s0_6_options_util import *

import pandas as pd

#NOTE: This script was discarded in the end as it was decided to not includes options in the final model.
#NOTE: This script was discarded in the end as it was decided to not includes options in the final model.
#NOTE: This script was discarded in the end as it was decided to not includes options in the final model.
#NOTE: This script was discarded in the end as it was decided to not includes options in the final model.

def determine_action(row, prev_row, diff_threshold = 0.25, high_threshold = 0.8, low_threshold = 0.8):

    pred_diff = row['high_y_pred'] - row['low_y_pred']

    if (pred_diff > diff_threshold) and (row['high_y_pred'] < prev_row['high_y_pred']) and (row['high_y_pred'] > high_threshold):
        action = 'sell'
    elif pred_diff < -diff_threshold and (row['low_y_pred'] < prev_row['low_y_pred']) and (row['low_y_pred'] > low_threshold):
        action = 'buy'
    else: 
        action = 'hodl'

    return action

#NOTE: This script was discarded in the end as it was decided to not includes options in the final model.
#NOTE: This script was discarded in the end as it was decided to not includes options in the final model.
#NOTE: This script was discarded in the end as it was decided to not includes options in the final model.
#NOTE: This script was discarded in the end as it was decided to not includes options in the final model.

def evaluate_close_action(trade, merged_df, take_profit = 0.25, stop_loss = 0.25):

    expiry_dates = get_wednesdays_closest_to_the_20th(trade["Trade Open Date"])
    if expiry_dates[0] > datetime.strptime(trade["Trade Open Date"], '%Y-%m-%d') + timedelta(days=-30): # The expiry date should be between 30 and 60 days.
        expiry_date = expiry_dates[1]
    else:
        expiry_date = expiry_dates[0]

    results = get_options_prices(start_date = trade["Trade Open Date"], 
                        expiry_date = expiry_date, 
                        strike_price = trade["Strike price"], 
                        cp_flag = trade["Trade type"].lower())

    trade["Expiry date"] = expiry_date.strftime("%Y-%m-%d")
    trade["Contract Bought Price"] = results[0][1]
    last_date_in_simulation = merged_df.iloc[-1]["Date"]
    for date, contract_price in results:
        if (contract_price > (trade["Contract Bought Price"] + trade["Contract Bought Price"]*take_profit)) or (date == last_date_in_simulation):
            trade["Trade Close Date"] = date
            trade["Contract Sold Price"] = contract_price 
            break
        elif contract_price < trade["Contract Bought Price"]*stop_loss:
            trade["Trade Close Date"] = date
            trade["Contract Sold Price"] = contract_price 
            break
    
    if "Contract Sold Price" not in trade.keys():
        trade["Trade Close Date"] = trade["Expiry date"]
        trade["Contract Sold Price"] = contract_price

    trade["Returns"] = (trade["Contract Sold Price"] - trade["Contract Bought Price"])/trade["Contract Bought Price"]
    trade["Trade Close Price"] = merged_df[merged_df["Date"] == trade["Trade Close Date"]]["Close"].values[0]
    trade["Trade state"] = "Closed"

    return trade


#NOTE: This script was discarded in the end as it was decided to not includes options in the final model.
#NOTE: This script was discarded in the end as it was decided to not includes options in the final model.
#NOTE: This script was discarded in the end as it was decided to not includes options in the final model.
#NOTE: This script was discarded in the end as it was decided to not includes options in the final model.

def prepare_dataframe(model_name, specific_model_name, range_type):
    test_pred_df = load_target_and_pred(model_name, specific_model_name, range_type = range_type)
    OHLC_df = load_specific_data(range_type = range_type, version = 3)
    OHLC_df = OHLC_df["all"][["Date", "Open", "High", "Low", "Close"]]

    test_pred_df.rename(columns={'date': 'Date'}, inplace=True)
    merged_df = pd.merge(test_pred_df, OHLC_df, how='left', left_on='Date', right_on='Date')
    merged_df.dropna(inplace=True)
    merged_df.reset_index(drop=True, inplace=True)

    return merged_df

#NOTE: This script was discarded in the end as it was decided to not includes options in the final model.
#NOTE: This script was discarded in the end as it was decided to not includes options in the final model.
#NOTE: This script was discarded in the end as it was decided to not includes options in the final model.
#NOTE: This script was discarded in the end as it was decided to not includes options in the final model.

def trading_simulation(merged_df, place_trade_diff_threshold = 0.25,  # some arbitrary threshold values
                                  place_trade_high_threshold = 0.8, 
                                  place_trade_low_threshold = 0.8, 
                                  close_trade_take_profit = 0.25, 
                                  close_trade_stop_loss = 0.25):
    """
    Trading sim using futures contracts.

    """
    global fixed_futures_time
    global break_even_diff

    trade_history = []
    date_to_start_again = ""

    for index, row in merged_df.copy().iterrows():
        if index == 0:
            prev_row = row
            continue
        
        if date_to_start_again > row['Date']:
            prev_row = row
            continue

        action = determine_action(row = row, prev_row= prev_row, diff_threshold = place_trade_diff_threshold, 
                                high_threshold = place_trade_high_threshold, low_threshold = place_trade_low_threshold) 

        # Place a new trade if there are no open trades
        if action != 'hodl':
            wednesdays = get_wednesdays_closest_to_the_20th(row['Date'])
            expiry_date = wednesdays[1] # Second wednesday of the month resulting in a 30-60 day expiry

            if action == 'buy':
                trade = {
                    "Trade state": "Open",
                    "Trade type": "call",
                    "Trade Open Date": row['Date'],
                    "Trade Open Price": row['Close'],
                    "Contract Bought Price": "tbd", # Gets defined in evaluate_close_action
                    "Strike price": round(row['Close']+2)
                }
                tmp_merged_df = merged_df.iloc[index:index+90].reset_index(drop=True)
                trade = evaluate_close_action(trade = trade, 
                                              merged_df = tmp_merged_df,
                                              take_profit = close_trade_take_profit,
                                              stop_loss   = close_trade_stop_loss)

            elif action == 'sell':
                trade = {
                    "Trade state": "Open",
                    "Trade type": "put",
                    "Trade Open Date": row['Date'],
                    "Trade Open Price": row['Close'],
                    "Contract Bought Price" : "tbd" , # Gets defined in evaluate_close_action
                    "Strike price": round(row['Close']-2)
                }
                tmp_merged_df = merged_df.iloc[index:index+90].reset_index(drop=True)
                trade = evaluate_close_action(trade = trade, 
                                              merged_df = tmp_merged_df,
                                              take_profit = close_trade_take_profit,
                                              stop_loss   = close_trade_stop_loss)

            date_to_start_again = trade["Trade Close Date"]

            trade_history.append(trade)

        prev_row = row

    # Close any remaining open trades at the end of the simulation
    close_price = merged_df.iloc[-1]['Close']
    for trade in trade_history:
        if trade["Trade state"] == "Open":
            close_trade(trade, close_row = merged_df.iloc[-1])

    # Convert trade history to DataFrame for analysis
    trade_history_df = pd.DataFrame(trade_history)

    return trade_history_df




if __name__ == "__main__":
    


    #NOTE: This script was discarded in the end as it was decided to not includes options in the final model.
    #NOTE: This script was discarded in the end as it was decided to not includes options in the final model.
    #NOTE: This script was discarded in the end as it was decided to not includes options in the final model.
    #NOTE: This script was discarded in the end as it was decided to not includes options in the final model.

    model_name = "LSTM_all_features_custom_loss"
    specific_model_name = "LSTM_78"

    for range_type in ["val", "test"]:
    # for range_type in ["test"]:

        merged_df = prepare_dataframe(model_name, specific_model_name, range_type)
        trade_history_df = trading_simulation(merged_df, place_trade_diff_threshold = 0.2,  # some arbitrary threshold values
                                  place_trade_high_threshold = 0.6, 
                                  place_trade_low_threshold = 0.6, 
                                  close_trade_take_profit = 2.5, 
                                  close_trade_stop_loss = 0.5)

        print("trade_history_df: \n")
        sum_of_returns = 1

        sum_of_returns = 0
        for row in trade_history_df.iterrows():
            sum_of_returns += row[1]['Returns']
            trades_placed = len(trade_history_df)
            if trades_placed == 0:
                avg_returns = 0
            else:
                avg_returns = sum_of_returns/trades_placed
        
            # sum_of_returns = sum_of_returns*(1+row[1]['Returns'])
            print(f" Open date: {row[1]['Trade Open Date']}, Close date: {row[1]['Trade Close Date']} ")
            print(f" Open price: {round(row[1]['Trade Open Price'],3)}, Close price: {round(row[1]['Trade Close Price'],3)}") 
            print(f" Strike price: {round(row[1]['Strike price'],3)}  Expiry date: {row[1]['Expiry date']}")
            print(f" Contract Bought Price: {round(row[1]['Contract Bought Price'],4)}, Contract Sold Price: {round(row[1]['Contract Sold Price'],4)}")
            print(f" Returns: {round(row[1]['Returns'],3)} ")
            print("")

        print("\n"*2)
        print("Total Trades: ", len(trade_history_df))
        print("Sum of Returns: ", sum_of_returns)
        print("Average Returns pr. trade: ", sum_of_returns/len(trade_history_df))
        print("")
        print("Start Date: ", merged_df.iloc[0]['Date'])
        print("End Date  : ", merged_df.iloc[-1]['Date'])

        if not os.path.exists(f"Trading_simulations/{model_name}/{specific_model_name}"):
            os.makedirs(f"Trading_simulations/{model_name}/{specific_model_name}")

        trade_history_df.to_csv(f"Trading_simulations/{model_name}/{specific_model_name}/{model_name}_{specific_model_name}_{range_type}_options_trading_simulation.csv", index=False)


        

            
        

