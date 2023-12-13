import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from s0_5_synthesize_VIX_option_prices import get_wednesdays_closest_to_the_20th, get_options_prices
from s1_0_Dataloader import *
from s1_1_predictionsloader import *
from s0_6_options_util import *



def determine_action(row, prev_row, diff_threshold = 0.25, high_threshold = 0.8, low_threshold = 0.8):

    pred_diff = row['high_y_pred'] - row['low_y_pred']
    if (pred_diff > diff_threshold) and (row['high_y_pred'] < prev_row['high_y_pred']) and (row['high_y_pred'] > high_threshold):
        action = 'put'
    elif (pred_diff < -diff_threshold) and (row['low_y_pred'] < prev_row['low_y_pred']) and (row['low_y_pred'] > low_threshold):
        action = 'call'
    else: 
        action = 'hodl'
    return action


def evaluate_close_action(trade, merged_df, take_profit = 0.5):

    last_date_in_simulation = merged_df.iloc[-1]["Date"]
    for idx, row in merged_df.iterrows():
        date = row["Date"]
        if trade["Trade Type"] == "call":
            take_profit_price = trade["Trade Open Price"]*(1+take_profit)
            if row["Close"] > take_profit_price:
                trade["Trade Close Date"] = date
                trade["Trade Close Price"] = take_profit_price
                trade["Trade State"] = "Closed"
                break

        elif trade["Trade Type"] == "put":
            take_profit_price = trade["Trade Open Price"]*((1-take_profit)) 
            if row["Close"] < take_profit_price:
                trade["Trade Close Date"] = date
                trade["Trade Close Price"] = take_profit_price
                trade["Trade State"] = "Closed"
                break

        if (date == last_date_in_simulation):
            trade["Trade Close Date"] = date
            trade["Trade Close Price"] = row["Close"]
            trade["Trade State"] = "Closed"
            break

    if trade["Trade Type"] == "call":
        trade["Returns"] = (trade["Trade Close Price"] - trade["Trade Open Price"])/trade["Trade Open Price"]
    elif trade["Trade Type"] == "put":
        trade["Returns"] = (trade["Trade Open Price"] - trade["Trade Close Price"])/trade["Trade Open Price"]
    else: 
        raise ValueError("Trade type must be either 'call' or 'put'")

    return trade


def prepare_dataframe(model_name, specific_model_name, range_type):
    test_pred_df = load_target_and_pred(model_name, specific_model_name, range_type = range_type)
    OHLC_df = load_specific_data(range_type = range_type, version = 3)
    OHLC_df = OHLC_df["all"][["Date", "Open", "High", "Low", "Close"]]

    test_pred_df.rename(columns={'date': 'Date'}, inplace=True)
    merged_df = pd.merge(test_pred_df, OHLC_df, how='left', left_on='Date', right_on='Date')
    merged_df.dropna(inplace=True)
    merged_df.reset_index(drop=True, inplace=True)

    return merged_df


def trading_simulation(merged_df, place_trade_diff_threshold = 0.25,  # some arbitrary threshold values
                                  place_trade_high_threshold = 0.8, 
                                  place_trade_low_threshold = 0.8, 
                                  close_trade_take_profit = 0.5):
    """
    Trading sim using futures contracts.

    """

    trade_history = []
    date_to_start_again = ""

    # Iterating through each day
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
            if action == 'call':
                trade = {
                    "Trade State": "Open",
                    "Trade Type": "call",
                    "Trade Open Date": row['Date'],
                    "Trade Open Price": row['Close'],
                }
                tmp_merged_df = merged_df.iloc[index:index+50].reset_index(drop=True)
                trade = evaluate_close_action(trade = trade, 
                                              merged_df = tmp_merged_df,
                                              take_profit = close_trade_take_profit)

            elif action == 'put':
                trade = {
                    "Trade State": "Open",
                    "Trade Type": "put",
                    "Trade Open Date": row['Date'],
                    "Trade Open Price": row['Close'],
                }
                tmp_merged_df = merged_df.iloc[index:index+50].reset_index(drop=True)
                trade = evaluate_close_action(trade = trade, 
                                              merged_df = tmp_merged_df,
                                              take_profit = close_trade_take_profit)

            date_to_start_again = trade["Trade Close Date"]

            trade_history.append(trade)

        prev_row = row

    trade_history_df = pd.DataFrame(trade_history)

    return trade_history_df



if __name__ == "__main__":
    
    model_names = ["LSTM_OHLC_MSE","LSTM_all_features_MSE", "LSTM_OHLC_custom_loss","LSTM_all_features_custom_loss"]
    specific_model_names = ["LSTM_64", "LSTM_36", "LSTM_68", "LSTM_78"]
    place_trade_high_thresholds = [0.75,  0.55, 0.8,  0.6]
    place_trade_low_thresholds =  [0.8,   0.85, 0.7,  0.9]
    place_trade_diff_thresholds = [0,     0,    0.1,  0.25]


    for i in range(4):

        model_name = model_names[i]
        specific_model_name = specific_model_names[i]
        place_trade_diff_threshold = place_trade_diff_thresholds[i]
        place_trade_high_threshold = place_trade_high_thresholds[i]
        place_trade_low_threshold = place_trade_low_thresholds[i]
        for range_type in ["val", "test"]:

            merged_df = prepare_dataframe(model_name, specific_model_name, range_type)
            trade_history_df = trading_simulation(merged_df, place_trade_diff_threshold = place_trade_diff_threshold, 
                                    place_trade_high_threshold = place_trade_high_threshold, 
                                    place_trade_low_threshold  = place_trade_low_threshold, 
                                    close_trade_take_profit = 0.5)

            

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

            print("trade_history_df: \n")

            total_returns = 0
            for idx,row in enumerate(trade_history_df.iterrows()):
                total_returns += row[1]['Returns']
            
                print(f" Trade {idx+1}: {row[1]['Trade Type']} ")
                print(f" Open date: {row[1]['Trade Open Date']}, Close date: {row[1]['Trade Close Date']} ")
                print(f" Open price: {round(row[1]['Trade Open Price'],3)}, Close price: {round(row[1]['Trade Close Price'],3)}") 
                print(f" Returns for Trade : {round(row[1]['Returns']*100,2)}%,  Avg Returns: {round(100*total_returns/(idx+1),2)}% ")
                print("")

            if trades_placed == 0:
                avg_returns = 0
            else:
                avg_returns = total_returns/trades_placed

            compounded_returns = round(100*(compounded_returns),2)
            total_returns = round(100*(total_returns),2)
            avg_returns = round(100*(avg_returns),2)

            print("\n"*2)
            print("Total Trades                                             : ", len(trade_history_df))
            print("Cumulative Returns (Investing with 20% pr. Trade)        : ", compounded_returns,"%")
            print("Average Returns pr. trade                                : ", avg_returns, "%")
            print("")
            print("Start Date                                               : ", merged_df.iloc[0]['Date'])
            print("End Date                                                 : ", merged_df.iloc[-1]['Date'])

            if not os.path.exists(f"Trading_simulations/{model_name}/{specific_model_name}"):
                os.makedirs(f"Trading_simulations/{model_name}/{specific_model_name}")

            trade_history_df.to_csv(f"Trading_simulations/{model_name}/{specific_model_name}/{model_name}_{specific_model_name}_{range_type}_trading_simulation.csv", index=False)


            

                
            

