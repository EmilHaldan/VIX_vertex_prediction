import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from s0_5_synthesize_VIX_option_prices import get_wednesdays_closest_to_the_20th, get_options_prices
from s1_0_Dataloader import *
from s1_1_predictionsloader import *
from s0_6_options_util import *
from s3_2_0_simulate_trades import evaluate_close_action, prepare_dataframe


def determine_action(row):

    if row['high_y_pred'] == 1:
        action = 'put'
    elif row['low_y_pred'] == 1:
        action = 'call'
    else: 
        action = 'hodl'
    return action


def baseline_trading_simulation(merged_df, close_trade_take_profit = 1):
    """
    Trading sim using futures contracts.

    """

    trade_history = []
    date_to_start_again = ""

    # Iterating through each day / candle
    for index, row in merged_df.copy().iterrows():
        if index == 0:
            prev_row = row
            continue
        
        if date_to_start_again > row['Date']:
            prev_row = row
            continue

        action = determine_action(row)

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
    
    
    model_names = ["Baseline"]
    specific_model_names = ["Baseline"]

    for i in range(1):

        model_name = model_names[i]
        specific_model_name = specific_model_names[i]

        for range_type in ["val", "test"]:

            merged_df = prepare_dataframe(model_name, specific_model_name, range_type)
            trade_history_df = baseline_trading_simulation(merged_df, close_trade_take_profit = 0.5)

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
            text_cum_returns = 1
            for idx,row in enumerate(trade_history_df.iterrows()):
                total_returns += row[1]['Returns']
                text_cum_returns = text_cum_returns*(1+(row[1]['Returns']*0.20))
            
                print(f" Trade {idx+1}: {row[1]['Trade Type']} ")
                print(f" Open date: {row[1]['Trade Open Date']}, Close date: {row[1]['Trade Close Date']} ")
                print(f" Open price: {round(row[1]['Trade Open Price'],3)}, Close price: {round(row[1]['Trade Close Price'],3)}") 
                print(f" Returns for Trade : {round(row[1]['Returns']*100,2)}%,  Avg Returns: {round(100*total_returns/(idx+1),2)}% ")
                print(f" Cumulative Returns: {round(100*(text_cum_returns-1),2)}% ")
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


            

                
            

