import numpy as np
import pandas as pd
import os
import sys
import random as rn
import time
from datetime import datetime
import shutil
from s1_0_Dataloader import *
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from tqdm import tqdm


def load_simulation(model_name, specific_model_name, range_type, trading_with_options):
    if trading_with_options:
        df = pd.read_csv(f"Trading_simulations/{model_name}/{specific_model_name}/{model_name}_{specific_model_name}_{range_type}_options_trading_simulation.csv")
    else:
        df = pd.read_csv(f"Trading_simulations/{model_name}/{specific_model_name}/{model_name}_{specific_model_name}_{range_type}_trading_simulation.csv")
    return df


def visualize_close_price_actions(OHLC_df, sim_df, range_type, model_name, specific_model_name):
    # Convert date strings to datetime for sim_df
    sim_df['Trade Open Date'] = pd.to_datetime(sim_df['Trade Open Date'])
    sim_df['Trade Close Date'] = pd.to_datetime(sim_df['Trade Close Date'])

    # Create a figure
    fig = go.Figure()

    

    # Initialize a variable to keep track of cumulative returns
    previous_trade_type = None

    # Iterate through the rows of the sim_df to plot arrows and text
    for i, row in sim_df.iterrows():
        trade_type = row['Trade Type']
        open_date = row['Trade Open Date']
        close_date = row['Trade Close Date']
        open_price = row['Trade Open Price']
        close_price = row['Trade Close Price']

        # Pointing up arrow at buy points (Long)
        # if trade_type == 'call':
            # fig.add_annotation(x=open_date, y=open_price, text='↑', font=dict(size=28, color='#02bd08'), showarrow=False)
            # fig.add_annotation(x=open_date, y=open_price, text='↑', font=dict(size=32, color='#9204c9'), showarrow=False)
            # fig.add_annotation(x=close_date, y=close_price, text='X', font=dict(size=22, color='#02bd08'), showarrow=False)

        # Pointing down arrow at sell points (Short)
        # elif trade_type == 'put':
            # fig.add_annotation(x=open_date, y=open_price, text='↓', font=dict(size=32, color='#9204c9'), showarrow=False)
            # fig.add_annotation(x=close_date, y=close_price, text='X', font=dict(size=22, color='red'), showarrow=False)

        if trade_type == 'call':
            color = 'green'  
        else:
            color = '#fc6f03'

        fig.add_shape(type='rect',
                      x0=open_date, y0=0, x1=close_date, y1=90,
                      line=dict(color=color),
                      fillcolor=color,
                      opacity=0.2)

        if row["Returns"] > 0:
            color = 'green'
        else: 
            color = 'red'
        fig.add_trace(go.Scatter(x=[open_date, close_date], y=[open_price, close_price],
                                 mode='lines', line=dict(color=color, dash='dot', width=3),
                                 name=f'Trade {i}'))

        # Update cumulative returns and previous Trade Type
        previous_trade_type = trade_type

    # Update layout
    title_text = f"Close Price and Trading Actions Options    {range_type}    {model_name}    {specific_model_name}"
    fig.update_layout(title=title_text, xaxis_title='Date', yaxis_title='Price', template="plotly_white", showlegend=False)

    # Add line plot for the close values from OHLC_df
    fig.add_trace(go.Scatter(x=OHLC_df['Date'], y=OHLC_df['Close'], mode='lines', name='Close Price', line=dict(color='blue', width=2)))

    # make labels of x-axis and y-axis bigger
    # make name of x-axis and y-axis bigger
    fig.update_xaxes(tickfont=dict(size=18), title_font=dict(size=18))
    fig.update_yaxes(tickfont=dict(size=18), title_font=dict(size=18))
    

    # save
    if not os.path.exists(f"Visualizations/Trading_simulations/{model_name}/{specific_model_name}/figures"):
        os.makedirs(f"Visualizations/Trading_simulations/{model_name}/{specific_model_name}/figures")
    fig.write_html(f"Visualizations/Trading_simulations/{model_name}/{specific_model_name}/figures/{model_name}_{specific_model_name}_{range_type}_close_price_and_trading_actions.html")

    # Show the figure
    fig.show()



def histogram_of_returns(sim_df, range_type, trading_with_options, model_name, specific_model_name):
    # Create subplots: 1 row, 2 columns
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Call Positions", "Put Positions"))

    # Filter data for Long and Short positions
    if trading_with_options:
        long_returns = sim_df[sim_df['Trade Type'] == 'call']['Returns']
        short_returns = sim_df[sim_df['Trade Type'] == 'put']['Returns']
    else:
        long_returns = sim_df[sim_df['Trade Type'] == 'call']['Returns']
        short_returns = sim_df[sim_df['Trade Type'] == 'put']['Returns']

    # Calculate mean and standard deviation for Long and Short positions
    mean_long, std_long = np.mean(long_returns), np.std(long_returns)
    mean_short, std_short = np.mean(short_returns), np.std(short_returns)

    # Histogram for Long Returns
    fig.add_trace(go.Histogram(x=long_returns, name='Call Returns', xbins=dict(size=0.05)), row=1, col=1)

    # Histogram for Short Returns
    fig.add_trace(go.Histogram(x=short_returns, name='Put Returns', xbins=dict(size=0.05)), row=1, col=2)

    # Annotate mean and standard deviation
    fig.add_annotation(xref='paper', yref='paper', x=0.1, y=0.9, xanchor='left', yanchor='bottom',
                       text=f'Mean: {mean_long:.2f}\nSD: {std_long:.2f}', showarrow=False, font=dict(color='black'), row=1, col=1)
    fig.add_annotation(xref='paper', yref='paper', x=0.6, y=0.9, xanchor='left', yanchor='bottom',
                       text=f'Mean: {mean_short:.2f}\nSD: {std_short:.2f}', showarrow=False, font=dict(color='black'), row=1, col=2)

    # Update layout for a consistent look and x-axis tick settings
    if trading_with_options:
        title_text = f"Histogram of Returns for Call and Put (Options) Positions    {range_type}    {model_name}    {specific_model_name}"
    else: 
        title_text = f"Histogram of Returns for Call and Put Positions    {range_type}    {model_name}    {specific_model_name}"
    fig.update_layout(template='plotly_white', title_text= title_text,
                      xaxis=dict(tickmode='linear', tick0=0, dtick=0.1),
                      xaxis2=dict(tickmode='linear', tick0=0, dtick=0.1))

    # save figure
    if not os.path.exists(f"Visualizations/Trading_simulations/{model_name}/{specific_model_name}/figures"):
        os.makedirs(f"Visualizations/Trading_simulations/{model_name}/{specific_model_name}/figures")
    fig.write_html(f"Visualizations/Trading_simulations/{model_name}/{specific_model_name}/figures/{model_name}_{specific_model_name}_{range_type}_histogram_of_returns.html")

    # Show the figure
    fig.show()


def visualize_total_wealth(OHLC_df, sim_df, range_type, trading_with_options, model_name, specific_model_name):
    # Initial wealth
    portion_to_trade_with = 0.2
    if trading_with_options:
        total_wealth = [100]
    else:
        total_wealth = [100]

    unrealised_wealth = total_wealth.copy()

    print('sim_df.tail()')
    print(sim_df.tail())
    print("")
    print('sim_df.tail()')
    print(sim_df.tail())
    print("")
    print("sim_df.iloc[0,'Trade Close Date']")
    print(sim_df.loc[0,'Trade Close Date'])

    dates_with_close_positions = [OHLC_df.loc[0,'Date']]
    dates_with_open_positions = [OHLC_df.loc[0,'Date']]
    all_dates = OHLC_df['Date'].tolist()

    for index, row in sim_df.iterrows():
        if trading_with_options:
            total_wealth.append(total_wealth[-1] + (total_wealth[-1] * portion_to_trade_with * row['Returns']))
        else:
            total_wealth.append(total_wealth[-1] + (total_wealth[-1] * portion_to_trade_with * row['Returns']))
        dates_with_close_positions.append(row["Trade Close Date"])
        dates_with_open_positions.append(row["Trade Open Date"])


    unrealised_wealth = [1]
    wealth_checkpoint_index = 0
    for idx, row in OHLC_df.iterrows():
        if idx == 0:
            prev_row = row
            continue

        for trade_index, trade_open_date in enumerate(dates_with_open_positions[1:]):
            trade_close_date = dates_with_close_positions[trade_index+1]
            if type(trade_open_date) != str:
                trade_open_date = trade_open_date.strftime("%Y-%m-%d")
            if type(trade_close_date) != str:
                trade_close_date = trade_close_date.strftime("%Y-%m-%d")

            if (trade_open_date <= row['Date']) and (row['Date'] <= trade_close_date):
                trade_type = sim_df.loc[trade_index, 'Trade Type']
                trade_open_price = sim_df.loc[trade_index, 'Trade Open Price']
                break
            trade_type = None


        if trade_type == "call":
            unrealised_wealth.append((row["Close"]-trade_open_price)/trade_open_price * portion_to_trade_with * (total_wealth[trade_index]/100) + (total_wealth[trade_index]/100))
        elif trade_type == "put":
            unrealised_wealth.append((trade_open_price-row["Close"])/trade_open_price * portion_to_trade_with * (total_wealth[trade_index]/100) + (total_wealth[trade_index]/100))
        else: 
            unrealised_wealth.append(unrealised_wealth[-1])
                

        prev_row = row
        
    unrealised_wealth = [(x * 100 - 100) for x in unrealised_wealth]

    sim_df['Cum Returns'] = total_wealth[1:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=all_dates, y=unrealised_wealth, mode='lines', name='Total Wealth', line=dict(color='blue', width=1)))

    y_green = []
    y_red = []
    tmp_sim_df = sim_df.copy()
    print("tmp_sim_df.head()")
    print(tmp_sim_df.head())

    for idx,wealth in enumerate(unrealised_wealth):
        if wealth < 0:
            y_green.append(wealth)  
            y_red.append(0)      
            
        else:
            y_green.append(0)     
            y_red.append(None)   

    # Adding green area (above 100)
    fig.add_trace(go.Scatter(x=all_dates, y=y_green, 
                             fill='tonexty',
                             mode='none', 
                             fillcolor='rgba(0, 255, 0, 0.3)', # Semi-transparent green
                             name='Above 0'))

    # Adding red area (below 100)
    fig.add_trace(go.Scatter(x=all_dates, y=y_red, 
                             fill='tonexty',
                             mode='none', 
                             fillcolor='rgba(255, 0, 0, 0.3)', # Semi-transparent red
                             name='Below 0'))


    # Adding dots for each trade ( was removed because it was too messy )
    for index, row in sim_df.iterrows():
        trade_date = dates_with_close_positions[index+1]
        trade_wealth = total_wealth[index+1]

        # Constructing hover text with trade information
        # if trading_with_options:
        #     hover_text = (f"Trade {index}<br>"
        #                 f"Open Date: {row['Trade Open Date']}<br>"
        #                 f"Close Date: {row['Trade Close Date']}<br>"
        #                 f"Contract Bought Price: {row['Contract Bought Price']}<br>"
        #                 f"Contract Sold Price: {row['Contract Sold Price']}<br>"
        #                 f"Returns: {row['Returns']}<br><br>"
        #                 f"Trade Type: {row['Trade Type']}<br>"
        #                 f"Strike price: {row['Strike price']}<br>"
        #                 f"Trade Open Price: {row['Trade Open Price']}<br>"
        #                 f"Trade Close Price: {row['Trade Close Price']}<br>")
        # else:
        #     hover_text = (f"Trade {index}<br>"
        #                 f"Open Date: {row['Trade Open Date']}<br>"
        #                 f"Close Date: {row['Trade Close Date']}<br>"
        #                 f"Returns: {row['Returns']}<br><br>"
        #                 f"Compounded Returns: {row['Cum Returns']}<br>"
        #                 f"Trade Type: {row['Trade Type']}<br>"
        #                 f"Trade Open Price: {row['Trade Open Price']}<br>"
        #                 f"Trade Close Price: {row['Trade Close Price']}<br>")

        # fig.add_trace(go.Scatter(x=[trade_date], y=[trade_wealth], mode='markers', name=f'Trade {index}',
        #                  text=[hover_text], hoverinfo='text', 
        #                  marker=dict(size=10, color='black')))  # Set marker color to black

    if trading_with_options:
        title_text = f"Total Wealth Over Time    {range_type}    {model_name}    {specific_model_name}"
    else: 
        title_text = f"Total Wealth Over Time    {range_type}    {model_name}    {specific_model_name}"

    fig.update_layout(title=title_text, xaxis_title='Date', yaxis_title='Cumulative  Returns (%)', template='plotly_white', showlegend=False)

    fig.update_xaxes(tickfont=dict(size=18))
    fig.update_yaxes(tickfont=dict(size=18))
    fig.update_xaxes(tickfont=dict(size=18), title_font=dict(size=18))
    fig.update_yaxes(tickfont=dict(size=18), title_font=dict(size=18))

    if not os.path.exists(f"Visualizations/Trading_simulations/{model_name}/{specific_model_name}/figures"):
        os.makedirs(f"Visualizations/Trading_simulations/{model_name}/{specific_model_name}/figures")
    fig.write_html(f"Visualizations/Trading_simulations/{model_name}/{specific_model_name}/figures/{model_name}_{specific_model_name}_{range_type}_total_wealth.html")

    fig.show()



if __name__ == "__main__":



    model_names = ["LSTM_OHLC_MSE","LSTM_all_features_MSE", "LSTM_OHLC_custom_loss","LSTM_all_features_custom_loss"]
    specific_model_names = ["LSTM_64", "LSTM_36", "LSTM_68", "LSTM_78"]

    for i in range(4):
        for range_type in ["val", "test"]:

            model_name = model_names[i]
            specific_model_name = specific_model_names[i]
            trading_with_options = False
            if "LSTM" not in model_name:
                continue

            try:
                sim_df = load_simulation(model_name, specific_model_name, range_type, trading_with_options)
            except FileNotFoundError:
                continue

            OHLC_df = load_specific_data(range_type = range_type, version = 3)["all"][["Date", "Close"]]

            # histogram_of_returns(sim_df, range_type, trading_with_options, model_name, specific_model_name)

            visualize_close_price_actions(OHLC_df, sim_df, range_type, model_name, specific_model_name)

            visualize_total_wealth(OHLC_df,sim_df, range_type, trading_with_options, model_name, specific_model_name)

