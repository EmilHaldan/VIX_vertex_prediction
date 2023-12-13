import pandas as pd 
from datetime import datetime, timedelta
from tqdm import tqdm
import numpy as np
from scipy.interpolate import griddata
from scipy.stats import norm
import sqlite3

from s1_0_Dataloader import load_specific_data

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


#NOTE: This script was discarded in the end as it was decided to not includes options in the final model.

def calc_time_to_expiration(current_date, expiration_date):
    current_date = datetime.strptime(current_date, '%Y-%m-%d')
    time_to_expiration = (expiration_date - current_date).days

    return time_to_expiration

#NOTE: This script was discarded in the end as it was decided to not includes options in the final model.

def black_scholes(current_price, strike_price, current_date, expiry_date, risk_free_rate, volatility, option_type='call', dividend_yield=0):
    """
    Calculate the Black-Scholes option price with more casually named variables.

    Parameters:
    current_price (float): Current price of the underlying asset
    strike_price (float): Strike price of the option
    days_to_expiration (float): Time to expiration in days
    risk_free_rate (float): Annual risk-free interest rate
    volatility (float): Annual volatility of the underlying asset
    option_type (str): Type of the option - 'call' or 'put'
    dividend_yield (float): Annual dividend yield of the underlying asset

    Returns:
    float: Price of the option
    """

    days_to_expiration = calc_time_to_expiration(current_date, expiry_date)
    time_to_expiration = days_to_expiration / 365

    d1 = (np.log(current_price / strike_price) + (risk_free_rate - dividend_yield + 0.5 * volatility ** 2) * time_to_expiration) / (volatility * np.sqrt(time_to_expiration))
    d2 = d1 - volatility * np.sqrt(time_to_expiration)

    if option_type == 'call':
        option_price = (current_price * np.exp(-dividend_yield * time_to_expiration) * norm.cdf(d1)) - (strike_price * np.exp(-risk_free_rate * time_to_expiration) * norm.cdf(d2))
    elif option_type == 'put':
        option_price = (strike_price * np.exp(-risk_free_rate * time_to_expiration) * norm.cdf(-d2)) - (current_price * np.exp(-dividend_yield * time_to_expiration) * norm.cdf(-d1))
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    return option_price

#NOTE: This script was discarded in the end as it was decided to not includes options in the final model.

def calculate_annualized_volatility(merged_df):
    """
    Calculates the 252-day rolling volatility up to a given date.

    Parameters:
    merged_df (DataFrame): A DataFrame with columns 'Date' and 'Log_Return'.
    input_date (str): The date up to which volatility is calculated (format: YYYY-MM-DD).

    Returns:
    float: 252-day rolling volatility up to the input date.
    This is a simplification of the actual calculation of volatility for options pricing. 
    As the VIX is calculated based on options prices of the S&P 500, the volatility of the VIX is used as a proxy for the volatility of the S&P 500 options prices.
    This by no means is a perfect solution, but it is a good approximation for the purpose of this project.
    """

    merged_df['Log_Return'] = np.log(merged_df['Close'] / merged_df['Close'].shift(1))
    merged_df['Annualized_Volatility'] = merged_df['Log_Return'].rolling(window=252, min_periods=1).std() * np.sqrt(252)

    first_valid_index = merged_df['Annualized_Volatility'].first_valid_index()
    first_valid_value = merged_df.loc[first_valid_index, 'Annualized_Volatility']
    merged_df['Annualized_Volatility'].fillna(first_valid_value, inplace=True)

    return merged_df

#NOTE: This script was discarded in the end as it was decided to not includes options in the final model.

def closest_wednesday(year, month):
    date_20th = datetime(year, month, 20)
    weekday = date_20th.weekday()

    # If 20th is Wednesday, return it, else find the last Wednesday before the 20th, because european options expire on the third Wednesday of the month
    if weekday == 2:
        return date_20th
    else:
        # Subtract the difference in days to the last Wednesday
        closest_wed = date_20th - timedelta(days=(weekday - 2 if weekday > 2 else weekday + 5))
        return closest_wed

#NOTE: This script was discarded in the end as it was decided to not includes options in the final model.

def get_wednesdays_closest_to_the_20th(cur_date):

    """
    Returns the Wednesdays closest to the 20th of the month for the following 3 months, starting at the current month.

    Parameters:
    cur_date_str (str): The date for which the Wednesdays are calculated, in the format 'YYYY-MM-DD'.

    Returns:
    list[datetime]: a list of 3 wednesday dates
    """
    wednesdays = []
    try:
        cur_date = datetime.strptime(cur_date, '%Y-%m-%d')
    except TypeError:
        pass # already datetime object

    for _ in range(5):
        year, month = cur_date.year, cur_date.month
        cur_closest_wednesday = closest_wednesday(year, month)
        if cur_closest_wednesday >= cur_date:
            wednesdays.append(cur_closest_wednesday)

        if len(wednesdays) == 3:
            break

        if month == 12:
            cur_date = datetime(year + 1, 1, 1)
        else:
            cur_date = datetime(year, month + 1, 1)

    return wednesdays

#NOTE: This script was discarded in the end as it was decided to not includes options in the final model.

def load_data(current_date, expiry_date, strike_price, cp_flag, options_price):
    conn = sqlite3.connect('Data/Options_price_data/options_prices.db')
    c = conn.cursor()

    # Create table if it doesn't exist
    c.execute('''CREATE TABLE IF NOT EXISTS options_contracts 
                 (current_date TEXT, expiry_date TEXT, strike_price REAL, cp_flag TEXT, options_price REAL)''')

    # Inserting a new record
    c.execute("INSERT INTO options_contracts VALUES (?,?,?,?,?)", 
              (current_date, expiry_date, strike_price, cp_flag, options_price))

    conn.commit()
    conn.close()

#NOTE: This script was discarded in the end as it was decided to not includes options in the final model.

def bulk_load_data(data_entries):
    conn = sqlite3.connect('Data/Options_price_data/options_prices.db')
    c = conn.cursor()

    # Create table if it doesn't exist
    c.execute('''CREATE TABLE IF NOT EXISTS options_contracts 
                 (current_date TEXT, expiry_date TEXT, strike_price REAL, cp_flag TEXT, options_price REAL)''')

    # Bulk insert
    c.executemany("INSERT INTO options_contracts VALUES (?,?,?,?,?)", data_entries)

    conn.commit()
    conn.close()

#NOTE: This script was discarded in the end as it was decided to not includes options in the final model.

def get_options_prices(start_date, expiry_date, strike_price, cp_flag):
    conn = sqlite3.connect('Data/Options_price_data/options_prices.db')
    c = conn.cursor()

    try:
        start_date = start_date.strftime("%Y-%m-%d")
    except AttributeError:
        pass
    expiry_date_string = expiry_date.strftime("%Y-%m-%d")
    c.execute("SELECT oc.current_date, oc.options_price FROM options_contracts AS oc WHERE oc.current_date>=? AND oc.current_date<=? AND oc.expiry_date=? AND oc.strike_price=? AND oc.cp_flag=?", 
                (start_date, expiry_date_string, expiry_date_string, strike_price, cp_flag))

    result = c.fetchall()
    conn.close()
    return result

#NOTE: This script was discarded in the end as it was decided to not includes options in the final model.

if __name__ == "__main__":

    range_type = "all"
    OHLC_df = load_specific_data(range_type = range_type, version = 3)
    OHLC_df = OHLC_df["all"][["Date", "Open", "High", "Low", "Close"]]
    OHLC_df = OHLC_df[OHLC_df["Date"] >= "2016-12-31"]
    OHLC_df.reset_index(drop=True, inplace=True)
    OHLC_df = calculate_annualized_volatility(OHLC_df)

    all_dates = OHLC_df["Date"].values
    all_dates = [date for date in all_dates]
    
    min_price = OHLC_df["Close"].min()*1000

    for cur_date in tqdm(all_dates):
        data_entries = []
        cur_VIX_price = OHLC_df[OHLC_df["Date"] == cur_date]["Close"].values[0]
        rounded_cur_VIX_price = round(cur_VIX_price)*1000
        volatility = OHLC_df[OHLC_df["Date"] == cur_date]['Annualized_Volatility'].values[0]
        future_wednesdays = get_wednesdays_closest_to_the_20th(cur_date)
        for expiry_date in future_wednesdays:
            for cp_flag in ["call", "put"]:
                strike_prices = [x/1000 for x in range(7000, rounded_cur_VIX_price+20000, 500) if x >= min_price]
                for strike_price in strike_prices:
                    option_price = black_scholes(current_price = cur_VIX_price, 
                                                strike_price = strike_price, 
                                                current_date = cur_date, 
                                                expiry_date = expiry_date, 
                                                risk_free_rate = 0.01, 
                                                volatility = volatility, 
                                                option_type=cp_flag, 
                                                dividend_yield=0)
                    data_entries.append((cur_date, expiry_date.strftime('%Y-%m-%d'), strike_price, cp_flag, option_price))
                    print("Option price: ", round(option_price,2), " for strike price: ", strike_price, " and expiry date: ", expiry_date.date(), " and cp_flag: ", cp_flag)
        
        bulk_load_data(data_entries)

#NOTE: This script was discarded in the end as it was decided to not includes options in the final model.