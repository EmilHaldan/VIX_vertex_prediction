import pandas as pd 
from datetime import datetime
from tqdm import tqdm
import numpy as np
from scipy.interpolate import griddata

from s1_0_Dataloader import load_specific_data



#NOTE: This script was discarded in the end as the interpolation was not working as expected.
#NOTE: the gaps of missing contracts were too significant to be interpolated, and to interpolate them, 
# the implied volatility would have to be entirely synthesized.
# instead of working with 80% synthetic data, and 20% real data, the entire dataset will now be synthesized.

def black_scholes(current_price, strike_price, current_date, expiry_date, expiration_date, risk_free_rate, volatility, option_type='call', dividend_yield=0):
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

    time_to_expiration = calc_time_to_expiration(current_date, expiration_date)
    time_to_expiration = days_to_expiration / 365

    d1 = (np.log(current_price / strike_price) + (risk_free_rate - dividend_yield + 0.5 * volatility ** 2) * time_to_expiration) / (volatility * np.sqrt(time_to_expiration))
    d2 = d1 - volatility * np.sqrt(time_to_expiration)

    # Calculate option price
    if option_type == 'call':
        option_price = (current_price * np.exp(-dividend_yield * time_to_expiration) * norm.cdf(d1)) - (strike_price * np.exp(-risk_free_rate * time_to_expiration) * norm.cdf(d2))
    elif option_type == 'put':
        option_price = (strike_price * np.exp(-risk_free_rate * time_to_expiration) * norm.cdf(-d2)) - (current_price * np.exp(-dividend_yield * time_to_expiration) * norm.cdf(-d1))
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    return option_price


#NOTE: This script was discarded in the end as the interpolation was not working as expected.
#NOTE: the gaps of missing contracts were too significant to be interpolated, and to interpolate them, 
# the implied volatility would have to be entirely synthesized.
# instead of working with 80% synthetic data, and 20% real data, the entire dataset will now be synthesized.

def calculate_30day_volatility(merged_df):
    """
    Calculates the 30-day rolling volatility up to a given date.

    Parameters:
    merged_df (DataFrame): A DataFrame with columns 'Date' and 'Log_Return'.
    input_date (str): The date up to which volatility is calculated (format: YYYY-MM-DD).

    Returns:
    float: 30-day rolling volatility up to the input date.
    This is a simplification of the actual calculation of volatility for options pricing. 
    As the VIX is calculated based on options prices of the S&P 500, the volatility of the VIX is used as a proxy for the volatility of the S&P 500 options prices.
    This by no means is a perfect solution, but it is a good approximation for the purpose of this project.
    """

    # Calculating the log returns used for the volatility calculation
    merged_df['Log_Return'] = np.log(merged_df['Close'] / merged_df['Close'].shift(1))

    # Calculate the rolling standard deviation of log returns for a 30-day window
    # And then multiply by sqrt(252 / 30) to annualize the 30-day volatility
    merged_df['30Day_Volatility'] = merged_df['Log_Return'].rolling(window=30, min_periods=1).std() * np.sqrt(252 / 30)

    # Fill initial NaN values with the first valid volatility value (Not ideal, but better than NaN)
    first_valid_index = merged_df['30Day_Volatility'].first_valid_index()
    first_valid_value = merged_df.loc[first_valid_index, '30Day_Volatility']
    merged_df['30Day_Volatility'].fillna(first_valid_value, inplace=True)

    return merged_df


#NOTE: This script was discarded in the end as the interpolation was not working as expected.
#NOTE: the gaps of missing contracts were too significant to be interpolated, and to interpolate them, 
# the implied volatility would have to be entirely synthesized.
# instead of working with 80% synthetic data, and 20% real data, the entire dataset will now be synthesized.

def basic_filtering_VIX_options_listings():
    df = pd.read_csv("Data/Options_price_data/VIX_historical_options_listings.csv")
    df = df[["date", "exdate", "last_date", "cp_flag", "strike_price", "best_bid", "best_offer", "impl_volatility"]]

    df["avg_price"] = (df["best_bid"] + df["best_offer"])/2
    df["strike_price"] = df["strike_price"]/1000
    # make dates to stings
    df["date"] = df["date"].astype(str)
    df["exdate"] = df["exdate"].astype(str)
    df["last_date"] = df["last_date"].astype(str)

    # Change the format for cp_type to call and put
    df["cp_flag"] = df["cp_flag"].replace("C", "call")
    df["cp_flag"] = df["cp_flag"].replace("P", "put")

    return df


#NOTE: This script was discarded in the end as the interpolation was not working as expected.
#NOTE: the gaps of missing contracts were too significant to be interpolated, and to interpolate them, 
# the implied volatility would have to be entirely synthesized.
# instead of working with 80% synthetic data, and 20% real data, the entire dataset will now be synthesized.

def child_interpolate_vol_and_price(daily_df, vix_price, current_date, cp_flag):

    if daily_df['impl_volatility'].isnull().sum() <= len(daily_df)/2:
        interp_func = interp1d(daily_df['strike_price'], daily_df['impl_volatility'], kind='cubic', fill_value='exterpolate')
        interpolated_vols = interp_func(strike_range)

        interpolated_df = pd.DataFrame({
            'strike_price': strike_range,
            'impl_volatility': interpolated_vols
        })

        interpolated_df = pd.merge(interpolated_df, daily_df[['strike_price', 'avg_price']], on='strike_price', how='left')

        interpolated_df['avg_price'] = interpolated_df.apply(
            lambda row: row['avg_price'] if not pd.isna(row['avg_price']) else black_scholes(
            current_price=vix_price,  # assuming current_price is the VIX price
            strike_price=row['strike_price'], 
            current_date=current_date,  # 
            expiry_date=daily_df['exdate'].iloc[0],  # all expiry date is constant for each child call
            risk_free_rate= 0.01,  # 1% risk-free rate of return (standard finance practice as a replacement for the actual libor rate)
            volatility=row['impl_volatility'],
            option_type=daily_df['cp_flag'].iloc[0],  # all option types are the same for each child call
            dividend_yield=0  # because VIX
        ), axis=1)

    else: 
        # linear interpolation for the avg_price
        interp_func = interp1d(daily_df['strike_price'], daily_df['avg_price'], kind='slinear', fill_value='exterpolate')
        interpolated_prices = interp_func(strike_range)
        
    return interpolated_df




#NOTE: This script was discarded in the end as the interpolation was not working as expected.
#NOTE: the gaps of missing contracts were too significant to be interpolated, and to interpolate them, 
# the implied volatility would have to be entirely synthesized.
# instead of working with 80% synthetic data, and 20% real data, the entire dataset will now be synthesized.

def child_interpolate_vol_and_price(daily_df, vix_price, current_date, cp_flag):
    """
    Interpolate or extrapolate implied volatility for missing strike prices and 
    calculate avg_price using the black_scholes function.

    Parameters:
    daily_df (DataFrame): DataFrame with columns 'date', 'strike_price', 'exdate', 'avg_price', 'cp_flag', 'impl_volatility'
    vix_price (float): Current VIX price
    current_date (str): The current date in the iteration
    cp_flag (str): The cp_flag of the child DataFrame "call" or "put"
    black_scholes (function): The black_scholes function for price calculation

    Returns:
    DataFrame: Updated DataFrame with interpolated rows for missing strike prices
    """
    # #NOTE: This script was discarded in the end as the interpolation was not working as expected.
    #NOTE: the gaps of missing contracts were too significant to be interpolated, and to interpolate them, 
    # the implied volatility would have to be entirely synthesized.
    # instead of working with 80% synthetic data, and 20% real data, the entire dataset will now be synthesized.

    min_strike = max(vix_price - 5, daily_df['strike_price'].min())
    max_strike = min(vix_price + 5, daily_df['strike_price'].max())
    strike_range = np.arange(min_strike, max_strike + 0.5, 0.5)

    # Convert expiration dates to numerical values (days to expiration)
    daily_df['days_to_expiration'] = (pd.to_datetime(daily_df['exdate']) - pd.to_datetime(current_date)).dt.days
    exdate_range = daily_df['days_to_expiration'].unique()
    print("Daily df: \n", daily_df)
    daily_df.to_csv("daily_df.csv", index=False)

    # Prepare data for 2D interpolation
    points = daily_df[['strike_price', 'days_to_expiration']].dropna().values
    values = daily_df['impl_volatility'].dropna().values
    strike_grid, exdate_grid = np.meshgrid(strike_range, exdate_range, indexing='ij')

    interpolated_vols = griddata(points, values, (strike_grid, exdate_grid), method='linear')

    # Creating DataFrame from interpolated data
    interpolated_df = pd.DataFrame({
        'strike_price': strike_grid.ravel(),
        'days_to_expiration': exdate_grid.ravel(),
        'impl_volatility': interpolated_vols.ravel()
    })

    # Add 'exdate' and 'cp_flag' back to the DataFrame
    interpolated_df['exdate'] = pd.to_datetime(current_date) + pd.to_timedelta(interpolated_df['days_to_expiration'], 'D')
    interpolated_df['cp_flag'] = cp_flag

    # Calculating avg_price using the black_scholes function
    interpolated_df['avg_price'] = interpolated_df.apply(
        lambda row: black_scholes(
            current_price=vix_price,
            strike_price=row['strike_price'], 
            current_date=current_date, 
            expiry_date=row['exdate'], 
            risk_free_rate=0.01, 
            volatility=row['impl_volatility'],
            option_type=cp_flag, 
            dividend_yield=0
        ) if pd.notna(row['impl_volatility']) else np.nan,
        axis=1)

    # Combine interpolated data with original data (turned out to be messy, hence s0_5_synthesize_VIX_option_prices.py was made...)
    combined_df = pd.concat([daily_df, interpolated_df]).drop_duplicates(subset=['strike_price', 'exdate'])

    return combined_df


#NOTE: This script was discarded in the end as the interpolation was not working as expected.
#NOTE: the gaps of missing contracts were too significant to be interpolated, and to interpolate them, 
# the implied volatility would have to be entirely synthesized.
# instead of working with 80% synthetic data, and 20% real data, the entire dataset will now be synthesized.

def interpolate_vol_and_price(df, OHLC_df):
    all_dates = df["date"].unique()
    list_of_dfs = []
    for date in tqdm(all_dates):

        df_for_date = df[df["date"] == date]
        vix_price = OHLC_df[OHLC_df["Date"] == date]["Close"].iloc[0]

        for cp_flag in ["call", "put"]:
            cp_df_for_date = df_for_date[df_for_date['cp_flag'] == cp_flag]
            interpolated_df = child_interpolate_vol_and_price(daily_df = cp_df_for_date, 
                                                            vix_price=vix_price, 
                                                            current_date=date, cp_flag=cp_flag)
            list_of_dfs.append(interpolated_df)

            print("")
            print(interpolated_df)
            print("")

            exit()
            # Never finished it as it was clear that the interpolation was not working as expected when debugging the code.

    return df


if __name__ == "__main__":

    #NOTE: This script was discarded in the end as the interpolation was not working as expected.
    #NOTE: the gaps of missing contracts were too significant to be interpolated, and to interpolate them, 
    # the implied volatility would have to be entirely synthesized.
    # instead of working with 80% synthetic data, and 20% real data, the entire dataset will now be synthesized.

    range_type = "all"
    OHLC_df = load_specific_data(range_type = range_type, version = 3)
    OHLC_df = OHLC_df["all"][["Date", "Open", "High", "Low", "Close"]]

    options_listings_df = basic_filtering_VIX_options_listings()
    options_listings_df.to_csv("Data/Options_price_data/Curated_VIX_historical_options_listings.csv", index=False)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    interpolated_df = interpolate_vol_and_price(df = options_listings_df, OHLC_df= OHLC_df)

    