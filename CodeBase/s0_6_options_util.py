import pandas as pd
from datetime import datetime, timedelta
from s1_0_Dataloader import *
from s1_1_predictionsloader import *

#NOTE: This script was discarded in the end as it was decided to not includes options in the final model.
#NOTE: This script was discarded in the end as it was decided to not includes options in the final model.
#NOTE: This script was discarded in the end as it was decided to not includes options in the final model.
#NOTE: This script was discarded in the end as it was decided to not includes options in the final model.

def load_VIX_data():
    return pd.read_csv("Data/Options_price_data/Curated_VIX_historical_options_listings.csv")


def calc_time_to_expiration(current_date, expiration_date):
    """
    Calculate the time to expiration in days.

    Parameters:
    current_date (str): The current date (format: YYYY-MM-DD).
    expiration_date (str): The expiration date (format: YYYY-MM-DD).

    Returns:
    float: Time to expiration in days.
    """
    print("current_date: ", current_date)
    print("expiration_date: ", expiration_date)

    # Converting date strings to datetime
    current_date = datetime.strptime(current_date, '%Y-%m-%d')
    expiration_date = datetime.strptime(expiration_date, '%Y-%m-%d')

    # Calculating time to expiration in days
    time_to_expiration = (expiration_date - current_date).days

    return time_to_expiration

#NOTE: This script was discarded in the end as it was decided to not includes options in the final model.
#NOTE: This script was discarded in the end as it was decided to not includes options in the final model.
#NOTE: This script was discarded in the end as it was decided to not includes options in the final model.
#NOTE: This script was discarded in the end as it was decided to not includes options in the final model.


def find_nearest_expiry_date(current_date, diff= 30):
    """
    There is an expiry date every Wednesday of the week.
    The function finds the nearest expiry date to the current date, with atleast 30 days difference.
    """
    # Convert date strings to datetime
    current_date = datetime.strptime(current_date, '%Y-%m-%d')

    nearest_wednesday = current_date + timedelta(days=diff-current_date.weekday())
    nearest_wednesday = nearest_wednesday.strftime('%Y-%m-%d')

    return nearest_wednesday
    
#NOTE: This script was discarded in the end as it was decided to not includes options in the final model.
#NOTE: This script was discarded in the end as it was decided to not includes options in the final model.
#NOTE: This script was discarded in the end as it was decided to not includes options in the final model.
#NOTE: This script was discarded in the end as it was decided to not includes options in the final model.

def find_nearest_strike_price(current_price, diff= 2):
    """
    Find the nearest strike price to the current price.

    Parameters:
    current_price (float): Current price of VIX
    diff (float): The difference between the strike price and the current price
    """

#NOTE: This script was discarded in the end as it was decided to not includes options in the final model.
#NOTE: This script was discarded in the end as it was decided to not includes options in the final model.
#NOTE: This script was discarded in the end as it was decided to not includes options in the final model.
#NOTE: This script was discarded in the end as it was decided to not includes options in the final model.

def find_VIX_option_price_using_data(current_date, expiry_date, strike_price, option_type='C'):

    raise DeprecationWarning("This function is deprecated, due to insufficient data. Many combination of strike price and expiry date are not available.")
    global options_price_data

    strike_price = float(round(strike_price,1))

    try:
        tmp_options_price_data = options_price_data.copy()
        tmp_options_price_data = tmp_options_price_data[tmp_options_price_data['strike_price'] == strike_price]
        print("tmp_options_price_data len: ", len(tmp_options_price_data))
        tmp_options_price_data = tmp_options_price_data[tmp_options_price_data['date'] == current_date]
        print("tmp_options_price_data len: ", len(tmp_options_price_data))
        print("tmp_options_price_data: \n", tmp_options_price_data)
        tmp_options_price_data = tmp_options_price_data[tmp_options_price_data['exdate'] == expiry_date]
        print("tmp_options_price_data len: ", len(tmp_options_price_data))
        tmp_options_price_data = tmp_options_price_data[tmp_options_price_data['option_type'] == option_type]

        print("tmp_options_price_data: \n", tmp_options_price_data)

    except KeyError:
        return None


if __name__ == "__main__":
    
    options_price_data = load_VIX_data()

    print("TUESDAY, Example find_nearest_expiry_date('2023-11-21', diff= 30): ", find_nearest_expiry_date("2023-11-21", diff= 30))
    print("WEDNESDAY, Example find_nearest_expiry_date('2023-11-22', diff= 30): ", find_nearest_expiry_date("2023-11-22", diff= 30))
    print("THURSDAY, Example find_nearest_expiry_date('2023-11-23', diff= 30): ", find_nearest_expiry_date("2023-11-23", diff= 30))
    print("\n")
    print("Preparing data frame")
    model_name = "LSTM_all_features_custom_loss"
    specific_model_name = "LSTM_78"
    range_type = "test"
    merged_df = local_prepare_dataframe(model_name, specific_model_name, range_type)
    print("merged_df head: \n", merged_df.head())
    print("")
    print("merged_df tail: \n", merged_df.tail())
    print("\n")
    print("Example calc_time_to_expiration('2022-05-01', '2022-06-15'): ", calc_time_to_expiration('2022-05-01', '2022-06-15'))
    print("Example calc_time_to_expiration('2021-11-21', '2021-12-04'): ", calc_time_to_expiration('2021-11-21', '2021-12-04'))
    print("Example calc_time_to_expiration('2021-11-21', '2021-12-06'): ", calc_time_to_expiration('2021-11-21', '2021-12-06'))
    print("\n")
 