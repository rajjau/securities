#!/usr/bin/env python
from pandas import concat, DataFrame, Timedelta, to_datetime

######################
### CUSTOM MODULES ###
######################
from modules.messages import msg_info

################
### SETTINGS ###
################
LAGGED_MAX_DAYS = 15

#################
### FUNCTIONS ###
#################
def open_to_close(data):
    # Display informational message to stdout.
    msg_info('Feature: Adding opening to close value descriptions.')
    # All instances where the closing value is greater than the opening value.
    data['close_greater_than_open'] = (data['c'] > data['o']).astype(int)
    # Return the $data with the new column.
    return data

def convert_to_float32(data, prefix):
    """Convert all numeric columns starting with a given prefix to float32."""
    # Define all columns that start with the specified $prefix and are numeric.
    cols = [c for c in data.columns if c.startswith(prefix) and data[c].dtype in ['float64', 'int64']]
    # Convert the defined columns to float32.
    data[cols] = data[cols].astype('float32')
    # Return the modified $data.
    return data

def time(data):
    # Display informational message to stdout.
    msg_info('Feature: Adding time descriptions.')
    # Convert the column containing Unix millisecond timestamps into Pandas datetime format.
    timestamps = to_datetime(data['t'], unit='ms')
    # Create new columns:
    # 't_d': Full date (YYYY-MM-DD).
    # 'day_of_week': Day of the week name.
    # 'lagged_day_of_week': Previous day of the week name.
    new_cols = {
        't_d': timestamps.dt.normalize(),  # faster than dt.date
        'day_of_week': timestamps.dt.day_name(),
        'lagged_day_of_week': (timestamps - Timedelta(days=1)).dt.day_name()
    }
    # Add the new columns to $data.
    data = concat([data, DataFrame(new_cols)], axis=1)
    # Create a copy of $data to avoid SettingWithCopyWarning.
    data = data.copy()
    # Return the $data.
    return data

def lagged(data):
    # Display informational message to stdout.
    msg_info('Feature: Adding features from previous days.')
    # Sort the DataFrame by the symbol name and timestamp. This groups stocks and ensures the data for a given stock is in order from earliest to most recent.
    data = data.sort_values(by=['T', 't'])
    # Convert relevant columns to float32 to save memory.
    for col in ['c', 'h', 'l', 'n', 'o', 'v', 'vw']: data[col] = data[col].astype('float32')
    # Define a dictionary to hold the new columns.
    new_cols = {}
    # Precompute groupby once for major speed improvement.
    grouped = data.groupby('T')
    # Iterate through each previous day.
    for previous_day in range(1, LAGGED_MAX_DAYS + 1):
        # Iterate through each column to create lagged features for.
        for col in ['c', 'h', 'l', 'n', 'o', 'v', 'vw']:
            # Define the new column name.
            new_name = f"lagged_{col}_{previous_day}"
            # Obtain data from the specified previous day, ensuring to group by the symbol name.
            new_cols[new_name] = grouped[col].shift(previous_day)
    # Add the new columns to $data.
    data = concat([data, DataFrame(new_cols)], axis=1)
    # Convert only numeric lagged columns to float32. This should be done after adding all lagged columns.
    data = convert_to_float32(data=data, prefix='lagged_')
    # Create a copy of $data to avoid SettingWithCopyWarning.
    data = data.copy()
    # Return the $data.
    return data

def feature_engineering(data):
    # Display informational message to stdout.
    msg_info('Feature: Creating new features based off existing ones.')
    # Sort the DataFrame by the symbol name and timestamp. This groups stocks and ensures the data for a given stock is in order from earliest to most recent.
    data = data.sort_values(by=['T', 't'])
    # Define a dictionary to hold the new columns.
    new_cols = {}
    # Pre-fetch lagged columns for faster access.
    lag_h = data.filter(like = 'lagged_h_')
    lag_l = data.filter(like = 'lagged_l_')
    lag_c = data.filter(like = 'lagged_c_')
    lag_o = data.filter(like = 'lagged_o_')
    # Iterate through each previous day.
    for previous_day in range(1, LAGGED_MAX_DAYS + 1):
        # Obtain the lagged high, low, close, and open values.
        high_ = lag_h[f"lagged_h_{previous_day}"]
        low_ = lag_l[f"lagged_l_{previous_day}"]
        close_ = lag_c[f"lagged_c_{previous_day}"]
        open_ = lag_o[f"lagged_o_{previous_day}"]
        # Add the spread between the high and low prices.
        new_cols[f"lagged_high_low_spread_{previous_day}"] = high_ - low_
        # Add the return from the previous days.
        new_cols[f"lagged_daily_return_{previous_day}"] = (close_ - open_) / open_
    # Add the new columns to $data.
    data = concat([data, DataFrame(new_cols)], axis=1)
    # Convert engineered numeric columns to float32
    data = convert_to_float32(data=data, prefix='lagged_high_low_spread_')
    data = convert_to_float32(data=data, prefix='lagged_daily_return_')
    # Create a copy of $data to avoid SettingWithCopyWarning.
    data = data.copy()
    # Return the $data.
    return data

############
### MAIN ###
############
def main(data):
    # Add string labels that describe when the opening value is less, equal, or greater than the closing value. These will be converted to int using one-hot-encoding before machine learning.
    data = open_to_close(data=data)
    # Calculate the different aspects of time for each row based on the timestamp ('t') value.
    data = time(data=data)
    # Add features from previous days.
    data = lagged(data=data)
    # Create new features from the existing data.
    data = feature_engineering(data=data)
    # Return the modified $data.
    return data