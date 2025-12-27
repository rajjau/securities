#!/usr/bin/env python
from math import sin, cos, pi
from pandas import concat, DataFrame, Timedelta, to_datetime

######################
### CUSTOM MODULES ###
######################
from modules.messages import msg_info

################
### SETTINGS ###
################
LAGGED_MAX_DAYS = 15
MOVING_AVERAGE_WINDOWS = [5, 10, 15, 20, 50]

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

def convert_to_float32(data, prefix = None, columns = None):
    """Convert 64bit columns to 32bit to save memory."""
    # If the user provided a $prefix but not a column list, then filter columns from $data that start with that prefix.
    if prefix and not columns:
        columns = [c for c in data.columns if c.startswith(prefix)]
    # If both $prefix and $columns are provided, filter $columns to only those that start with $prefix.
    elif prefix and columns:
        columns = [c for c in columns if c.startswith(prefix)]
    # Iterate through each specified column.
    for col in columns:
        # Convert only if the column exists in $data and is numeric.
        if col in data.columns and data[col].dtype in ['float64', 'int64']: data[col] = data[col].astype('float32')
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

def add_lagged_features(data):
    # Display informational message to stdout.
    msg_info("Feature: Adding lagged columns and derived lagged features.")
    # Track existing columns before adding new ones.
    original_columns = set(data.columns)
    # Sort the DataFrame by symbol and timestamp to ensure correct rolling behavior.
    data = data.sort_values(by=["T", "t"])
    # Convert core OHLCV columns to float32 for memory efficiency.
    for col in ['c', 'h', 'l', 'n', 'o', 'v', 'vw']: data[col] = data[col].astype("float32")
    # Precompute groupby once for major speed improvement.
    grouped = data.groupby("T")
    # Define a dictionary to hold all new lagged and derived columns.
    new_cols = {}
    # Generate lagged OHLCV columns and derived features in a single pass.
    for previous_day in range(1, LAGGED_MAX_DAYS + 1):
        # Create lagged OHLCV columns.
        for col in ['c', 'h', 'l', 'n', 'o', 'v', 'vw']: new_cols[f"lagged_{col}_{previous_day}"] = grouped[col].shift(previous_day)
        # Compute derived lagged features using the newly created lagged columns.
        lagged_high = grouped['h'].shift(previous_day)
        lagged_low = grouped['l'].shift(previous_day)
        lagged_close = grouped['c'].shift(previous_day)
        lagged_open = grouped['o'].shift(previous_day)
        new_cols[f"lagged_high_low_spread_{previous_day}"] = lagged_high - lagged_low
        new_cols[f"lagged_daily_return_{previous_day}"] = (lagged_close - lagged_open) / lagged_open
    # Add all new columns to the DataFrame.
    data = concat([data, DataFrame(new_cols)], axis=1)
    # Identify new columns created in this function.
    new_columns = set(data.columns) - original_columns
    # Convert only new numeric columns to float32.
    convert_to_float32(data=data, prefix=None, columns=new_columns)
    # Create a copy of $data to avoid SettingWithCopyWarning.
    data = data.copy()
    # Return the modified DataFrame.
    return data

def add_technical_indicators(data):
    # Display informational message to stdout.
    msg_info("Feature: Adding technical indicators (SMA, EMA, momentum, ROC, volatility, rolling stats, z-scores).")
    # Track existing columns before adding new ones.
    original_columns = set(data.columns)
    # Sort the DataFrame by the symbol name and timestamp. This groups stocks and ensures the data for a given stock is in order from earliest to most recent.
    data = data.sort_values(by=['T','t'])
    # Precompute groupby once for major speed improvement.
    grouped = data.groupby('T')
    # Precompute daily returns (SHIFTED to avoid leakage).
    returns = grouped['c'].transform(lambda entry: entry.pct_change().shift(1))
    # Compute all window-based indicators.
    for window in MOVING_AVERAGE_WINDOWS:
        # Simple Moving Average (SMA) - Shifted to use previous window.
        data[f"sma_{window}"] = grouped['c'].transform(lambda entry: entry.rolling(window).mean().shift(1))
        # Exponential Moving Average (EMA) - Shifted to use previous window.
        data[f"ema_{window}"] = grouped['c'].transform(lambda entry: entry.ewm(span=window,adjust=False).mean().shift(1))
        # Momentum - Shifted to use previous window.
        data[f"momentum_{window}"] = grouped['c'].transform(lambda entry: (entry - entry.shift(window)).shift(1))
        # Rate of Change (ROC) - Shifted to use previous window.
        data[f"roc_{window}"] = grouped['c'].transform(lambda entry: ((entry - entry.shift(window)) / entry.shift(window)).shift(1))
        # Volatility (Standard Deviation of Returns).
        data[f"volatility_{window}"] = returns.rolling(window).std()
        # Rolling Maximum, Minimum, Range - Shifted to avoid including today's high/low.
        data[f"rolling_max_close_{window}"] = grouped['c'].transform(lambda entry: entry.rolling(window).max().shift(1))
        data[f"rolling_min_close_{window}"] = grouped['c'].transform(lambda entry: entry.rolling(window).min().shift(1))
        data[f"rolling_range_{window}"] = data[f"rolling_max_close_{window}"] - data[f"rolling_min_close_{window}"]
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        # Z-Score of Returns.
        data[f"zscore_return_{window}"] = (returns - rolling_mean) / rolling_std
    # Identify new columns created in this function.
    new_columns = set(data.columns) - original_columns
    # Convert only new numeric columns to float32.
    convert_to_float32(data=data, prefix=None, columns=new_columns)
    # Return modified DataFrame
    return data

def add_candlestick_features(data):
    # Display informational message to stdout.
    msg_info("Feature: Adding candlestick features (body, shadows, relative body).")
    # Track existing columns before adding new ones.
    original_columns = set(data.columns)
    # Shifting: We calculate these based on the previous day's OHLC to prevent look-ahead bias.
    prev_c = data.groupby('T')['c'].shift(1)
    prev_o = data.groupby('T')['o'].shift(1)
    prev_h = data.groupby('T')['h'].shift(1)
    prev_l = data.groupby('T')['l'].shift(1)
    # Compute candle body size.
    data["candle_body"] = (prev_c - prev_o).abs()
    # Compute upper shadow.
    data["upper_shadow"] = prev_h - concat([prev_c, prev_o], axis=1).max(axis=1)
    # Compute lower shadow.
    data["lower_shadow"] = concat([prev_c, prev_o], axis=1).min(axis=1) - prev_l
    # Compute relative body size.
    data["relative_body"] = data["candle_body"] / (prev_h - prev_l)
    # Identify new columns created in this function.
    new_columns = set(data.columns) - original_columns
    # Convert only new numeric columns to float32.
    convert_to_float32(data=data, prefix=None, columns=new_columns)
    # Return modified DataFrame.
    return data

def add_interaction_features(data):
    # Display informational message to stdout.
    msg_info("Feature: Adding interaction features (price-volume, return-volume, spread-volume).")
    # Track existing columns before adding new ones.
    original_columns = set(data.columns)
    # Using previous day's data to prevent lookahead issues.
    prev_c = data.groupby('T')['c'].shift(1)
    prev_o = data.groupby('T')['o'].shift(1)
    prev_v = data.groupby('T')['v'].shift(1)
    prev_h = data.groupby('T')['h'].shift(1)
    prev_l = data.groupby('T')['l'].shift(1)
    # Compute price * volume interaction.
    data["close_volume"] = prev_c * prev_v
    # Compute return * volume interaction.
    data["return_volume"] = ((prev_c - prev_o) / prev_o) * prev_v
    # Compute spread * volume interaction.
    data["spread_volume"] = (prev_h - prev_l) * prev_v
    # Identify new columns created in this function.
    new_columns = set(data.columns) - original_columns
    # Convert only new numeric columns to float32.
    convert_to_float32(data=data, prefix=None, columns=new_columns)
    # Return modified DataFrame.
    return data

def add_calendar_features(data):
    # Display informational message to stdout.
    msg_info("Feature: Adding calendar and cyclical time features.")
    # Track existing columns before adding new ones.
    original_columns = set(data.columns)
    # Convert timestamps to datetime.
    timestamps = to_datetime(data['t'], unit='ms')
    # Extract day of month.
    data['day_of_month'] = timestamps.dt.day
    # Extract month.
    data['month'] = timestamps.dt.month
    # Extract quarter.
    data['quarter'] = timestamps.dt.quarter
    # Month-end flag.
    data['is_month_end'] = timestamps.dt.is_month_end.astype(int)
    # Cyclical encoding for day of week (7-day cycle).
    day_of_week = timestamps.dt.dayofweek
    data['dow_sin'] = (2 * pi * day_of_week / 7).map(sin)
    data['dow_cos'] = (2 * pi * day_of_week / 7).map(cos)
    # Cyclical encoding for day of month (variable-length cycle).
    day_of_month = timestamps.dt.day
    days_in_month = timestamps.dt.days_in_month
    data['dom_sin'] = (2 * pi * day_of_month / days_in_month).map(sin)
    data['dom_cos'] = (2 * pi * day_of_month / days_in_month).map(cos)
    # Identify new columns created in this function.
    new_columns = set(data.columns) - original_columns
    # Convert only new numeric columns to float32.
    convert_to_float32(data=data, prefix=None, columns=new_columns)
    # Return modified DataFrame.
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
    data = add_lagged_features(data=data)
    # Add various technical indicators.
    data = add_technical_indicators(data=data)
    # Add candlestick features.
    data = add_candlestick_features(data=data)
    # Add interaction features.
    data = add_interaction_features(data=data)
    # Add calendar features.
    data = add_calendar_features(data=data)
    # Remove NaN rows.
    data = data.dropna()
    # Ensure the data is still sorted correctly after dropping rows
    data = data.sort_values(by=['T', 't']).reset_index(drop = True)
    # Return the modified $data.
    return data