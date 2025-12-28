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
    new_cols = {
        't_d': timestamps.dt.normalize(),
        'day_of_week': timestamps.dt.day_name(),
        'lagged_day_of_week': (timestamps - Timedelta(days=1)).dt.day_name()
    }
    # Add the new columns to $data.
    data = concat([data, DataFrame(new_cols)], axis=1)
    # Create a copy of $data to avoid SettingWithCopyWarning.
    data = data.copy()
    # Return the $data.
    return data

def add_overnight_features(data):
    # Display informational message to stdout.
    msg_info('Feature: Adding overnight gap and momentum features.')
    # Track existing columns before adding new ones.
    original_columns = set(data.columns)
    # Sort the DataFrame by symbol and timestamp.
    data = data.sort_values(by=['T', 't'])
    # Precompute groupby once for speed.
    grouped = data.groupby('T')
    # Calculate the Overnight Gap: (Today's Open - Yesterday's Close) / Yesterday's Close.
    prev_close = grouped['c'].shift(1)
    data['overnight_gap'] = (data['o'] - prev_close) / prev_close
    # Calculate Lagged Gap: The gap value from the previous trading day.
    data['lagged_overnight_gap_1'] = data['overnight_gap'].shift(1)
    # Significant Gap Flag: Binary indicator for gaps greater than 0.5%.
    data['is_significant_gap'] = (data['overnight_gap'].abs() > 0.005).astype(int)
    # Identify new columns created in this function.
    new_columns = set(data.columns) - original_columns
    # Convert only new numeric columns to float32.
    convert_to_float32(data=data, prefix=None, columns=new_columns)
    # Return the modified $data.
    return data

def add_lagged_features(data):
    # Display informational message to stdout.
    msg_info('Feature: Adding lagged columns and derived lagged features.')
    # Sort the DataFrame by symbol and timestamp to ensure correct rolling behavior.
    data = data.sort_values(by=['T', 't'])
    # Define core columns to lag.
    core_cols = ['c', 'h', 'l', 'n', 'o', 'v', 'vw']
    # Precompute groupby once for major speed improvement.
    grouped = data.groupby('T')
    # Initialize a list to hold lagged DataFrames for a single concatenation at the end.
    lag_list = []
    # Generate lagged OHLCV blocks and derived features.
    for i in range(1, LAGGED_MAX_DAYS + 1):
        # Shift the entire block of core columns for the current lag interval.
        lagged_df = grouped[core_cols].shift(i)
        # Rename the columns for the current lag interval.
        lagged_df.columns = [f"lagged_{col}_{i}" for col in core_cols]
        # Compute derived lagged features using the vectorized block.
        lagged_df[f"lagged_high_low_spread_{i}"] = lagged_df[f"lagged_h_{i}"] - lagged_df[f"lagged_l_{i}"]
        lagged_df[f"lagged_daily_return_{i}"] = (lagged_df[f"lagged_c_{i}"] - lagged_df[f"lagged_o_{i}"]) / lagged_df[f"lagged_o_{i}"]
        # Add the current lag block to the list.
        lag_list.append(lagged_df)
    # Concatenate all lagged blocks to the original data in a single operation to save memory and time.
    data = concat([data] + lag_list, axis=1)
    # Identify all newly created columns (all columns following the original 'quarter' or last calendar col).
    new_cols = data.columns.difference(['T', 't', 'c', 'h', 'l', 'n', 'o', 'v', 'vw', 't_d', 'day_of_week', 'lagged_day_of_week', 'overnight_gap', 'lagged_overnight_gap_1', 'is_significant_gap'])
    # Convert all new numeric columns to float32.
    convert_to_float32(data=data, prefix=None, columns=new_cols)
    # Create a copy of $data to avoid SettingWithCopyWarning.
    data = data.copy()
    # Return the modified DataFrame.
    return data

def add_technical_indicators(data):
    # Display informational message to stdout.
    msg_info('Feature: Adding technical indicators (SMA/EMA distance, ROC, volatility, normalized range).')
    # Track existing columns before adding new ones.
    original_columns = set(data.columns)
    # Sort the DataFrame by the symbol name and timestamp.
    data = data.sort_values(by=['T','t'])
    # Precompute groupby once for major speed improvement.
    grouped = data.groupby('T')
    # Precompute previous close and daily returns (SHIFTED to avoid leakage).
    prev_c = grouped['c'].shift(1)
    returns = grouped['c'].transform(lambda entry: entry.pct_change().shift(1))
    # Compute all window-based indicators.
    for window in MOVING_AVERAGE_WINDOWS:
        # Simple Moving Average (SMA) Distance - Shifted percentage distance from previous close to the mean.
        sma_val = grouped['c'].transform(lambda entry: entry.rolling(window).mean().shift(1))
        data[f"dist_from_sma_{window}"] = (prev_c - sma_val) / sma_val
        # Exponential Moving Average (EMA) Distance - Shifted percentage distance from previous close to the exponential mean.
        ema_val = grouped['c'].transform(lambda entry: entry.ewm(span=window, adjust=False).mean().shift(1))
        data[f"dist_from_ema_{window}"] = (prev_c - ema_val) / ema_val
        # Rate of Change (ROC) - The percentage change between the previous day and the start of the window.
        data[f"roc_{window}"] = grouped['c'].transform(lambda entry: ((entry.shift(1) - entry.shift(window+1)) / entry.shift(window+1)))
        # Volatility (Standard Deviation of Returns).
        data[f"volatility_{window}"] = returns.rolling(window).std()
        # Normalized Rolling Range - High/Low spread over the window normalized by the SMA to ensure stationarity.
        rolling_max = grouped['c'].transform(lambda entry: entry.rolling(window).max().shift(1))
        rolling_min = grouped['c'].transform(lambda entry: entry.rolling(window).min().shift(1))
        data[f"norm_range_{window}"] = (rolling_max - rolling_min) / sma_val
    # Identify new columns created in this function.
    new_columns = set(data.columns) - original_columns
    # Convert only new numeric columns to float32.
    convert_to_float32(data=data, prefix=None, columns=new_columns)
    # Return modified $data.
    return data

def add_candlestick_features(data):
    # Display informational message to stdout.
    msg_info('Feature: Adding candlestick features (body, shadows, relative body).')
    # Track existing columns before adding new ones.
    original_columns = set(data.columns)
    # Shifting: We calculate these based on the previous day's OHLC to prevent look-ahead bias.
    prev_c = data.groupby('T')['c'].shift(1)
    prev_o = data.groupby('T')['o'].shift(1)
    prev_h = data.groupby('T')['h'].shift(1)
    prev_l = data.groupby('T')['l'].shift(1)
    # Compute candle body size.
    data['candle_body'] = (prev_c - prev_o).abs()
    # Compute upper shadow.
    data['upper_shadow'] = prev_h - concat([prev_c, prev_o], axis=1).max(axis=1)
    # Compute lower shadow.
    data['lower_shadow'] = concat([prev_c, prev_o], axis=1).min(axis=1) - prev_l
    # Compute relative body size.
    data['relative_body'] = data['candle_body'] / (prev_h - prev_l)
    # Identify new columns created in this function.
    new_columns = set(data.columns) - original_columns
    # Convert only new numeric columns to float32.
    convert_to_float32(data=data, prefix=None, columns=new_columns)
    # Return modified DataFrame.
    return data

def add_interaction_features(data):
    # Display informational message to stdout.
    msg_info('Feature: Adding interaction features (price-volume, return-volume, spread-volume).')
    # Track existing columns before adding new ones.
    original_columns = set(data.columns)
    # Using previous day's data to prevent lookahead issues.
    prev_c = data.groupby('T')['c'].shift(1)
    prev_o = data.groupby('T')['o'].shift(1)
    prev_v = data.groupby('T')['v'].shift(1)
    prev_h = data.groupby('T')['h'].shift(1)
    prev_l = data.groupby('T')['l'].shift(1)
    # Compute price * volume interaction.
    data['close_volume'] = prev_c * prev_v
    # Compute return * volume interaction.
    data['return_volume'] = ((prev_c - prev_o) / prev_o) * prev_v
    # Compute spread * volume interaction.
    data['spread_volume'] = (prev_h - prev_l) * prev_v
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
    # Extract month, day, quarter.
    data['day_of_month'] = timestamps.dt.day
    data['month'] = timestamps.dt.month
    data['quarter'] = timestamps.dt.quarter
    data['is_month_end'] = timestamps.dt.is_month_end.astype(int)
    # Cyclical encoding for day of week (7-day cycle).
    day_of_week = timestamps.dt.dayofweek
    data['dow_sin'] = (2 * pi * day_of_week / 7).map(sin)
    data['dow_cos'] = (2 * pi * day_of_week / 7).map(cos)
    # Cyclical encoding for day of month.
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
    # Add string labels that describe when the opening value is less, equal, or greater than the closing value.
    data = open_to_close(data=data)
    # Calculate the different aspects of time for each row based on the timestamp ('t') value.
    data = time(data=data)
    # Add overnight gap features to establish the start-state of the trading day.
    data = add_overnight_features(data=data)
    # Add features from previous days.
    data = add_lagged_features(data=data)
    # Add various technical indicators using stationary (percentage-based) calculations.
    data = add_technical_indicators(data=data)
    # Add candlestick features.
    data = add_candlestick_features(data=data)
    # Add interaction features.
    data = add_interaction_features(data=data)
    # Add calendar features.
    data = add_calendar_features(data=data)
    # Return the modified $data.
    return data