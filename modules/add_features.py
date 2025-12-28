#!/usr/bin/env python
from numpy import abs, where, sin, cos, pi
from pandas import concat, DataFrame, Series, to_datetime

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
    timestamps = to_datetime(data['t'], unit = 'ms')
    # Create new columns:
    new_cols = {
        't_d': timestamps.dt.normalize(),
        'day_of_week': timestamps.dt.day_name()
    }
    # Add the new columns to $data.
    data = concat([data, DataFrame(new_cols, index=data.index)], axis = 1)
    # Create a copy of $data to avoid SettingWithCopyWarning.
    data = data.copy()
    # Return the $data.
    return data

def add_overnight_features(data, grouped):
    # Display informational message to stdout.
    msg_info('Feature: Adding overnight gap and momentum features.')
    # Track existing columns before adding new ones.
    original_columns = set(data.columns)
    # Initialize a dictionary to collect all new features for a single concatenation.
    new_cols_dict = {}
    # Calculate the Overnight Gap: (Today's Open - Yesterday's Close) / Yesterday's Close.
    prev_close = grouped['c'].shift(1)
    # Use vectorized 'where' to handle division by zero.
    new_cols_dict['overnight_gap'] = where(prev_close != 0, (data['o'] - prev_close) / prev_close, 0)
    # Calculate Lagged Gap: Wrap the 1D gap array, group by ticker, shift, and extract 1D values.
    new_cols_dict['lagged_overnight_gap_1'] = DataFrame(new_cols_dict['overnight_gap']).groupby(data['T'])[0].shift(1).values
    # Significant Gap Flag: Binary indicator for gaps greater than 0.5% using numpy's absolute function.
    new_cols_dict['is_significant_gap'] = (abs(new_cols_dict['overnight_gap']) > 0.005).astype(int)
    # Join the new features to the original $data.
    data = concat([data, DataFrame(new_cols_dict, index=data.index)], axis = 1)
    # Identify new columns created in this function.
    new_columns = set(data.columns) - original_columns
    # Convert only new numeric columns to float32.
    convert_to_float32(data=data, prefix=None, columns=new_columns)
    # Return the modified $data.
    return data

def add_lagged_features(data, grouped):
    # Display informational message to stdout.
    msg_info('Feature: Adding lagged columns and derived lagged features.')
    # Define core columns to lag.
    core_columns = ['c', 'h', 'l', 'n', 'o', 'v', 'vw']
    # Add lagged day of week using shift to respect actual trading history.
    data['lagged_day_of_week'] = grouped['day_of_week'].shift(1)
    # Track existing columns before adding new ones.
    original_columns = set(data.columns)
    # Initialize a list to hold lagged DataFrames for a single concatenation.
    lag_list = []
    # Generate lagged OHLCV blocks and derived features.
    for i in range(1, LAGGED_MAX_DAYS + 1):
        # Shift the core OHLCV data by $i trading days within each symbol group.
        lagged_df = grouped[core_columns].shift(i)
        # Rename the columns to include the lag index $i to ensure unique names and clear lineage.
        lagged_df.columns = [f"lagged_{col}_{i}" for col in core_columns]
        # Compute derived lagged features.
        lagged_df[f"lagged_high_low_spread_{i}"] = lagged_df[f"lagged_h_{i}"] - lagged_df[f"lagged_l_{i}"]
        # Compute daily return with zero-division protection.
        lagged_df[f"lagged_daily_return_{i}"] = where(lagged_df[f"lagged_o_{i}"] != 0, (lagged_df[f"lagged_c_{i}"] - lagged_df[f"lagged_o_{i}"]) / lagged_df[f"lagged_o_{i}"], 0)
        # Add the DataFrame to the list.
        lag_list.append(lagged_df)
    # Concatenate the original $data with all generated lag blocks at once.
    data = concat([data] + lag_list, axis = 1)
    # Automatically identify new columns by finding the difference between current columns and $cols_before.
    new_columns = [c for c in data.columns if c not in original_columns]
    # Convert all newly identified numeric columns to float32.
    convert_to_float32(data=data, prefix=None, columns=new_columns)
    # De-fragment the DataFrame after a large concatenation to maintain performance.
    data = data.copy()
    # Return the modified $data.
    return data

def add_technical_indicators(data, grouped):
    # Display informational message to stdout.
    msg_info('Feature: Adding technical indicators (SMA/EMA distance, ROC, volatility, normalized range).')
    # Track existing columns before adding new ones.
    original_columns = set(data.columns)
    # Precompute previous close and daily returns (SHIFTED to avoid leakage).
    prev_c = grouped['c'].shift(1)
    # Calculate the percentage change of the closing price within each symbol group.
    returns = grouped['c'].pct_change()
    # Initialize a dictionary to collect all new window-based features for a single concatenation.
    new_cols_dict = {}
    # Compute all window-based indicators.
    for window in MOVING_AVERAGE_WINDOWS:
        # Simple Moving Average (SMA) Distance.
        sma_val = grouped['c'].transform(lambda entry: entry.rolling(window).mean()).shift(1)
        new_cols_dict[f"dist_from_sma_{window}"] = where(sma_val != 0, (prev_c - sma_val) / sma_val, 0)
        # Exponential Moving Average (EMA) Distance.
        ema_val = grouped['c'].transform(lambda entry: entry.ewm(span=window, adjust = False).mean()).shift(1)
        new_cols_dict[f"dist_from_ema_{window}"] = where(ema_val != 0, (prev_c - ema_val) / ema_val, 0)
        # Rate of Change (ROC).
        prev_window_c = grouped['c'].shift(window + 1)
        new_cols_dict[f"roc_{window}"] = where(prev_window_c != 0, (prev_c - prev_window_c) / prev_window_c, 0)
        # Volatility (Standard Deviation of daily returns).
        new_cols_dict[f"volatility_{window}"] = returns.groupby(data['T']).rolling(window).std().reset_index(level = 0, drop = True).shift(1)
        # Normalized Rolling Range.
        rolling_max = grouped['c'].transform(lambda entry: entry.rolling(window).max()).shift(1)
        rolling_min = grouped['c'].transform(lambda entry: entry.rolling(window).min()).shift(1)
        new_cols_dict[f"norm_range_{window}"] = where(sma_val != 0, (rolling_max - rolling_min) / sma_val, 0)
    # Convert the dictionary of results into a DataFrame and join it to the original $data.
    data = concat([data, DataFrame(new_cols_dict, index=data.index)], axis = 1)
    # Identify new columns created in this function.
    new_columns = set(data.columns) - original_columns
    # Convert only new numeric columns to float32.
    convert_to_float32(data=data, prefix=None, columns=new_columns)
    # Create a copy of $data to consolidate the memory layout and prevent performance warnings.
    data = data.copy()
    # Return modified $data.
    return data

def add_candlestick_features(data, grouped):
    # Display informational message to stdout.
    msg_info('Feature: Adding candlestick features (body, shadows, relative body).')
    # Track existing columns before adding new ones.
    original_columns = set(data.columns)
    # Initialize a dictionary to collect all new features for a single concatenation.
    new_cols_dict = {}
    # Shifting via group to prevent look-ahead bias.
    prev_c = grouped['c'].shift(1)
    prev_o = grouped['o'].shift(1)
    prev_h = grouped['h'].shift(1)
    prev_l = grouped['l'].shift(1)
    # Compute features.
    new_cols_dict['candle_body'] = (prev_c - prev_o).abs()
    new_cols_dict['upper_shadow'] = prev_h - concat([prev_c, prev_o], axis = 1).max(axis = 1)
    new_cols_dict['lower_shadow'] = concat([prev_c, prev_o], axis = 1).min(axis = 1) - prev_l
    # Relative body size with zero-division check.
    total_range = prev_h - prev_l
    new_cols_dict['relative_body'] = where(total_range != 0, new_cols_dict['candle_body'] / total_range, 0)
    # Join the new features to the original $data.
    data = concat([data, DataFrame(new_cols_dict, index=data.index)], axis = 1)
    # Identify new columns created in this function.
    new_columns = set(data.columns) - original_columns
    # Convert only new numeric columns to float32.
    convert_to_float32(data=data, prefix=None, columns=new_columns)
    # Return modified DataFrame.
    return data

def add_interaction_features(data, grouped):
    # Display informational message to stdout.
    msg_info('Feature: Adding interaction features (price-volume, return-volume, spread-volume).')
    # Track existing columns before adding new ones.
    original_columns = set(data.columns)
    # Initialize a dictionary to collect all new features for a single concatenation.
    new_cols_dict = {}
    # Shifting via group to prevent look-ahead bias.
    prev_c = grouped['c'].shift(1)
    prev_o = grouped['o'].shift(1)
    prev_v = grouped['v'].shift(1)
    prev_h = grouped['h'].shift(1)
    prev_l = grouped['l'].shift(1)
    # Calculations.
    new_cols_dict['close_volume'] = prev_c * prev_v
    new_cols_dict['return_volume'] = where(prev_o != 0, ((prev_c - prev_o) / prev_o) * prev_v, 0)
    new_cols_dict['spread_volume'] = (prev_h - prev_l) * prev_v
    # Join the new features to the original $data.
    data = concat([data, DataFrame(new_cols_dict, index=data.index)], axis = 1)
    # Identify new columns created in this function.
    new_columns = set(data.columns) - original_columns
    # Convert only new numeric columns to float32.
    convert_to_float32(data=data, prefix=None, columns=new_columns)
    # Return modified $data.
    return data

def add_calendar_features(data):
    # Display informational message to stdout.
    msg_info("Feature: Adding calendar and cyclical time features.")
    # Track existing columns before adding new ones.
    original_columns = set(data.columns)
    # Initialize a dictionary to collect all new features for a single concatenation.
    new_cols_dict = {}
    # Convert timestamps to datetime.
    timestamps = to_datetime(data['t'], unit = 'ms')
    # Extract components.
    new_cols_dict['day_of_month'] = timestamps.dt.day
    new_cols_dict['month'] = timestamps.dt.month
    new_cols_dict['quarter'] = timestamps.dt.quarter
    new_cols_dict['is_month_end'] = timestamps.dt.is_month_end.astype(int)
    # Cyclical encoding using vectorized numpy functions.
    day_of_week_idx = timestamps.dt.dayofweek
    new_cols_dict['dow_sin'] = sin(2 * pi * day_of_week_idx / 7)
    new_cols_dict['dow_cos'] = cos(2 * pi * day_of_week_idx / 7)
    # Cyclical encoding for day of month.
    days_in_month = timestamps.dt.days_in_month
    new_cols_dict['dom_sin'] = sin(2 * pi * new_cols_dict['day_of_month'] / days_in_month)
    new_cols_dict['dom_cos'] = cos(2 * pi * new_cols_dict['day_of_month'] / days_in_month)
    # Join the new features to the original $data.
    data = concat([data, DataFrame(new_cols_dict, index=data.index)], axis = 1)
    # Identify new columns created in this function.
    new_columns = set(data.columns) - original_columns
    # Convert only new numeric columns to float32.
    convert_to_float32(data=data, prefix=None, columns=new_columns)
    # Create a copy of $data to consolidate the memory layout and prevent performance warnings.
    data = data.copy()
    # Return modified DataFrame.
    return data

############
### MAIN ###
############
def main(data):
    # Sort data by symbol and time once at the start.
    data = data.sort_values(by=['T', 't']).reset_index(drop = True)
    # Add base time features.
    data = open_to_close(data=data)
    data = time(data=data)
    # Create the grouped object once to be shared across functions.
    grouped = data.groupby('T')
    # Run feature engineering pipeline.
    data = add_overnight_features(data=data, grouped=grouped)
    data = add_lagged_features(data=data, grouped=grouped)
    data = add_technical_indicators(data=data, grouped=grouped)
    data = add_candlestick_features(data=data, grouped=grouped)
    data = add_interaction_features(data=data, grouped=grouped)
    data = add_calendar_features(data=data)
    # Return the modified $data.
    return data