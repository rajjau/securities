#!/usr/bin/env python
from numpy import abs, where, sin, cos, pi
from pandas import concat, DataFrame, to_datetime

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
    data['close_greater_than_open'] = (data['c'] > data['o']).astype('int32')
    # Return the $data with the new column.
    return data

class FeatureEngineering:
    def __init__(self, data):
        # Make data a class variable.
        self.data = data
        # Create the grouped object once to be shared across functions.
        self.grouped = data.groupby('T')
        # Shifting via group to prevent look-ahead bias.
        self.prev_close = self.grouped['c'].shift(1).astype('float32')
        self.prev_open = self.grouped['o'].shift(1).astype('float32')
        self.prev_volume = self.grouped['v'].shift(1).astype('float32')
        self.prev_high = self.grouped['h'].shift(1).astype('float32')
        self.prev_low = self.grouped['l'].shift(1).astype('float32')
        # Dictionary to hold all new columns.
        self.new_columns = {}
        # Add new engineered features.
        self.add_time_and_date_features()
        self.add_overnight_features()
        self.add_lagged_features()
        self.add_technical_indicators()
        self.add_candlestick_features()
        self.add_interaction_features()
        # Join the new features to the original $data.
        self.data = concat([self.data, DataFrame(self.new_columns, index = self.data.index)], axis = 1)
        # De-fragment the DataFrame after a large concatenation to maintain performance.
        self.data = self.data.copy()

    def __call__(self):
        # Return the modified $data
        return(self.data)
    
    def add_time_and_date_features(self):
        # Display informational message to stdout.
        msg_info("Feature: Calculating time and date features.")
        # Convert timestamps to datetime.
        timestamps = to_datetime(self.data['t'], unit = 'ms')
        # Obtain the date only (setting time to 00:00:00).
        self.new_columns['t_d'] = timestamps.dt.normalize().astype('datetime64[s]')
        # Days and months related features.
        self.new_columns['day_of_week'] = timestamps.dt.day_name().astype('category')
        self.new_columns['day_of_month'] = timestamps.dt.day.astype('int32')
        self.new_columns['month'] = timestamps.dt.month.astype('int32')
        self.new_columns['quarter'] = timestamps.dt.quarter.astype('int32')
        self.new_columns['is_month_end'] = timestamps.dt.is_month_end.astype('int32')
        # Cyclical encoding using vectorized numpy functions.
        day_of_week_idx = timestamps.dt.dayofweek
        self.new_columns['dow_sin'] = sin(2 * pi * day_of_week_idx / 7).astype('float32')
        self.new_columns['dow_cos'] = cos(2 * pi * day_of_week_idx / 7).astype('float32')
        # Cyclical encoding for day of month.
        days_in_month = timestamps.dt.days_in_month
        self.new_columns['dom_sin'] = sin(2 * pi * self.new_columns['day_of_month'] / days_in_month).astype('float32')
        self.new_columns['dom_cos'] = cos(2 * pi * self.new_columns['day_of_month'] / days_in_month).astype('float32')

    def add_overnight_features(self):
        # Display informational message to stdout.
        msg_info('Feature: Calculating overnight gap and momentum features.')
        # Use vectorized 'where' to handle division by zero.
        self.new_columns['overnight_gap'] = where(self.prev_close != 0, (self.data['o'] - self.prev_close) / self.prev_close, 0).astype('float32')
        # Calculate Lagged Gap: Wrap the 1D gap array, group by ticker, shift, and extract 1D values.
        self.new_columns['lagged_overnight_gap_1'] = DataFrame(self.new_columns['overnight_gap']).groupby(self.data['T'])[0].shift(1).values
        # Significant Gap Flag: Binary indicator for gaps greater than 0.5% using numpy's absolute function.
        self.new_columns['is_significant_gap'] = (abs(self.new_columns['overnight_gap']) > 0.005).astype('int32')

    def add_lagged_features(self):
        # Display informational message to stdout.
        msg_info('Feature: Calculating lagged columns and derived lagged features.')
        # Define core columns to lag.
        core_columns = ['c', 'h', 'l', 'n', 'o', 'v', 'vw']
        # Add lagged day of week using shift to respect actual trading history.
        self.new_columns['lagged_day_of_week'] = self.new_columns['day_of_week'].shift(1)
        # Generate lagged OHLCV blocks and derived features.
        for i in range(1, LAGGED_MAX_DAYS + 1):
            # Shift the core OHLCV data by $i trading days within each symbol group.
            lagged_df = self.grouped[core_columns].shift(i)
            # Rename the columns to include the lag index $i to ensure unique names and clear lineage.
            lagged_df.columns = [f"lagged_{col}_{i}" for col in core_columns]
            # Add each column to the dictionary.
            for col in lagged_df.columns: self.new_columns[col] = lagged_df[col].astype('float32')
            # Compute derived lagged features.
            self.new_columns[f"lagged_high_low_spread_{i}"] = (lagged_df[f"lagged_h_{i}"] - lagged_df[f"lagged_l_{i}"]).astype('float32')
            # Compute daily return with zero-division protection.
            self.new_columns[f"lagged_daily_return_{i}"] = where(lagged_df[f"lagged_o_{i}"] != 0, (lagged_df[f"lagged_c_{i}"] - lagged_df[f"lagged_o_{i}"]) / lagged_df[f"lagged_o_{i}"], 0).astype('float32')

    def add_technical_indicators(self):
        # Display informational message to stdout.
        msg_info('Feature: Calculating technical indicators.')
        # Calculate the percentage change of the closing price within each symbol group.
        returns = self.grouped['c'].pct_change()
        # Compute all window-based indicators.
        for window in MOVING_AVERAGE_WINDOWS:
            # Simple Moving Average (SMA) Distance.
            sma_val = self.grouped['c'].transform(lambda entry: entry.rolling(window).mean()).shift(1).astype('float32')
            self.new_columns[f"dist_from_sma_{window}"] = where(sma_val != 0, (self.prev_close - sma_val) / sma_val, 0).astype('float32')
            # Exponential Moving Average (EMA) Distance.
            ema_val = self.grouped['c'].transform(lambda entry: entry.ewm(span=window, adjust = False).mean()).shift(1).astype('float32')
            self.new_columns[f"dist_from_ema_{window}"] = where(ema_val != 0, (self.prev_close - ema_val) / ema_val, 0).astype('float32')
            # Rate of Change (ROC).
            prev_window_c = self.grouped['c'].shift(window + 1).astype('float32')
            self.new_columns[f"roc_{window}"] = where(prev_window_c != 0, (self.prev_close - prev_window_c) / prev_window_c, 0).astype('float32')
            # Volatility (Standard Deviation of daily returns).
            self.new_columns[f"volatility_{window}"] = returns.groupby(self.data['T']).rolling(window).std().reset_index(level = 0, drop = True).shift(1).astype('float32')
            # Normalized Rolling Range.
            rolling_max = self.grouped['c'].transform(lambda entry: entry.rolling(window).max()).shift(1).astype('float32')
            rolling_min = self.grouped['c'].transform(lambda entry: entry.rolling(window).min()).shift(1).astype('float32')
            self.new_columns[f"norm_range_{window}"] = where(sma_val != 0, (rolling_max - rolling_min) / sma_val, 0).astype('float32')

    def add_candlestick_features(self):
        # Display informational message to stdout.
        msg_info('Feature: Calculating candlestick features.')
        # Compute features.
        self.new_columns['candle_body'] = (self.prev_close - self.prev_open).abs().astype('float32')
        self.new_columns['upper_shadow'] = self.prev_high - concat([self.prev_close, self.prev_open], axis = 1).max(axis = 1).astype('float32')
        self.new_columns['lower_shadow'] = (concat([self.prev_close, self.prev_open], axis = 1).min(axis = 1) - self.prev_low).astype('float32')
        # Relative body size with zero-division check.
        total_range = (self.prev_high - self.prev_low).astype('float32')
        self.new_columns['relative_body'] = where(total_range != 0, self.new_columns['candle_body'] / total_range, 0).astype('float32')

    def add_interaction_features(self):
        # Display informational message to stdout.
        msg_info('Feature: Calculating interaction features.')
        # Calculations.
        self.new_columns['close_volume'] = (self.prev_close * self.prev_volume).astype('float32')
        self.new_columns['return_volume'] = where(self.prev_open != 0, ((self.prev_close - self.prev_open) / self.prev_open) * self.prev_volume, 0).astype('float32')
        self.new_columns['spread_volume'] = ((self.prev_high - self.prev_low) * self.prev_volume).astype('float32')

############
### MAIN ###
############
def main(data):
    # Sort data by symbol and time once at the start.
    data = data.sort_values(by=['T', 't']).reset_index(drop = True)
    # Add feature if the closing price is greater than the opening price (no = 0; yes = 1).
    data = open_to_close(data=data)
    # Run feature engineering pipeline.
    data = FeatureEngineering(data=data)()
    # Return the modified $data.
    return data