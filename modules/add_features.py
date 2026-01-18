#!/usr/bin/env python
from numpy import abs, divide, where, sin, cos, ones_like, pi
from pandas import concat, DataFrame, to_datetime

######################
### CUSTOM MODULES ###
######################
# Import custom messaging module for stdout info.
from modules.messages import msg_info

################
### SETTINGS ###
################
# Set the maximum number of days for historical lag features.
LAGGED_MAX_DAYS = 15

# Define the lookback windows for moving averages and technical indicators.
MOVING_AVERAGE_WINDOWS = [5, 10, 15, 20, 50]

#################
### FUNCTIONS ###
#################
def open_to_close(data):
    # Display informational message regarding target creation to stdout.
    msg_info('Target: Adding binary classification target.')
    # Create a binary column: 1 if closing price is higher than opening price, else 0.
    data['close_greater_than_open'] = (data['c'] > data['o']).astype('int32')
    # Return the modified dataframe with the target label.
    return data

class FeatureEngineering:
    def __init__(self, data):
        # Assign the input dataframe to the class instance.
        self.data = data
        # Group data by ticker symbol and ensure categorical order is observed.
        self.grouped = data.groupby('T', observed=True)
        # Calculate yesterday's closing price per group.
        self.prev_close = self.grouped['c'].shift(1).astype('float32')
        # Calculate yesterday's opening price per group.
        self.prev_open = self.grouped['o'].shift(1).astype('float32')
        # Calculate yesterday's trading volume per group.
        self.prev_volume = self.grouped['v'].shift(1).astype('float32')
        # Calculate yesterday's high price per group.
        self.prev_high = self.grouped['h'].shift(1).astype('float32')
        # Calculate yesterday's low price per group.
        self.prev_low = self.grouped['l'].shift(1).astype('float32')
        # Initialize a dictionary to store newly generated feature columns.
        self.new_columns = {}
        # Call function to generate time-based cyclical features.
        self.add_time_and_date_features()
        # Call function to generate overnight price gap features.
        self.add_overnight_features()
        # Call function to generate historical price/volume lags.
        self.add_lagged_features()
        # Call function to generate technical indicator distances.
        self.add_technical_indicators()
        # Call function to generate candlestick pattern descriptions.
        self.add_candlestick_features()
        # Call function to generate volume/price interaction features.
        self.add_interaction_features()
        # Merge the new feature dictionary into a dataframe and join with original data.
        self.data = concat([self.data, DataFrame(self.new_columns, index = self.data.index)], axis = 1)
        # Create a copy of the dataframe to resolve fragmentation and consolidate memory.
        self.data = self.data.copy()

    def __call__(self):
        # Return the fully engineered dataframe when the class is called.
        return(self.data)
    
    def add_time_and_date_features(self):
        # Display informational message to stdout.
        msg_info("Feature: Calculating time and date features.")
        # Convert the 't' column from milliseconds to pandas datetime objects.
        timestamps = to_datetime(self.data['t'], unit = 'ms')
        # Extract and store the normalized date (midnight) for each record.
        self.new_columns['t_d'] = timestamps.dt.normalize().values
        # Extract the day of the month as an integer.
        self.new_columns['day_of_month'] = timestamps.dt.day.astype('int32').values
        # Extract the month of the year as an integer.
        self.new_columns['month'] = timestamps.dt.month.astype('int32').values
        # Extract the fiscal quarter as an integer.
        self.new_columns['quarter'] = timestamps.dt.quarter.astype('int32').values
        # Create a binary flag for the last day of the month.
        self.new_columns['is_month_end'] = timestamps.dt.is_month_end.astype('int32').values
        # Convert day of week to an index for cyclical encoding.
        day_of_week_idx = timestamps.dt.dayofweek.values
        # Calculate sine component for the weekly cycle.
        self.new_columns['dow_sin'] = sin(2 * pi * day_of_week_idx / 7).astype('float32')
        # Calculate cosine component for the weekly cycle.
        self.new_columns['dow_cos'] = cos(2 * pi * day_of_week_idx / 7).astype('float32')
        # Get the number of days in the specific month for normalization.
        dim = timestamps.dt.days_in_month.values
        # Calculate sine component for the monthly cycle.
        self.new_columns['dom_sin'] = sin(2 * pi * self.new_columns['day_of_month'] / dim).astype('float32')
        # Calculate cosine component for the monthly cycle.
        self.new_columns['dom_cos'] = cos(2 * pi * self.new_columns['day_of_month'] / dim).astype('float32')

    def add_overnight_features(self):
        # Display informational message to stdout.
        msg_info('Feature: Calculating overnight gap features.')
        # Retrieve yesterday's closing prices as a numpy array.
        p_close = self.prev_close.values
        # Retrieve today's opening prices as a numpy array.
        curr_open = self.data['o'].values
        # Compute the percentage difference between yesterday's close and today's open.
        self.new_columns['overnight_gap'] = divide(
            curr_open - p_close,
            p_close,
            out=ones_like(p_close) * 0,
            where=p_close!=0
        ).astype('float32')
        # Create a 1-day lag of the overnight gap to see how the stock gapped yesterday.
        self.new_columns['lagged_overnight_gap_1'] = DataFrame(self.new_columns['overnight_gap']).groupby(self.data['T'])[0].shift(1).values.flatten()
        # Generate a binary flag if the price gap is greater than 0.5%.
        self.new_columns['is_significant_gap'] = (abs(self.new_columns['overnight_gap']) > 0.005).astype('int32')

    def add_lagged_features(self):
        # Display informational message to stdout.
        msg_info('Feature: Calculating relative lagged columns.')
        # Define lists of price-based and quantity-based columns.
        price_cols, other_cols = ['c', 'h', 'l', 'o', 'vw'], ['n', 'v']
        # Iterate through each day in the LAGGED_MAX_DAYS setting.
        for i in range(1, LAGGED_MAX_DAYS + 1):
            # Shift the grouped data by the current lag amount.
            lag_base = self.grouped[price_cols + other_cols].shift(i)
            # Process each price column to calculate percentage change from history.
            for col in price_cols:
                # Get the raw values for the lagged data.
                l_val = lag_base[col].values
                # Use yesterday's value (shift 1) as the baseline to avoid current-row leakage.
                p_val = self.prev_close.values if col == 'c' else self.grouped[col].shift(1).values
                # Divide the difference by the lagged value to get the return.
                self.new_columns[f"lag_{col}_{i}"] = divide((p_val - l_val), l_val, out=ones_like(l_val)*0, where=l_val!=0).astype('float32')
            # Process each non-price column (volume, trade count) for growth rates.
            for col in other_cols:
                # Get the historical values for the lag.
                l_val = lag_base[col].values
                # Get yesterday's values as the baseline.
                p_val = self.grouped[col].shift(1).values
                # Calculate the percentage growth in volume or count.
                self.new_columns[f"lag_{col}_{i}"] = divide((p_val - l_val), l_val, out=ones_like(l_val)*0, where=l_val!=0).astype('float32')

    def add_technical_indicators(self):
        # Display informational message to stdout.
        msg_info('Feature: Calculating technical indicators.')
        # Store yesterday's close in a local variable for calculation.
        p_close = self.prev_close.values
        # Loop through each window size defined in the settings.
        for window in MOVING_AVERAGE_WINDOWS:
            # Calculate the rolling mean of the closing price and shift it to yesterday.
            sma_val = self.grouped['c'].transform(lambda x: x.rolling(window).mean()).shift(1).values
            # Compute the distance of yesterday's close from the SMA.
            self.new_columns[f"dist_sma_{window}"] = divide((p_close - sma_val), sma_val, out=ones_like(sma_val)*0, where=sma_val!=0).astype('float32')
            # Calculate the exponential moving average and shift it to yesterday.
            ema_val = self.grouped['c'].transform(lambda x: x.ewm(span=window, adjust=False).mean()).shift(1).values
            # Compute the distance of yesterday's close from the EMA.
            self.new_columns[f"dist_ema_{window}"] = divide((p_close - ema_val), ema_val, out=ones_like(ema_val)*0, where=ema_val!=0).astype('float32')
            # Shift the price by the window size plus one to find the historical anchor.
            p_win_c = self.grouped['c'].shift(window + 1).values
            # Calculate the Rate of Change relative to that historical anchor.
            self.new_columns[f"roc_{window}"] = divide((p_close - p_win_c), p_win_c, out=ones_like(p_win_c)*0, where=p_win_c!=0).astype('float32')
            # Calculate the rolling standard deviation of percentage changes for volatility.
            self.new_columns[f"vol_{window}"] = self.grouped['c'].transform(lambda x: x.pct_change().rolling(window).std()).shift(1).values.flatten()

    def add_candlestick_features(self):
        # Display informational message to stdout.
        msg_info('Feature: Calculating candlestick features.')
        # Assign shifted OHLC values to local variables representing yesterday's candle.
        p_c, p_o, p_h, p_l = self.prev_close.values, self.prev_open.values, self.prev_high.values, self.prev_low.values
        # Determine the absolute height of the candle body (open to close).
        self.new_columns['yest_body'] = abs(p_c - p_o).astype('float32')
        # Calculate the size of the upper shadow (wick).
        self.new_columns['yest_upper_shadow'] = (p_h - where(p_c > p_o, p_c, p_o)).astype('float32')
        # Calculate the size of the lower shadow (wick).
        self.new_columns['yest_lower_shadow'] = (where(p_c < p_o, p_c, p_o) - p_l).astype('float32')
        # Calculate the total vertical range of the candle.
        t_range = (p_h - p_l)
        # Normalize the body size by dividing it by the total candle range.
        self.new_columns['yest_rel_body'] = divide(self.new_columns['yest_body'], t_range, out=ones_like(t_range)*0, where=t_range!=0).astype('float32')

    def add_interaction_features(self):
        # Display informational message to stdout.
        msg_info('Feature: Calculating relative interaction features.')
        # Load yesterday's OHLCV values into local variables.
        p_c, p_v, p_o, p_h, p_l = self.prev_close.values, self.prev_volume.values, self.prev_open.values, self.prev_high.values, self.prev_low.values
        # Compute the dollar volume traded yesterday.
        dollar_vol_prev = p_c * p_v
        # Iterate through the window sizes to define relative intensity.
        for window in MOVING_AVERAGE_WINDOWS:
            # Calculate the rolling average dollar volume and shift it to yesterday.
            avg_dollar_vol = self.grouped.apply(lambda x: (x['c'] * x['v']).rolling(window).mean(), include_groups=False).reset_index(level=0, drop=True).shift(1).fillna(0).values.flatten()
            # Calculate how yesterday's volume compared to the historical average.
            rel_vol = divide(dollar_vol_prev, avg_dollar_vol, out=ones_like(avg_dollar_vol)*1.0, where=avg_dollar_vol!=0).astype('float32')
            # Store the relative dollar volume feature.
            self.new_columns[f'rel_dollar_vol_{window}'] = rel_vol
            # Calculate yesterday's simple return.
            p_ret = divide((p_c - p_o), p_o, out=ones_like(p_o)*0, where=p_o!=0)
            # Combine price move with volume intensity to create an "intensity" feature.
            self.new_columns[f'ret_vol_intensity_{window}'] = (p_ret * rel_vol).astype('float32')

############
### MAIN ###
############
def main(data):
    # Sort the dataframe by ticker and time to ensure sequential integrity.
    data = data.sort_values(by=['T', 't']).reset_index(drop = True)
    # Instantiate and run the FeatureEngineering class to generate historical predictors.
    data = FeatureEngineering(data=data)()
    # Append the target variable based on the current row's outcome.
    data = open_to_close(data=data)
    # Return the final dataframe ready for machine learning training.
    return data