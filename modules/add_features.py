#!/usr/bin/env python
from numpy import divide
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
    msg_info('Features: Is closing price > opening price.')
    # Create a binary column: 1 if closing price is higher than opening price, else 0.
    data['close_greater_than_open'] = (data['c'] > data['o']).astype('int32')
    # Return the modified dataframe with the target label.
    return data

class FeatureEngineering:
    def __init__(self, data):
        # Assign the input dataframe to the class instance.
        self.data = data
        # Yesterday's close price.
        self.prev_close = self.data['c'].shift(1).astype('float32')
        # Yesterday's opening price.
        self.prev_open = self.data['o'].shift(1).astype('float32')
        # Yesterday's trading volume.
        self.prev_volume = self.data['v'].shift(1).astype('float32')
        # Yesterday's high price.
        self.prev_high = self.data['h'].shift(1).astype('float32')
        # Yesterday's low price.
        self.prev_low = self.data['l'].shift(1).astype('float32')
        # Initialize a dictionary to store newly generated feature columns.
        self.new_columns = {}
        # Time and date features.
        self.add_time_and_date_features()
        # Overnight gap features.
        self.add_overnight_features()
        # Lagged features.
        self.add_lagged_features()
        # Merge the new feature dictionary into a dataframe and join with original data.
        self.data = concat([self.data, DataFrame(self.new_columns, index = self.data.index)], axis = 1)
        # Create a copy of the dataframe to resolve fragmentation and consolidate memory.
        self.data = self.data.copy()

    def __call__(self):
        # Return the fully engineered dataframe when the class is called.
        return(self.data)
    
    def add_time_and_date_features(self):
        # Display informational message to stdout.
        msg_info("Features: Date and time.")
        # Convert the 't' column from milliseconds to pandas datetime objects.
        timestamps = to_datetime(self.data['t'], unit = 'ms')
        # Normalize timestamps to just the date (time set to 00:00:00).
        self.new_columns['t_d'] = timestamps.dt.normalize().values
        # Day of the week (0=Monday, 6=Sunday).
        self.new_columns['day_of_week'] =  timestamps.dt.dayofweek.astype('int32').values
        # Day of month.
        self.new_columns['day_of_month'] = timestamps.dt.day.astype('int32').values
        # Month of year.
        self.new_columns['month'] = timestamps.dt.month.astype('int32').values
        # Is first day of the month.
        self.new_columns['is_month_start'] = timestamps.dt.is_month_start.astype('int32').values
        # Is last day of the month.
        self.new_columns['is_month_end'] = timestamps.dt.is_month_end.astype('int32').values
        # Quarter of the year.
        self.new_columns['quarter'] = timestamps.dt.quarter.astype('int32').values
        
    def add_overnight_features(self):
        # Display informational message to stdout.
        msg_info('Features: Overnight gap.')
        # Retrieve yesterday's closing prices as a numpy array.
        p_close = self.prev_close.values
        # Retrieve today's opening prices as a numpy array.
        today_open = self.data['o'].values
        # Overnight Gap = (today's open - yesterday's close) / yesterday's close. This provides the percentage change.
        overnight_gap = divide((today_open - p_close), p_close).astype('float32')
        # Add the overnight gap to the new columns dictionary.
        self.new_columns['overnight_gap'] = overnight_gap

    def add_lagged_features(self):
        # Display informational message to stdout.
        msg_info('Features: Lagged.')
        # Keep all columns except for the ticker ('T'), time ('t'), and date ('f').
        existing_columns = [column for column in self.data.columns if column not in ['T', 't', 'f']]
        # Define lagged versions of all existing columns.
        lagged_features = self.data[existing_columns].shift(periods = range(1, LAGGED_MAX_DAYS + 1)).astype('float32')
        # Iterate through each lagged column.
        for column in lagged_features.columns:
            # Define the new column name for the lagged feature.
            name = f"{column.replace('_', '_lag_')}"
            # Add the lagged feature to the new columns dictionary.
            self.new_columns[name] = lagged_features[column].values

############
### MAIN ###
############
def main(data):
    # Sort the dataframe by time to ensure sequential integrity.
    data = data.sort_values(by = 't').reset_index(drop = True)
    # Instantiate and run the FeatureEngineering class to generate historical predictors.
    data = FeatureEngineering(data=data)()
    # Append the target variable based on the current row's outcome.
    data = open_to_close(data=data)
    # Return the final dataframe ready for machine learning training.
    return data