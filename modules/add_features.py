#!/usr/bin/env python
from pandas import Timedelta,to_datetime

######################
### CUSTOM MODULES ###
######################
from modules.messages import msg_info

################
### SETTINGS ###
################
LAGGED_MAX_DAYS = 5

#################
### FUNCTIONS ###
#################
def open_to_close(data):
    # Display informational message to stdout.
    msg_info('Feature: Adding opening to close value descriptions.')
    # All open values in $data.
    data_open_values = data['o']
    # All close values in $data.
    data_close_values = data['c']
    # Define the name of the new column that describes whether the opening value for the symbol is less than, equal to, or greater than the closing value.
    column = 'close_greater_than_open'
    # All instances where the closing value is less than or equal to the opening value.
    data.loc[data_close_values <= data_open_values, column] = 0
    # All instances where the closing value is greater than the opening value.
    data.loc[data_close_values > data_open_values, column] = 1
    # Return the $data with the new column.
    return(data)

def time(data):
    # Display informational message to stdout.
    msg_info('Feature: Adding time descriptions.')
    # Convert the column containing Unix millisecond timestamps into Pandas datetime format.
    timestamps = to_datetime(data['t'], unit = 'ms')
    # Add the full date (YYYY-MM-DD).
    data['t_d'] = timestamps.dt.date
    # Add the day of week.
    data['day_of_week'] = timestamps.dt.day_name()
    # Add the previous day of the week name as well.
    data['lagged_day_of_week'] = (timestamps - Timedelta(days = 1)).dt.day_name()
    # Return the $data.
    return(data)

class lagged:
    def __init__(self, data):
        # Display informational message to stdout.
        msg_info('Feature: Adding features from previous days.')
        # Sort the DataFrame by the symbol name and timestamp. This groups stocks and ensures the data for a given stock is in order from earliest to most recent.
        data = data.sort_values(by = ['T', 't'])
        # Add features from the previous days.
        for previous_day in range(1, LAGGED_MAX_DAYS + 1):
            # Add the previous day's closing value.
            data = self.obtain_data_from_previous_day(data = data, column_name = 'c', new_column_name = f'lagged_close_{previous_day}', previous_day = previous_day)
            # Add the previous day's highest price value.
            data = self.obtain_data_from_previous_day(data = data, column_name = 'h', new_column_name = f'lagged_high_{previous_day}', previous_day = previous_day)
            # Add the previous day's lowest price value.
            data = self.obtain_data_from_previous_day(data = data, column_name = 'l', new_column_name = f'lagged_low_{previous_day}', previous_day = previous_day)
            # Add the previous day's number of transactions.
            data = self.obtain_data_from_previous_day(data = data, column_name = 'n', new_column_name = f'lagged_number_transactions_{previous_day}', previous_day = previous_day)
            # Add the previous day's opening value.
            data = self.obtain_data_from_previous_day(data = data, column_name = 'o', new_column_name = f'lagged_open_{previous_day}', previous_day = previous_day)
            # Add the previous day's volume.
            data = self.obtain_data_from_previous_day(data = data, column_name = 'v', new_column_name = f'lagged_volume_{previous_day}', previous_day = previous_day)
            # Add the previous day's volume weighted averaged price.
            data = self.obtain_data_from_previous_day(data = data, column_name = 'vw', new_column_name = f'lagged_volume_weighted_avg_{previous_day}', previous_day = previous_day) 
        # Make $data an internal variable to the class.
        self._data = data

    @property
    def data(self):
        # Return the $data.
        return(self._data)
    
    def obtain_data_from_previous_day(self, data, column_name, new_column_name, previous_day = 1):
        # Obtain data from the specified previous day, ensuring to group by the symbol name. The previous day variable only pulls data from the day that occured X days ago, not everything up to it.
        data[new_column_name] = data.groupby('T')[column_name].shift(previous_day)
        # Return the $data.
        return(data)

class feature_engineering:
    def __init__(self, data):
        # Display informational message to stdout.
        msg_info('Feature: Creating new features based off existing ones.')
        # Sort the DataFrame by the symbol name and timestamp. This groups stocks and ensures the data for a given stock is in order from earliest to most recent.
        data = data.sort_values(by = ['T', 't'])
        # Add new features from the previous days.
        for previous_day in range(1, LAGGED_MAX_DAYS + 1):
            # Add the spread between the high and low prices.
            data[f"lagged_high_low_spread_{previous_day}"] = data[f"lagged_high_{previous_day}"] - data[f"lagged_low_{previous_day}"]
            # Add the return from the previous days.
            data[f"lagged_daily_return_{previous_day}"] = (data[f"lagged_close_{previous_day}"] - data[f"lagged_open_{previous_day}"]) / data[f"lagged_open_{previous_day}"]
        # Make $data an internal variable to the class.
        self._data = data

    @property
    def data(self):
        # Return the $data.
        return(self._data)

############
### MAIN ###
############
def main(data):
    # Add string labels that describe when the opening value is less, equal, or greater than the closing value. These will be converted to int using one-hot-encoding before machine learning.
    data = open_to_close(data = data)
    # Calculate the different aspects of time for each row based on the timestamp ('t') value.
    data = time(data = data)
    # Add features from previous days.
    data = lagged(data = data).data
    # Create new features from the existing data.
    data = feature_engineering(data = data).data
    # Return the modified $data.
    return(data)