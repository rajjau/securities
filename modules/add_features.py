#!/usr/bin/env python
from pandas import Timedelta,to_datetime

######################
### CUSTOM MODULES ###
######################
from modules.messages import msg_info

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
    column = 'o_c'
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
    data['t_w'] = timestamps.dt.day_name()
    # Add the previous day of the week name as well.
    data['t_w_p'] = (timestamps - Timedelta(days = 1)).dt.day_name()
    # Return the $data.
    return(data)

def previous_day(data):
    # Display informational message to stdout.
    msg_info('Feature: Adding values from the previous day.')
    # Sort the DataFrame by the symbol name and timestamp. This groups stocks and ensures the data for a given stock is in order from earliest to most recent.
    data = data.sort_values(by = ['T', 't'])
    # Add the previous day's closing value.
    data['c_p'] = data.groupby('T')['c'].shift(1)
    # Add the previous day's highest price value.
    data['h_p'] = data.groupby('T')['h'].shift(1)
    # Add the previous day's lowest price value.
    data['l_p'] = data.groupby('T')['l'].shift(1)
    # Add the previous day's number of transactions.
    data['n_p'] = data.groupby('T')['n'].shift(1)
    # Add the previous day's opening value.
    data['o_p'] = data.groupby('T')['o'].shift(1)
    # Add the previous day's volume.
    data['v_p'] = data.groupby('T')['v'].shift(1)
    # Add the previous day's volume weighted averaged price.
    data['vw_p'] = data.groupby('T')['vw'].shift(1)
    # Return the $data.
    return(data)

############
### MAIN ###
############
def main(data):
    # Add string labels that describe when the opening value is less, equal, or greater than the closing value. These will be converted to int using one-hot-encoding before machine learning.
    data = open_to_close(data)
    # Calculate the different aspects of time for each row based on the timestamp ('t') value.
    data = time(data)
    # Add the values from the previous day.
    data = previous_day(data)
    # Return the modified $data.
    return(data)