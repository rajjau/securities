#!/usr/bin/env python
from datetime import datetime, timedelta
from multiprocessing import Pool
from os import cpu_count
from pathlib import Path
from sys import argv

###################
### DIRECTORIES ###
###################
# Directory this script is located in
root = Path(argv[0]).parent.absolute()

######################
### CUSTOM MODULES ###
######################
from modules.create_directory import main as create_directory
from modules.messages import msg_warn
from modules.massive_com import main as massive_com

#################
### FUNCTIONS ###
#################
def args():
    try:
        # Argument 1: Directory to save data to.
        data_dir = Path(argv[1]).absolute()
    except IndexError:
        raise IndexError('Argument 1: Directory to save data to.')
    try:
        # Argument 2: Date to obtain data from.
        date_start = argv[2]
    except IndexError:
        raise IndexError('Argument 2: Date in YYYY-MM-DD format to obtain data from.')
    try:
        # Argument 3 [OPTIONAL]: Date to obtain data to.
        date_end = argv[3]
    except IndexError:
        # If not set, then set the ending date to bool False.
        date_end = False
        # Display a warning message to stdout.
        msg_warn(f"Argument 3 [OPTIONAL]: Date in YYYY-MM-DD format to obtain data to. If not set, then only the starting date of {date_start} will be used.")
    # Combine the starting and ending dates into a tuple.
    dates = (date_start, date_end)
    # Return user-defined variables.
    return(data_dir, dates)

class date:
    def __init__(self, dates):
        # Convert all strings into a datetime object. This will allow the script to calculate all intermediate dates between the specified starting and ending dates, and also filter weekends as well.
        dates = [self.define(date) for date in dates]
        # Keep only entries that have been defined.
        dates = tuple(entry for entry in dates if entry)
        # Check the length of $dates.
        if len(dates) == 1:
            # If it contains only one date, then return it as-is.
            self.dates = dates
        else:
            # If both a starting and ending date have been specified, then define all dates in between.
            self.dates = self.expand(dates)

    def __call__(self):
        # Return the class global variable.
        return(self.dates)

    def define(self, date):
        try:
            # Convert the $date string into a datetime object in YYYY-MM-DD format.
            date = datetime.strptime(date, '%Y-%m-%d')
        except TypeError:
            # The $date is a different Type (e.g., a boolean).
            msg_warn(f"No ending date found.")
            # Return bool False.
            return(False)
        except ValueError:
            # If the format of the $date is false, then display an error message to stdout.
            raise ValueError(f"Invalid date, please specify in YYYY-MM-DD format: '{date}'")
        # Return the $date as a datetime object.
        return(date)
    
    def expand(self, dates):
        # Define the list that will hold all days between the starting and ending days, inclusively.
        all_dates = []
        # Define the date that will be continuously increased below.
        current = dates[0]
        # Create a while-loop to iterate through all days between the starting and ending dates.
        while current <= dates[1]:
            # Only add the $current date (in YYYY-MM-DD format) to the output list if it's not a weekend day.
            if not current.strftime('%A') in ('Saturday', 'Sunday'): all_dates.append(current.strftime('%Y-%m-%d'))
            # Increase the day by one.
            current = current + timedelta(days = 1)
        # Convert the list of all intermediate dates into a tuple and return.
        return(tuple(all_dates))

############
### MAIN ###
############
def main(dir_data, dates):
    # Create the data directory (if needed). $dir_data can be bool False, but only if this script is imported as a module.
    if dir_data: create_directory(dir_data, fail_if_exists = False)
    # Parse the specified $dates and convert into datetime objects. If both a starting and ending date was specified by user, then exapnd the $dates tuple to include all intermediate dates, excluding weekends.
    dates = date(dates)()
    # Download the data for the specified stock $tickers from Massive.com in JSON format. Returns a list of all newly downloaded data JSON files.
    filenames_output = massive_com(dir_data = dir_data, dates = dates)
    # Return the list of JSON data files.
    return(filenames_output)

#############
### START ###
#############
if __name__ == '__main__':
    # Obtain user-defined arguments
    [dir_data, dates] = args()
    # Start the script
    Pool(processes = cpu_count() - 1).apply(main, args = (dir_data, dates))
