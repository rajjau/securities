#!/usr/bin/env python
from datetime import datetime, timedelta
from pandas import concat,read_csv,to_datetime
from pathlib import Path
from shutil import copy
from sys import argv

######################
### CUSTOM MODULES ###
######################
from PRE_1_download import main as download
from PRE_3_features import main as features
from modules.combine_json import main as combine_json
from modules.is_dir import main as is_dir
from modules.messages import msg_info,msg_warn

#################
### FUNCTIONS ###
#################
def args():
    try:
        # Argument 1: Directory containing the combined datasets to continue off from.
        directory_combined = Path(argv[1]).absolute()
    except IndexError:
        raise IndexError('Argument 1: Directory that contains the combined CSV files to continue from.')
    try:
        # Argument 2: Directory containing all raw data from the source.
        directory_raw = Path(argv[2]).absolute()
    except IndexError:
        raise IndexError('Argument 2: Directory to save new data to. This should be the same directory the existing raw stock data is in.')
    try:
        # Argument 3: Output filename.
        filename_output = Path(argv[3]).absolute()
    except IndexError:
        # Obtain today's date and time and subtract one day from it.
        yesterday = datetime.today() - timedelta(1)
        # Define the output filename using yesterday's date (in YYYY-MM-DD format).
        filename_output = Path(f"data_{yesterday.strftime('%Y-%m-%d')}.csv").absolute()
        # Display a warning message to user that 
        msg_warn(f"(OPTIONAL) Argument 3: Filename to save the final prepared data to. This is the data that will be used for machine learning. Default: {filename_output}")  
    # Return user-defined variable(s).
    return(directory_combined, directory_raw, filename_output)

def get_final_timestamp(filename):
    # Read the data from the file.
    data = read_csv(filepath_or_buffer = filename, usecols = ['t'])
    # Remove all rows that contain any NaNs.
    data = data.dropna(axis = 0)
    # Sort by timestamp.
    data = data.sort_values(by = data.columns.to_list())
    # Get the last timestamp.
    timestamp_last = data.iloc[-1].item()
    # Return the last timestamp from the $filename.
    return(timestamp_last)

def days(last):
    # Convert the column containing Unix millisecond timestamps into Pandas datetime format.
    last = to_datetime(last, unit = 'ms')
    # Convert the last date into a `datetime` object.
    last = last.date()
    # Obtain today's date without any time. Here, subtract a single day from $today because the market for the current day won't close until very late. This essentially means the most current date is always yesterday.
    today = datetime.today().date() - timedelta(days = 1)
    # Check if the existing data is up to date.
    if today <= last:
        # If so, then display a message to stdout.
        msg_info(f"The data is up-to-date. The last entry was on: {last.strftime('%A, %B %d %Y')}.")
        # Return bool False.
        return(False)
    else:
        # Otherwise, return a tuple of the last date and today's date in YYYY-MM-DD format.
        return((last.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d')))

def update(dir_data, filename_last, dates, filename_raw_csv, filename_features_csv):
    # Download all new data to their respective JSON files, which is returned as a list.
    filenames_json = download(dir_data = dir_data, dates = dates)
    # Check if the list of JSON files has entries.
    if filenames_json:
        # Copy the existing file to the new output filename.
        copy(filename_last, filename_raw_csv)
        # Combine the data for all filenames.
        combine_json(filenames = filenames_json, filename_output = filename_raw_csv)
        # Display informational message to stdout.
        msg_info(f"Done. Combined raw data written to the output file: '{filename_raw_csv}'")
        # Add features.
        features(filename = filename_raw_csv, filename_output = filename_features_csv)
        # Display informational message to stdout.
        msg_info(f"Done. Final data to use for machine learning written to the output file: '{filename_features_csv}'")
    else:
        # If there was no new JSON files downloaded, display a warning message to stdout.
        msg_warn(f"There was no data available to download. This can happen due to API limits or the market has not closed yet for the day. Please try again later.")

############
### MAIN ###
############
def main(dir_combined, dir_raw, filename_output):
    # Ensure the specified directories are valid.
    is_dir(directory = dir_combined, exit_on_error = True)
    is_dir(directory = dir_raw, exit_on_error = True)
    # Define a list of all files within the directory containing combined CSV data files.
    filenames = [filename for filename in Path(dir_combined).iterdir() if filename.is_file()]
    # Keep only CSV files and order the list by filename. This will sort the files from earlist to latest as long as the dates are in YYYY-MM-DD format in the filenames.
    filenames = sorted([filename for filename in filenames if filename.suffix == '.csv'])
    # Keep only the last CSV since it will contain all previous entries.
    filename_last = filenames[-1]
    # Obtain the final timestamp from the last CSV.
    timestamp_last = get_final_timestamp(filename = filename_last)
    # Use the final timestamp to determine the age of the data.
    dates = days(last = timestamp_last)
    # Check if dates have been returned, meaning an update is needed.
    if dates:
        # Define the output file for the combined raw CSV contents. This places the output file in the $dir_csv directory with the filename containing the prefix (up to the last underscore) from the $filename_prev name. Today's date is added as well.
        filename_raw_csv = Path(dir_combined, f"{filename_last.stem.rsplit('_', 1)[0]}_{dates[-1]}.csv")
        # Download and update the existing data. This will write the new data to the output file.
        update(dir_data = dir_raw, filename_last = filename_last, dates = dates, filename_raw_csv = filename_raw_csv, filename_features_csv = filename_output)

#############
### START ###
#############
if __name__ == '__main__':
    # Obtain user-defined variables.
    [directory_combined, directory_raw, filename_output] = args()
    # Start the main function.
    main(dir_combined = directory_combined, dir_raw = directory_raw, filename_output = filename_output)