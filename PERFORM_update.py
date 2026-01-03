#!/usr/bin/env python
from argparse import ArgumentParser
from datetime import datetime, timedelta
from pandas import read_csv, to_datetime
from pathlib import Path
from shutil import copy

######################
### CUSTOM MODULES ###
######################
from PRE_1_download import main as download
from PRE_3_features import main as features
from modules.combine_json import main as combine_json
from modules.is_file import main as is_file
from modules.is_dir import main as is_dir
from modules.messages import msg_info,msg_warn

#################
### FUNCTIONS ###
#################
def args():
    # Create an argument parser.
    parser = ArgumentParser(description = 'Update the raw stock data and prepare it for machine learning.')
    # Add the arguments to the parser.
    parser.add_argument('directory_combined', type = Path, help = 'Directory containing the combined datasets to continue off from (e.g., "raw_combined").')
    parser.add_argument('directory_raw', type = Path, help = 'Directory containing all raw data from the source (e.g., "raw").')
    parser.add_argument('--output', type = Path, default = None, help = 'Filename to save the final prepared data to. This is the data that will be used for machine learning. Default: data_<yesterday_date>.parquet')
    # Parse the arguments.
    arguments = parser.parse_args()
    # Check if the output filename was provided, otherwise set it to a default value.
    if not arguments.output:
        # Obtain yesterday's date and format it.
        yesterday = datetime.today() - timedelta(days = 1)
        # Define the output filename using yesterday's date (in YYYY-MM-DD format).
        arguments.output = Path(f"data_{yesterday.strftime('%Y-%m-%d')}.parquet").absolute()
        # Display a warning message to user that the default filename will be used.
        msg_warn(f"(OPTIONAL) --output: Filename to save the final prepared data to. Default: {arguments.output}")
    # Return the user-defined variables.
    return arguments.directory_combined.absolute(), arguments.directory_raw.absolute(), arguments.output.absolute()

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
    return timestamp_last

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
        return False
    else:
        # Otherwise, return a tuple of the last date and today's date in YYYY-MM-DD format.
        return (last.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d'))

def update(dir_data, filename_last, dates, filename_raw_csv, filename_features_csv):
    # Download all new data to their respective JSON files, which is returned as a list.
    filenames_json = download(dir_data=dir_data, dates=dates)
    # Check if the list of JSON files has entries.
    if filenames_json:
        # Copy the existing file to the new output filename.
        copy(filename_last, filename_raw_csv)
        # Combine the data for all filenames.
        combine_json(filenames=filenames_json, filename_output=filename_raw_csv)
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
    is_dir(directory=dir_combined, exit_on_error=True)
    is_dir(directory=dir_raw, exit_on_error=True)
    # Define a list of all files within the directory containing combined CSV data files.
    filenames = [filename for filename in Path(dir_combined).iterdir() if filename.is_file()]
    # Keep only CSV files and order the list by filename. This will sort the files from earlist to latest as long as the dates are in YYYY-MM-DD format in the filenames.
    filenames = sorted([filename for filename in filenames if filename.suffix == '.csv'])
    # Keep only the last CSV since it will contain all previous entries.
    filename_last = filenames[-1]
    # Obtain the final timestamp from the last CSV.
    timestamp_last = get_final_timestamp(filename = filename_last)
    # Use the final timestamp to determine the age of the data.
    dates = days(last=timestamp_last)
    # Define the output file for the combined raw CSV contents. This places the output file in the $dir_csv directory with the filename containing the prefix (up to the last underscore) from the $filename_prev name. Today's date is added as well.
    filename_raw_csv = Path(dir_combined, f"{filename_last.stem.rsplit('_', 1)[0]}_{dates[-1]}.csv")
    # Check of the output file already exists.
    if is_file(filename=filename_raw_csv, exit_on_error=False) is False:
        # If not, then download and update the existing data. This will write the new data to the output file.
        update(
            dir_data=dir_raw,
            filename_last=filename_last,
            dates=dates,
            filename_raw_csv=filename_raw_csv,
            filename_features_csv=filename_output
        )

#############
### START ###
#############
if __name__ == '__main__':
    # Obtain user-defined variables.
    directory_combined, directory_raw, filename_output = args()
    # Start the main function.
    main(dir_combined=directory_combined, dir_raw=directory_raw, filename_output=filename_output)