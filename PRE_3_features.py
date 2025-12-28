#!/usr/bin/env python
from argparse import ArgumentParser
from pandas import read_csv
from pathlib import Path

######################
### CUSTOM MODULES ###
######################
from modules.add_features import main as add_features
from modules.is_file import main as is_file
from modules.messages import msg_info

################
### SETTINGS ###
################
SORT_BY_COLUMNS = ['T', 't']

#################
### FUNCTIONS ###
#################
def args():
    """Parse and return command-line arguments."""
    # Create an ArgumentParser object.
    parser = ArgumentParser(description='Add features to financial data.')
    # Add arguments.
    parser.add_argument('filename', type=Path, help='CSV containing the combined stock data to add features to.')
    parser.add_argument('filename_output', type=Path, help='Parquet output filename.')
    # Parse the arguments.
    args = parser.parse_args()
    # Return the filename and symbols.
    return args.filename.absolute(), args.filename_output.absolute()

############
### MAIN ###
############
def main(filename, filename_output):
    # Ensure the specified file exists.
    is_file(filename=filename, exit_on_error=True)
    # Read the data.
    data = read_csv(filepath_or_buffer=filename)
    # Remove all duplicate rows.
    data = data.drop_duplicates(keep = 'first')
    # Remove all rows that contain any NaNs.
    data = data.dropna(axis = 0)
    # Sort the DataFrame by the symbol name and timestamp. This groups stocks and ensures the data for a given stock is in order from earliest to most recent.
    data = data.sort_values(by = SORT_BY_COLUMNS)
    # Add various extra features to the $data.
    data = add_features(data)
    # Message to stdout.
    msg_info('Sorting all columns in alphabetical order.')
    # Sort the columns in alphabetical order.
    data = data.sort_index(axis = 1)
    # Message to stdout.
    msg_info('Removing all rows that contain any NaNs after adding features.')
    # Again, remove all rows that contain any NaNs after adding features.
    data = data.dropna(axis = 0)
    # Ensure the data is still sorted correctly after dropping rows.
    data = data.sort_values(by = SORT_BY_COLUMNS).reset_index(drop = True)
    # Message to stdout.
    msg_info('Saving data with new features to the output file.')
    # Save the $data with new features to the output file.
    data.to_parquet(path = filename_output, index = False)

#############
### START ###
#############
if __name__ == '__main__':
    # Obtain user-defined variables.
    filename, filename_output = args()
    # Start the script.
    main(filename=filename, filename_output=filename_output)