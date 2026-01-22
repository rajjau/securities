#!/usr/bin/env python
from argparse import ArgumentParser
from configparser import ConfigParser
from pandas import concat, read_csv
from pathlib import Path

######################
### CUSTOM MODULES ###
######################
from modules.add_features import main as add_features
from modules.border import main as border
from modules.convert_to_list import main as convert_to_list
from modules.is_file import main as is_file
from modules.messages import msg_warn

################
### SETTINGS ###
################
# Define the root directory, which is the parent directory of this script.
ROOT = Path(__file__).parent.resolve()

# Path to the configuration INI file.
CONFIG_INI = Path(ROOT, 'configuration.ini')

# Define the chunk size for reading large CSV files.
CHUNKSIZE = 5000

# Columns to sort the DataFrame by.
SORT_BY_COLUMNS = ['t']

#################
### FUNCTIONS ###
#################
def args():
    """Parse and return command-line arguments."""
    # Create an ArgumentParser object.
    parser = ArgumentParser(description='Add features to financial data.')
    # Add arguments.
    parser.add_argument('filename', type=Path, help='CSV containing the combined stock data to add features to.')
    parser.add_argument('directory_output', type=Path, help='Directory that will contain the Parquet output files for every ticker defined in the configuration INI.')
    # Parse the arguments.
    args = parser.parse_args()
    # Return the filename and tickers.
    return args.filename.absolute(), args.directory_output.absolute()

############
### MAIN ###
############
def main(filename, directory_output):
    # Verify the file containing raw combined data exists.
    is_file(filename=filename, exit_on_error=True)
    # Create the output directory if it does not exist.
    directory_output.mkdir(exist_ok = True)
    #---------------------#
    #--- Configuration ---#
    #---------------------#
    # Verify the configuration file exists.
    is_file(CONFIG_INI, exit_on_error=True)
    # Initiate the configuration parser.
    configuration_ini = ConfigParser()
    # Read the configuration INI file.
    configuration_ini.read(CONFIG_INI)
    #------------#
    #--- Data ---#
    #------------#
    # Define the tickers to process.
    tickers = convert_to_list(string=configuration_ini['DATA']['TICKERS'], delimiter=',')
    # Iterate through each ticker.
    for ticker in tickers:
        # Message to stdout.
        border(ticker, border_char = '-')
        # Define the output filename for the current ticker.
        filename_output = Path(directory_output, f"{ticker}.parquet")
        # If the output file already exists, skip processing for this ticker.
        if filename_output.is_file():
            # Message to stdout.
            msg_warn(f"SKIP: Output file already exists: {filename_output}")
            # Continue to the next ticker.
            continue
        # Define a list to hold all data for the current ticker.
        data_ticker = []
        # Read the entire dataset in chunks to manage memory usage.
        chunks = read_csv(filepath_or_buffer = filename, chunksize = CHUNKSIZE)
        # Iterate through each chunk.
        for chunk in chunks:
            # Filter the chunk for the current ticker.
            matches = chunk.loc[chunk['T'] == ticker, :]
            # If matches were found, append them to the list.
            if not matches.empty: data_ticker.append(matches)
        # Concatenate all chunks for the current ticker into a single DataFrame.
        data_ticker = concat(data_ticker, axis = 0).reset_index(drop = True)
        # Remove all duplicate rows.
        data_ticker = data_ticker.drop_duplicates(keep = 'first')
        # Remove all rows that contain any NaNs.
        data_ticker = data_ticker.dropna(axis = 0).reset_index(drop = True)
        # Add various extra features to the $data.
        data_ticker = add_features(data_ticker)
        # Ensure the data is still sorted correctly after dropping rows.
        data_ticker = data_ticker.sort_values(by = SORT_BY_COLUMNS).reset_index(drop = True)
        # Save the $data with new features to the output file.
        data_ticker.to_parquet(path = filename_output, index = False)

#############
### START ###
#############
if __name__ == '__main__':
    # Obtain user-defined variables.
    filename, directory_output = args()
    # Start the script.
    main(filename=filename, directory_output=directory_output)