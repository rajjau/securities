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
    parser.add_argument('directory_output', type=Path, help='Directory that will contain the Parquet output files for every symbol defined in the configuration INI.')
    # Parse the arguments.
    args = parser.parse_args()
    # Return the filename and symbols.
    return args.filename.absolute(), args.directory_output.absolute()

def convert_to_list(string, delimiter):
    """Split a string representation of a list into an actual list based on the provided $delimiter."""
    return [item.strip() for item in string.split(delimiter)]

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
    # Define the symbols to process.
    symbols = convert_to_list(string=configuration_ini['DATA']['SYMBOLS'], delimiter=',')
    # Iterate through each symbol.
    for symbol in symbols:
        # Message to stdout.
        border(symbol, border_char = '-')
        # Define the output filename for the current symbol.
        filename_output = Path(directory_output, f"{symbol}.parquet")
        # If the output file already exists, skip processing for this symbol.
        if filename_output.is_file():
            # Message to stdout.
            msg_warn(f"SKIP: Output file already exists: {filename_output}")
            # Continue to the next symbol.
            continue
        # Define a list to hold all data for the current symbol.
        data_symbol = []
        # Read the entire dataset in chunks to manage memory usage.
        chunks = read_csv(filepath_or_buffer = filename, chunksize = CHUNKSIZE)
        # Iterate through each chunk.
        for chunk in chunks:
            # Filter the chunk for the current symbol.
            matches = chunk.loc[chunk['T'] == symbol, :]
            # If matches were found, append them to the list.
            if not matches.empty: data_symbol.append(matches)
        # Concatenate all chunks for the current symbol into a single DataFrame.
        data_symbol = concat(data_symbol, axis = 0).reset_index(drop = True)
        # Remove all duplicate rows.
        data_symbol = data_symbol.drop_duplicates(keep = 'first')
        # Remove all rows that contain any NaNs.
        data_symbol = data_symbol.dropna(axis = 0)
        # Sort the DataFrame by the symbol name and timestamp. This groups stocks and ensures the data for a given stock is in order from earliest to most recent.
        data_symbol = data_symbol.sort_values(by = SORT_BY_COLUMNS)
        # Add various extra features to the $data.
        data_symbol = add_features(data_symbol)
        # Again, remove all rows that contain any NaNs after adding features.
        data_symbol = data_symbol.dropna(axis = 0)
        # Ensure the data is still sorted correctly after dropping rows.
        data_symbol = data_symbol.sort_values(by = SORT_BY_COLUMNS).reset_index(drop = True)
        # Save the $data with new features to the output file.
        data_symbol.to_parquet(path = filename_output, index = False)

#############
### START ###
#############
if __name__ == '__main__':
    # Obtain user-defined variables.
    filename, directory_output = args()
    # Start the script.
    main(filename=filename, directory_output=directory_output)