#!/usr/bin/env python
from pandas import read_parquet
from pathlib import Path

######################
### CUSTOM MODULES ###
######################
from modules.border import main as border
from modules.is_dir import main as is_dir
from modules.messages import msg_info, msg_error

#################
### FUNCTIONS ###
#################
def process_data(filename, symbols, filename_cache):
    # Read the data from $filename and import it as a DataFrame.
    data = read_parquet(path = filename)
    # Check if the user has defined certain symbols to train and test on.
    if symbols:
        # Message to stdout.
        msg_info(f"Training and testing will only be performed for: {symbols}")
        # Keep only rows that have the specified symbol(s), if specified.
        data = data[data['T'].isin(symbols)]
    # Check if the data is empty after filtering by symbol(s). If so, then exit with an error message.
    if data.empty: msg_error('The data is empty after filtering by symbol(s). Please check the input file and the specified symbols in the configuration file.')
    # Save the imported and modified data to a parquet file for faster loading in the future.
    data.to_parquet(path = filename_cache, index = False)
    # Return the $data variable.
    return data

############
### MAIN ###
############
def main(filename, symbols, cache_directory, columns_x, columns_y):
    # Create a border to denote a process.
    border('IMPORT DATA', border_char='><')
    # Verify that the cache directory exists.
    is_dir(cache_directory, exit_on_error=False)
    # Create the cache directory (if needed).
    cache_directory.mkdir(parents=True, exist_ok=True)
    # Define a parquet filename based on the provided $filename and $symbols.
    filename_cache = Path(cache_directory, f"{filename.stem}_{'_'.join(symbols) if symbols else 'all'}.parquet")
    # Check if the parquet file exists.
    if filename_cache.is_file():
        # Display a message to stdout that the data will be read from the parquet file.
        msg_info(f"Reading data from the cache file: {filename_cache}")
        # Read the data from the cache file.
        data = read_parquet(path = filename_cache)
    else:
        # Import the data from the parquet file and process it. This includes filtering by $symbols and saving to cache.
        data = process_data(filename=filename, symbols=symbols, filename_cache=filename_cache)
    # If a wildcard was passed to define the feature set columns, then use all columns except for $columns_y.
    if columns_x[0] == '*': columns_x = data.columns.difference(columns_y).to_list()
    # Return the processed $data and the updated $columns_x.
    return data, columns_x