#!/usr/bin/env python
from pandas import read_csv, read_pickle
from pathlib import Path

######################
### CUSTOM MODULES ###
######################
from modules.border import main as border
from modules.is_dir import main as is_dir
from modules.messages import msg_info, msg_error
from modules.one_hot_encoding import main as one_hot_encoding

#################
### FUNCTIONS ###
#################
def process_data(filename, symbols, filename_pickle):
    # Read the data from $filename and import it as a DataFrame.
    data = read_csv(filename)
    # Check if the user has defined certain symbols to train and test on.
    if symbols:
        # Message to stdout.
        msg_info(f"Training and testing will only be performed for: {symbols}")
        # Keep only rows that have the specified symbol(s), if specified.
        data = data[data['T'].isin(symbols)]
    # Check if the data is empty after filtering by symbol(s). If so, then exit with an error message.
    if data.empty: msg_error("The data is empty after filtering by symbol(s). Please check the input file and the specified symbols in the configuration file.")
    # Find all entries that contain NaNs.
    nans = data.isna().any(axis = 1)
    # Keep only entries that do not contain NaNs.
    data = data.loc[~nans, :].reset_index(drop = True)
    # Save the imported and modified data to a pickle file for faster loading in the future.
    data.to_pickle(filename_pickle)
    # Return the $data variable.
    return data
    
def one_hot_encode_data(data, columns_one_hot_encoding, columns_x):
    # Check if the global variable has been defined for a list of columns to perform one-hot encoding (OHE) to.
    if columns_one_hot_encoding:
        # Iterate through each column name in the list.
        for name in columns_one_hot_encoding:
            # Message to stdout.
            msg_info(f"One-hot encoding column: '{name}'")
            # Perform OHE for the current $column, remove the original column, and add the new OHE columns to $data.
            data, names_ohe = one_hot_encoding(data=data, name=name, drop_and_replace=True)
            # Define the list of feature columns.
            columns_x = sorted([entry for entry in columns_x if entry != name] + names_ohe)
    # Return $data and $columns_X. This works if one-hot encoding was not performed as well.
    return data, columns_x

############
### MAIN ###
############
def main(filename, symbols, cache_directory, columns_one_hot_encoding, columns_x):
    # Create a border to denote a process.
    border('IMPORT DATA', border_char='><')
    # Verify that the cache directory exists.
    is_dir(cache_directory, exit_on_error=False)
    # Create the cache directory (if needed).
    cache_directory.mkdir(parents=True, exist_ok=True)
    # Define a pickle filename based on the provided $filename.
    filename_pickle = Path(cache_directory, f"{filename.stem}_{'_'.join(symbols) if symbols else 'all'}.pkl")
    # Check if the pickle file exists.
    if filename_pickle.is_file():
        # Display message to stdout that the data will be read from the pickle file.
        msg_info(f"Reading data from previously saved pickle file: {filename_pickle}")
        # Read the data from the pickle file.
        data = read_pickle(filename_pickle)
    else:
        # Import the data from the CSV file and process it. This includes filtering by symbol(s), removing NaNs, and saving to a pickle file for caching.
        data = process_data(filename=filename, symbols=symbols, filename_pickle=filename_pickle)
    # Perform one-hot encoding (OHE) for specified columns.
    data, columns_x = one_hot_encode_data(data=data, columns_one_hot_encoding=columns_one_hot_encoding, columns_x=columns_x)
    # Modify the the feature (X) column to include the new 'lagged_' column names.
    columns_x = columns_x + [entry for entry in data.columns if entry.startswith('lagged_')]
    # Return the processed data and the feature (X) columns.
    return data, columns_x