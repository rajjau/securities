#!/usr/bin/env python
from argparse import ArgumentParser
from multiprocessing import Pool
from os import cpu_count
from pandas import read_csv, read_pickle
from pathlib import Path
from sys import argv

######################
### CUSTOM MODULES ###
######################
from modules.border import main as border
from modules.feature_selection import main as feature_selection
from modules.machine_learning import main as machine_learning
from modules.messages import msg_info
from modules.one_hot_encoding import main as one_hot_encoding
from modules.train_test_split import main as train_test_split

################
### SETTINGS ###
################
# Directory containing cache files such as pickle files containing imported and already processed data.
CACHE_DIRECTORY = Path(Path(argv[0]).parent.absolute(), 'cache')
#
# Define the column(s) to perform one-hot encoding to.
COLUMNS_ONE_HOT_ENCODING = ['day_of_week', 'lagged_day_of_week']
#
# Define the name of all feature columns (X). This does not include all of the 'lagged_' feature columns, which will be added once the data is imported.
COLUMNS_X = ['day_of_week', 'o', 'v']
#
# Define the name of the label (y).
COLUMNS_Y = ['close_greater_than_open']
#
# Define the total number of days to holdout for the test set.
HOLDOUT_DAYS = 60
#
# Choose whether cross-validation will be performed.
PERFORM_CROSS_VALIDATION = True
#
# Choose whether feature selection will be performed.
PERFORM_FEATURE_SELECTION = False
#
# Choose whether GridSearchCV will be performed for hyperparameter optimization.
PERFORM_HYPERPARAMETER_OPTIMIZATION = True

#################
### FUNCTIONS ###
#################
def args():
    # Create an ArgumentParser object.
    parser = ArgumentParser(description = 'Perform machine learning on financial data.')
    # Add arguments.
    parser.add_argument('filename', type = Path, help = 'Path to the CSV file containing all data for training and testing.')
    parser.add_argument('--symbols', type = str, help = 'Symbol(s) to train for. If none are specified, then all will be used.', default = False)
    # Parse the arguments.
    args = parser.parse_args()
    # If the user specified symbols, convert them to a list.
    if args.symbols: args.symbols = [entry.upper().strip() for entry in args.symbols.split(',')]
    # Return the filename and symbols.
    return(args.filename.absolute(), args.symbols)

def import_data(filename, symbols):
    # Create a border to denote a process.
    border('IMPORT DATA', border_char = '><')
    # Define a pickle filename based on the provided $filename.
    filename_pickle = Path(CACHE_DIRECTORY, f"{filename.stem}_{'_'.join(symbols) if symbols else 'all'}.pkl")
     # Check if the pickle file exists.
    if filename_pickle.is_file():
        # Display message to stdout that the data will be read from the pickle file.
        msg_info(f"Reading data from previously saved pickle file: {filename_pickle}")
        # Read the data from the pickle file.
        data = read_pickle(filename_pickle)
    else:
        # Read the data from $filename and import it as a DataFrame.
        data = read_csv(filename)
        # Check if the user has defined certain symbols to train and test on.
        if symbols:
            # Message to stdout.
            msg_info(f"Training and testing will only be performed for: {symbols}")
            # Keep only rows that have the specified symbol(s), if specified.
            data = data[data['T'].isin(symbols)]
        # Find all entries that contain NaNs.
        nans = data.isna().any(axis = 1)
        # Keep only entries that do not contain NaNs.
        data = data.loc[~nans, :].reset_index(drop = True)
         # Save the imported and modified data to a pickle file for faster loading in the future.
        data.to_pickle(filename_pickle)
    # Check if the global variable has been defined for a list of columns to perform one-hot encoding (OHE) to.
    if COLUMNS_ONE_HOT_ENCODING:
        # Iterate through each column name in the list.
        for name in COLUMNS_ONE_HOT_ENCODING:
            # Message to stdout.
            msg_info(f"One-hot encoding column: '{name}'")
            # Perform OHE for the current $column, remove the original column, and add the new OHE columns to $data.
            [data, names_ohe] = one_hot_encoding(data = data, name = name, drop_and_replace = True)
            # Define the list of feature columns.
            columns_X = sorted([entry for entry in COLUMNS_X if entry != name] + names_ohe)
            # Modify the global variables that define the feature (X) and label (y) columns based on the new OHE column names.
            modify_columns_X_y(new_value = columns_X, is_X = True)
    # Return $data.
    return(data)

def modify_columns_X_y(new_value, is_X = True):
    # Check if the variable was set that determines whether the feature (X) or label (y) column will be modified.
    if is_X is True:
        # Set the global keyword.
        global COLUMNS_X
        # Set the new value for the features (X) columns.
        COLUMNS_X = new_value
    else:
         # Set the global keyword.
        global COLUMNS_Y
        # Set the new value for the label (y) column.
        COLUMNS_Y = new_value

def calculate_baseline_accuracy(counts):
    # Calculate the baseline accuracy, which is the number of entries within the largest class divided by the total entries across all classes.
    baseline_accuracy = counts.max() / counts.sum()
    # Display the baseline accuracy to stdout.
    msg_info(f"Num. of Class 0: {counts[0.0]} | Num. of Class 1: {counts[1.0]} | Baseline: {baseline_accuracy:.2%}")

############
### MAIN ###
############
def main(filename, symbols):
    #------------#
    #--- Data ---#
    #------------#
    # Read and prepare data for ML.
    data = import_data(filename = filename, symbols = symbols)
    # Modify the global variables that define the feature (X) column based on the new 'lagged_' column names.
    modify_columns_X_y(new_value = COLUMNS_X + [entry for entry in data.columns if entry.startswith('lagged_')], is_X = True)
    # Calculate the baseline accuracy and display it to stdout.
    calculate_baseline_accuracy(data[COLUMNS_Y].value_counts())
    # Split the $data into training and testing sets, where the test set is the final X days from the $data. Additionally, the columns are normalized.
    [X_train, X_test, y_train, y_test] = train_test_split(data = data, columns_x = COLUMNS_X, columns_y = COLUMNS_Y, holdout_days = HOLDOUT_DAYS, normalize_X = True)
    #-------------------------#
    #--- Feature Selection ---#
    #-------------------------#
    # Create a border to denote a process.
    border('FEATURE SELECTION', border_char = '><')
    if PERFORM_FEATURE_SELECTION is True:
        # Perform feature selection.
        [X_train, X_test, selected_features] = feature_selection(X_train = X_train, X_test = X_test, y_train = y_train, feature_names = COLUMNS_X)
        # Diplay message to stdout regarding selected features.
        msg_info(f"Selected features: {selected_features.to_list()}")
    else:
        # Display message to stdout that feature selection will not be performed.
        msg_info('Feature selection was set to False and will not be performed. Edit this script to change this.')
    #--------------------------#
    #--- Training & Testing ---#
    #--------------------------#
    machine_learning(X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test, symbols = symbols, perform_hyperparameter_optimization = PERFORM_HYPERPARAMETER_OPTIMIZATION, perform_cross_validation = PERFORM_CROSS_VALIDATION)

#############
### START ###
#############
if __name__ == '__main__':
    # Create the cache directory (if needed).
    CACHE_DIRECTORY.mkdir(exist_ok = True)
    # Obtain user-defined variables.
    [filename, symbols] = args()
    # Start the script.
    with Pool(processes = cpu_count() - 1) as p: p.apply(main, args = (filename, symbols,))
    # main(filename = filename, symbols = symbols)