#!/usr/bin/env python
from multiprocessing import Pool
from os import cpu_count
from pandas import read_csv
from pathlib import Path
from sys import argv

######################
### CUSTOM MODULES ###
######################
from modules.border import main as border
from modules.feature_selection import main as feature_selection
from modules.machine_learning import main as machine_learning
from modules.messages import msg_info,msg_warn
from modules.one_hot_encoding import main as one_hot_encoding
from modules.train_test_split import main as train_test_split

################
### SETTINGS ###
################
# Define the name of all feature columns (X). This does not include all of the 'lagged_' feature columns, which will be added once the data is imported.
COLUMNS_X = ['day_of_week', 'o', 'v']
#
# Define the name of the label (y).
COLUMNS_Y = ['close_greater_than_open']
#
# Define the total number of days to holdout for the test set.
HOLDOUT_DAYS = 60
#
# Define the column(s) to perform one-hot encoding to.
COLUMNS_ONE_HOT_ENCODING = ['day_of_week', 'lagged_day_of_week']

#################
### FUNCTIONS ###
#################
def args():
    try:
        # Argument 1: Path to the data file to use for training and testing.
        filename = Path(argv[1])
    except IndexError:
        raise IndexError('Argument 1: Path to the CSV file containing all data for training and testing.')
    try:
        # Argument 2: Symbol(s) to train for. This is an optional argument.
        symbols = argv[2:]
    except IndexError:
        # Display a warning to user that all symbols will be used.
        msg_warn('(OPTIONAL) Argument 2-K: Symbol(s) to train for, starting from argument 2 and to argument K, where K is some number. If none are specified, then all will be used.')
        # Set the variable to bool False.
        symbols = False
    # Return the user-defined variables.
    return(filename, symbols)

def import_data(filename, symbols):
    # Create a border to denote a process.
    border('IMPORT DATA', border_char = '><')
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
    # Split the $data into training and testing sets, where the test set is the final X days from the $data. Additionally, the columns are normalized.
    [X_train, X_test, y_train, y_test] = train_test_split(data = data, columns_x = COLUMNS_X, columns_y = COLUMNS_Y, holdout_days = HOLDOUT_DAYS, normalize_X = True)
    #-------------------------#
    #--- Feature Selection ---#
    #-------------------------#
    # Create a border to denote a process.
    border('FEATURE SELECTION', border_char = '><')
    # Perform feature selection.
    [X_train, X_test, selected_features] = feature_selection(X_train = X_train, X_test = X_test, y_train = y_train, feature_names = COLUMNS_X)
    # Diplay message to stdout regarding selected features.
    msg_info(f"Selected features: {selected_features.to_list()}")
    #--------------------------#
    #--- Training & Testing ---#
    #--------------------------#
    machine_learning(X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test, symbols = symbols, perform_hyperparameter_optimization = True, perform_cross_validation = True)

#############
### START ###
#############
if __name__ == '__main__':
    # Obtain user-defined variables.
    [filename, symbols] = args()
    # Start the script.
    Pool(processes = cpu_count() - 1).apply(main, args = (filename, symbols))
    # main(filename = filename, symbols = symbols)