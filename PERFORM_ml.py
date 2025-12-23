#!/usr/bin/env python
from argparse import ArgumentParser
from configparser import ConfigParser
from pathlib import Path

######################
### CUSTOM MODULES ###
######################
from modules.border import main as border
from modules.feature_selection import main as feature_selection
from modules.import_and_parse import main as import_and_parse
from modules.is_file import main as is_file
from modules.machine_learning import main as machine_learning
from modules.messages import msg_info
from modules.scaling import UseMinMaxScaler, UseRobustScaler, UseStandardScaler
from modules.train_test_split import main as train_test_split

################
### SETTINGS ###
################
# Define the root directory, which is the parent directory of this script.
ROOT = Path(__file__).parent.resolve()

# Path to the configuration INI file.
CONFIG_INI = Path(ROOT, 'configuration.ini')

#################
### FUNCTIONS ###
#################
def args():
    """Parse and return command-line arguments."""
    # Create an ArgumentParser object.
    parser = ArgumentParser(description='Perform machine learning on financial data.')
    # Add arguments.
    parser.add_argument('filename', type=Path, help='Path to the CSV file containing all data for training and testing.')
    # Parse the arguments.
    args = parser.parse_args()
    # Return the filename and symbols.
    return args.filename.absolute()

def convert_to_bool(string):
    """Convert a string representation of a boolean to an actual boolean value."""
    return bool(int(string))

def convert_to_list(string, delimiter):
    """Split a string representation of a list into an actual list based on the provided $delimiter."""
    return [item.strip() for item in string.split(delimiter)]

def calculate_baseline_accuracy(counts):
    """Calculate the baseline accuracy, which is the number of entries within the largest class divided by the total entries across all classes."""
    baseline_accuracy = counts.max() / counts.sum()
    # Display the baseline accuracy to stdout.
    msg_info(f"Num. of Class 0: {counts[0.0]} | Num. of Class 1: {counts[1.0]} | Baseline: {baseline_accuracy:.2%}")

def data_feature_selection(columns_x, X_train, y_train, X_test, perform_feature_selection):
    # Create a border to denote a process.
    border('FEATURE SELECTION', border_char='><')
    if perform_feature_selection:
        # Perform feature selection.
        X_train, X_test, selected_features = feature_selection(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            feature_names=columns_x
        )
        # Diplay message to stdout regarding selected features.
        msg_info(f"Selected features: {selected_features.to_list()}")
    else:
        # Display message to stdout that feature selection will not be performed.
        msg_info('Feature selection was set to False and will not be performed. Edit this script to change this.')
    # Return the training and testing data after feature selection.
    return X_train, X_test

############
### MAIN ###
############
def main(filename):
    """Main function to perform machine learning on financial data."""
    #---------------------#
    #--- Configuration ---#
    #---------------------#
    # Verify the configuration file exists.
    is_file(CONFIG_INI, exit_on_error=True)
    # Initiate the configuration parser.
    configuration_ini = ConfigParser()
    # Read the configuration INI file.
    configuration_ini.read(CONFIG_INI)
    # Define a dictionary to hold all configuration parameters.
    configuration = {}
    # Iterate through each section in the configuration INI file and add it to the configuration dictionary.
    for section in configuration_ini.sections(): configuration[section] = dict(configuration_ini.items(section))
    #------------#
    #--- Data ---#
    #------------#
    # Define the symbols to process.
    symbols = convert_to_list(string=configuration['DATA']['symbols'], delimiter=',')
    # Import and preprocess data for ML.
    data, columns_x = import_and_parse(
        filename=filename,
        symbols=symbols,
        cache_directory=Path(ROOT, configuration['GENERAL']['cache_directory']).absolute(),
        columns_one_hot_encoding=convert_to_list(string=configuration['DATA']['columns_one_hot_encoding'], delimiter=','),
        columns_x=convert_to_list(string=configuration['DATA']['columns_x'], delimiter=',')
    )
    # Define the label column(s) (y).
    columns_y = convert_to_list(string=configuration['DATA']['columns_y'], delimiter=',')
    # Calculate the baseline accuracy and display it to stdout.
    calculate_baseline_accuracy(data[columns_y].value_counts())
    #------------------------#
    #--- Train/Test Split ---#
    #------------------------#
    # Split the data into training and testing datasets.
    X_train, X_test, y_train, y_test = train_test_split(
        data=data,
        columns_x=columns_x,
        columns_y=columns_y,
        holdout_days=int(configuration['DATA']['holdout_days']),
        normalize_X=True
    )
    #-------------------------#
    #--- Feature Selection ---#
    #-------------------------#
    # Perform feature selection.
    X_train, X_test = data_feature_selection(
        columns_x=columns_x,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        perform_feature_selection=convert_to_bool(configuration['GENERAL']['perform_feature_selection'])
    )
    #------------------------#
    #--- Machine Learning ---#
    #------------------------#
    # Define the random seed(s).
    random_seeds = [int(item) for item in convert_to_list(string=configuration['GENERAL']['random_seed'], delimiter=',')]
    # Iterate through each random seed.
    for seed in random_seeds:
        msg_info(f'Using random seed: {seed}')
        # Perform machine learning.
        machine_learning(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            symbols=symbols,
            perform_hyperparameter_optimization=convert_to_bool(configuration['GENERAL']['perform_hyperparameter_optimization']),
            perform_cross_validation=convert_to_bool(configuration['GENERAL']['perform_cross_validation'])
        )

#############
### START ###
#############
if __name__ == '__main__':
    # Obtain user-defined variables.
    filename = args()
    # Start the script.
    main(filename=filename)