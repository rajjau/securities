#!/usr/bin/env python
from configparser import ConfigParser
from pathlib import Path

######################
### CUSTOM MODULES ###
######################
from modules.convert_to_list import main as convert_to_list
from modules.feature_selection_wrapper import main as feature_selection_wrapper
from modules.import_and_parse import main as import_and_parse
from modules.is_file import main as is_file
from modules.messages import msg_info
from modules.train_test_split import main as train_test_split

################
### SETTINGS ###
################
# Define the root directory, which is the parent directory of this script.
ROOT = Path(__file__).parent.resolve()

# Path to the configuration INI file.
CONFIG_INI = Path(ROOT, 'configuration.ini')

# Define columns that represent the "future" or "result" of the current bar. Keep 'o' (Open) because it is known at the start of the trade.
LEAKY_COLUMNS = ['c', 'h', 'l', 'v', 'n', 'vw']

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
    #------------#
    #--- Data ---#
    #------------#
    # Define the label column(s) (y).
    columns_y = convert_to_list(string=configuration_ini['DATA']['COLUMNS_Y'], delimiter=',')
    # Import and preprocess data for ML.
    data, columns_x = import_and_parse(
        filename=filename,
        columns_x=convert_to_list(string=configuration_ini['DATA']['COLUMNS_X'], delimiter=','),
        columns_y=columns_y
    )
    #----------------#
    #--- Cleaning ---#
    #----------------#
    # Filter columns_x to remove leakage while keeping lagged features and 'o'.
    columns_x = [col for col in columns_x if col not in LEAKY_COLUMNS]
    # Message to stdout to confirm cleaning.
    msg_info(f"Removed leaky features: {LEAKY_COLUMNS}")
    #------------------------#
    #--- Train/Test Split ---#
    #------------------------#
    # Split the data into training and testing datasets.
    X_train, X_test, y_train, y_test, columns_x = train_test_split(
        data=data,
        columns_x=columns_x,
        columns_y=columns_y,
        columns_one_hot_encoding=convert_to_list(string=configuration_ini['DATA']['COLUMNS_ONE_HOT_ENCODING'], delimiter=','),
        holdout_days=configuration_ini.getint('DATA', 'HOLDOUT_DAYS'),
        normalize_X=True,
        normalize_method=configuration_ini['NORMALIZATION']['NORMALIZE_METHOD']
    )
    #----------------------#
    #--- Random Seed(s) ---#
    #----------------------#
    # Define the random seed(s).
    random_seeds = [int(item) for item in convert_to_list(string=configuration_ini['GENERAL']['RANDOM_SEED'], delimiter=',')]
    #-------------------------#
    #--- Feature Selection ---#
    #-------------------------#
    # Perform feature selection.
    X_train, X_test = feature_selection_wrapper(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        columns_x=columns_x,
        random_seeds=random_seeds,
        configuration_ini=configuration_ini
    )
    # Return the training and testing data.
    return X_train, X_test, y_train, y_test