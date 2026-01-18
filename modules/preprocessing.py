#!/usr/bin/env python
from configparser import ConfigParser, ExtendedInterpolation
from joblib import dump
from pathlib import Path

######################
### CUSTOM MODULES ###
######################
from modules.convert_to_list import main as convert_to_list
from modules.date_and_time import main as date_and_time
from modules.feature_selection_wrapper import main as feature_selection_wrapper
from modules.import_and_parse import main as import_and_parse
from modules.is_file import main as is_file
from modules.messages import msg_info
from modules.save_to_filename import main as save_to_filename
from modules.train_test_split import main as train_test_split

################
### SETTINGS ###
################
# Define the root directory, which is one level up from this script.
ROOT = Path(__file__).parent.parent.resolve()

# Path to the configuration INI file.
CONFIG_INI = Path(ROOT, 'configuration.ini')

# Define columns that represent the "future" or "result" of the current bar. Keep 'o' (Open) because it is known at the start of the trade.
LEAKY_COLUMNS = ['c', 'h', 'l', 'v', 'n', 'vw']

############
### MAIN ###
############
def main(filename):
    """Main function to perform machine learning on financial data."""
    # Create a timestamp.
    timestamp = date_and_time(n_days_ago = 0, include_time = True).replace(' ', '_').replace(':', '')
    #---------------------#
    #--- Configuration ---#
    #---------------------#
    # Verify the configuration file exists.
    is_file(CONFIG_INI, exit_on_error=True)
    # Initiate the configuration parser.
    configuration_ini = ConfigParser(interpolation = ExtendedInterpolation())
    # Read the configuration INI file.
    configuration_ini.read(CONFIG_INI)
    # Obtain the directory that will contain saved objects.
    dir_data_saved = Path(configuration_ini.get('GENERAL', 'DATA_SAVED_DIRECTORY')).resolve()
    # Obtain the option that determines whether the current run is for production.
    is_production = configuration_ini.getboolean('GENERAL', 'IS_PRODUCTION')
    #------------#
    #--- Data ---#
    #------------#
    # Extract the symbol(s) from the filename.
    symbols = filename.stem
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
    X_train, X_test, y_train, y_test, columns_x, scaler = train_test_split(
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
    #------------#
    #--- Save ---#
    #------------#
    # Check if the current run is a production run.
    if is_production is True:
        # If so, then define the output file for the fitted scaler.
        filename_saved_fittedscaler = save_to_filename(
            dir_data_saved=dir_data_saved,
            name='FittedScaler',
            symbols=symbols,
            extension='joblib',
            random_state=False,
            timestamp=timestamp
        )
        # Define the output file for the selected features.
        filename_saved_selectedfeatures = save_to_filename(
            dir_data_saved=dir_data_saved,
            name='SelectedFeatures',
            symbols=symbols,
            extension='joblib',
            random_state=False,
            timestamp=timestamp
        )
        # Save the fitted scaler.
        dump(scaler, filename_saved_fittedscaler)
        # Display message to stdout.
        msg_info(f"SAVED: Fitted scaler saved to: {filename_saved_fittedscaler}")
        # Save the selected features.
        dump(X_train.columns, filename_saved_selectedfeatures)
        # Display message to stdout.
        msg_info(f"SAVED: Selected features saved to: {filename_saved_selectedfeatures}")
    # Return the training and testing data.
    return X_train, X_test, y_train, y_test