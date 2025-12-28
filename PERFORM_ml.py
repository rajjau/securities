#!/usr/bin/env python
from argparse import ArgumentParser
from configparser import ConfigParser
from pathlib import Path

######################
### CUSTOM MODULES ###
######################
from modules.border import main as border
from modules.feature_selection import main as feature_selection
from modules.calculate_results import main as calculate_results
from modules.import_and_parse import main as import_and_parse
from modules.is_file import main as is_file
from modules.machine_learning import main as machine_learning
from modules.messages import msg_info
from modules.train_test_split import main as train_test_split

################
### SETTINGS ###
################
# Define the root directory, which is the parent directory of this script.
ROOT = Path(__file__).parent.resolve()

# Path to the configuration INI file.
CONFIG_INI = Path(ROOT, 'configuration.ini')

# Path to the YAML file containing all learners.
LEARNERS_YAML = Path(ROOT, 'learners.yaml')

# Define columns that represent the "future" or "result" of the current bar. Keep 'o' (Open) because it is known at the start of the trade.
LEAKY_COLUMNS = ['c', 'h', 'l', 'v', 'n', 'vw']

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

def convert_to_list(string, delimiter):
    """Split a string representation of a list into an actual list based on the provided $delimiter."""
    return [item.strip() for item in string.split(delimiter)]

def calculate_baseline_accuracy(counts):
    """Calculate the baseline accuracy, which is the number of entries within the largest class divided by the total entries across all classes."""
    baseline_accuracy = counts.max() / counts.sum()
    # Display the baseline accuracy to stdout.
    msg_info(f"Num. of Class 0: {counts[0.0]} | Num. of Class 1: {counts[1.0]} | Baseline: {baseline_accuracy:.2%}")

def data_feature_selection(X_train, y_train, X_test, columns_x, perform_feature_selection):
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
        msg_info(f"Kept {len(selected_features)} out of {len(columns_x)} total features.")
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
    # Verify the learners YAML file exists.
    is_file(LEARNERS_YAML, exit_on_error=True)
    # Initiate the configuration parser.
    configuration_ini = ConfigParser()
    # Read the configuration INI file.
    configuration_ini.read(CONFIG_INI)
    #------------#
    #--- Data ---#
    #------------#
    # Define the symbols to process.
    symbols = convert_to_list(string=configuration_ini['DATA']['SYMBOLS'], delimiter=',')
    # Define the label column(s) (y).
    columns_y = convert_to_list(string=configuration_ini['DATA']['COLUMNS_Y'], delimiter=',')
    # Import and preprocess data for ML.
    data, columns_x = import_and_parse(
        filename=filename,
        symbols=symbols,
        cache_directory=Path(ROOT, configuration_ini['GENERAL']['CACHE_DIRECTORY']).absolute(),
        columns_x=convert_to_list(string=configuration_ini['DATA']['COLUMNS_X'], delimiter=','),
        columns_y=columns_y
    )
    # Calculate the baseline accuracy and display it to stdout.
    calculate_baseline_accuracy(data[columns_y].value_counts())
    # Everything below this point assumes that the data has been successfully imported and preprocessed. The cache is now available for future runs.
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
    #-------------------------#
    #--- Feature Selection ---#
    #-------------------------#
    # Perform feature selection.
    X_train, X_test = data_feature_selection(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        columns_x=columns_x,
        perform_feature_selection=configuration_ini.getboolean('GENERAL', 'PERFORM_FEATURE_SELECTION')
    )
    #------------------------#
    #--- Machine Learning ---#
    #------------------------#
    # Define the machine learning algorithms to use.
    use_learners = convert_to_list(string=configuration_ini['ML']['USE_LEARNERS'], delimiter=',')
    # Define the random seed(s).
    random_seeds = [int(item) for item in convert_to_list(string=configuration_ini['GENERAL']['RANDOM_SEED'], delimiter=',')]
    # Iterate through each random seed.
    for learner in use_learners:
        # Create a border to denote a process.
        border(f"MACHINE LEARNING: {learner}", border_char='*')
        # Initialize accumulators to zero at the start of each learner's seed loop
        total_score = []
        total_cv_score = []
        total_cv_std = []
        # Iterate through each random seed.
        for seed in random_seeds:
            # Perform machine learning.
            score_seed, score_cv_seed, score_cv_stddev_seed = machine_learning(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                name=learner,
                learners_yaml=LEARNERS_YAML,
                symbols=symbols,
                random_state=seed,
                configuration_ini=configuration_ini
            )
            # Add the score for the current $learner for the current $seed.
            total_score.append(score_seed)
            # Add the cross-validation score and standard deviation for the current $learner for the current $seed.
            total_cv_score.append(score_cv_seed)
            total_cv_std.append(score_cv_stddev_seed)
        #-------------#
        #--- Score ---#
        #-------------#
        # Define the output filename.
        output_filename = Path(ROOT, f'RESULTS_{learner.replace(" ", "_")}.csv').absolute()
        # Calculate and display the average scores for the current learner across all random seeds.
        calculate_results(
            learner=learner,
            total_score=total_score,
            total_cv_score=total_cv_score,
            total_cv_std=total_cv_std,
            random_seeds=random_seeds,
            save_results_to_file=configuration_ini.getboolean('GENERAL', 'SAVE_RESULTS_TO_FILE'),
            output_filename=output_filename
        )

#############
### START ###
#############
if __name__ == '__main__':
    # Obtain user-defined variables.
    filename = args()
    # Start the script.
    main(filename=filename)