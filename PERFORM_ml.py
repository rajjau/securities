#!/usr/bin/env python
from argparse import ArgumentParser
from configparser import ConfigParser, ExtendedInterpolation
from pathlib import Path

######################
### CUSTOM MODULES ###
######################
from modules.border import main as border
from modules.calculate_results import main as calculate_results
from modules.convert_to_list import main as convert_to_list
from modules.is_file import main as is_file
from modules.machine_learning import main as machine_learning
from modules.machine_learning_voting import main as machine_learning_voting
from modules.messages import msg_info
from modules.preprocessing import main as preprocessing

################
### SETTINGS ###
################
# Define the root directory, which is the parent directory of this script.
ROOT = Path(__file__).parent.resolve()

# Path to the configuration INI file.
CONFIG_INI = Path(ROOT, 'configuration.ini')

# Path to the YAML file containing all learners.
LEARNERS_YAML = Path(ROOT, 'learners.yaml')

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
    # Return the filename and tickers.
    return args.filename.absolute()

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
    configuration_ini = ConfigParser(interpolation = ExtendedInterpolation())
    # Read the configuration INI file.
    configuration_ini.read(CONFIG_INI)
    #---------------------#
    #--- Preprocessing ---#
    #---------------------#
    X_train, X_test, y_train, y_test = preprocessing(filename=filename)
    # Define the random seed(s).
    random_seeds = [int(item) for item in convert_to_list(string=configuration_ini['GENERAL']['RANDOM_SEED'], delimiter=',')]
    # Extract the ticker(s) from the input filename.
    tickers = filename.stem
    #------------------------#
    #--- Machine Learning ---#
    #------------------------#
    # Define the machine learning algorithms to use.
    use_learners = convert_to_list(string=configuration_ini['ML']['USE_LEARNERS'], delimiter=',')
    # Iterate through each random seed.
    for learner in use_learners:
        # Create a border to denote a process.
        border(f"MACHINE LEARNING: {learner}", border_char='><')
        # Initialize accumulators to zero at the start of each learner's seed loop.
        total_pipelines = []
        total_score = []
        total_cv_score = []
        total_cv_std = []
        # Iterate through each random seed.
        for seed in random_seeds:
            # Message to stdout.
            msg_info(f"Seed {seed}")
            # Perform machine learning.
            score_seed, score_cv_seed, score_cv_stddev_seed, pipeline_seed = machine_learning(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                name=learner,
                tickers=tickers,
                random_state=seed,
                configuration_ini=configuration_ini,
                learners_yaml=LEARNERS_YAML
            )
            # Add the unfitted pipeline for the current $seed to the total list.
            total_pipelines.append((f"{learner}_{seed}", pipeline_seed))
            # Add the score for the current $learner for the current $seed.
            total_score.append(score_seed)
            # Add the cross-validation score and standard deviation for the current $learner for the current $seed.
            total_cv_score.append(score_cv_seed)
            total_cv_std.append(score_cv_stddev_seed)      
        #-------------------#
        #--- Seed Scores ---#
        #-------------------#
        # Define the output filename.
        filename_output = Path(ROOT, f'RESULTS_{learner.replace(" ", "_")}.csv').absolute()
        # Calculate and display the average scores for the current learner across all random seeds.
        scores = calculate_results(
            learner=learner,
            total_score=total_score,
            total_cv_score=total_cv_score,
            total_cv_std=total_cv_std,
            random_seeds=random_seeds,
            save_results_to_file=configuration_ini.getboolean('GENERAL', 'SAVE_RESULTS_TO_FILE'),
            filename_output=filename_output
        )
        #----------------------------#
        #--- Consensus Prediction ---#
        #----------------------------#
        machine_learning_voting(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            name=learner,
            tickers=tickers,
            pipelines=total_pipelines,
            scores=scores,
            configuration_ini=configuration_ini
        )  

#############
### START ###
#############
if __name__ == '__main__':
    # Obtain user-defined variables.
    filename = args()
    # Start the script.
    main(filename=filename)