#!/usr/bin/env python
from json import loads
from pathlib import Path
from sklearn.ensemble import VotingClassifier
from functools import partial

######################
### CUSTOM MODULES ###
######################
from modules.calculate_results import DECIMAL_PLACES
from modules.dynamic_module_load import main as dynamic_module_load
from modules.machine_learning import train_predict_rolling
from modules.messages import msg_info
from modules.save_model import main as save_model
from modules.save_to_filename import main as save_to_filename

#################
### FUNCTIONS ###
#################
def create_scoring_metric(configuration_ini):
    # Dynamically load the scoring metric.
    scoring_metric = dynamic_module_load(module_str=configuration_ini.get('ML', 'SCORING_METRIC'))
    # Load extra parameters defined in the configuration.ini.
    scoring_metric_params = loads(configuration_ini.get('ML', 'SCORING_METRIC_PARAMETERS'))
    # Create the scorer.
    scorer = partial(scoring_metric, **scoring_metric_params)
    # Return the scorer.
    return scorer

############
### MAIN ###
############
def main(X_train, y_train, X_test, y_test, name, symbols, estimators, configuration_ini):
    #-------------#
    #--- Model ---#
    #-------------#
    # Define the Voting model using unfitted models from every seed.
    model = VotingClassifier(estimators = estimators, voting = 'soft')
    #----------------#
    #--- Training ---#
    #----------------#
    # Check if the test set has been defined.
    if (X_test is not None) and (y_test is not None):
        # If so, then perform walk-forward rolling retraining.
        score = train_predict_rolling(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            retrain_step_frequency=configuration_ini.getint('ML', 'RETRAIN_STEP_FREQUENCY'),
            sliding_window_size=configuration_ini.getint('ML', 'SLIDING_WINDOW_SIZE'),
            scoring_metric=create_scoring_metric(configuration_ini=configuration_ini)
        )
        # Message to stdout.
        msg_info(f"VOTING CONSENSUS SCORE: {round(score * 100, DECIMAL_PLACES)}")
    #------------#
    #--- Save ---#
    #------------#
    # Generate the filename for saving the model to an output file.
    saved_model = save_to_filename(
        dir_data_saved=Path(configuration_ini.get('GENERAL', 'DATA_SAVED_DIRECTORY')).resolve(),
        name=f"Consenus_Model_{name}",
        symbols=symbols,
        extension='joblib',
        random_state=False,
        timestamp=False
    )
    # Save the model if performance requirements are met.
    save_model(
        saved_model=saved_model,
        model=model,
        score=score,
        save_threshold=configuration_ini.getfloat('ML', 'SAVE_THRESHOLD'),
        is_production=configuration_ini.getboolean('GENERAL', 'IS_PRODUCTION')
    )