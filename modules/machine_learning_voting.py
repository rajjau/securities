#!/usr/bin/env python
from json import loads
from pathlib import Path
from sklearn.ensemble import VotingClassifier
from functools import partial

######################
### CUSTOM MODULES ###
######################
from modules.calculate_results import COL_CROSSVAL, COL_CROSSVAL_STDDEV, COL_SEED, DECIMAL_PLACES
from modules.dynamic_module_load import main as dynamic_module_load
from modules.machine_learning import train_predict_rolling
from modules.messages import msg_info
from modules.save_model import main as save_model
from modules.save_to_filename import main as save_to_filename

####################
### COLUMN NAMES ###
####################
# Quantifies the difference between model performance and its cross-validation score.
COL_GAP = 'Gap'

#################
### FUNCTIONS ###
#################
def filters(name, scores, configuration_ini):
    # Keep all seeds except the average row.
    scores = scores[scores[COL_SEED].str.upper() != 'AVERAGE']
    # Name of the model performance column.
    col_perf = f"{name} %"
    # Filter negative performance scores (if applicable).
    scores = scores.loc[scores[col_perf] > 0]
    # Filter performance scores less than the median.
    scores = scores.loc[scores[col_perf] >= scores[col_perf].median()]
    # Filter negative cross-validation scores (if applicable).
    scores = scores.loc[scores[COL_CROSSVAL] > 0]
    # Calculate the absolute difference between the performance of the model and its cross-validation score. Lower is better.
    scores[COL_GAP] = abs(scores[col_perf] - scores[COL_CROSSVAL]).astype('float32')
    # Filter gaps that are greater than the median.
    scores = scores.loc[scores[COL_GAP] <= scores[COL_GAP].median()]
    # Sort the specific columns from most important to least. This is a scorer agnostic method of ranking.
    scores = scores.sort_values([COL_CROSSVAL_STDDEV, 'Gap'], ascending = True)
    # Define the total number of models left.
    total_models = len(scores)
    # Obtain the number of top models to keep from the configuration INI.
    n_models = configuration_ini.get('ML', 'VOTINGCLASSIFIER_TOP_N_MODELS').strip()
    # Check if all models were selected to be kept.
    if n_models == '*':
        # If so, then set the number of top models to the total number of models.
        n_models = total_models
    else:
        # Otherwise, set the number of top models. `min` is used to ensure the top N models isn't greater than the total number of models left.
        n_models = min(total_models, int(n_models))
    # Define model indexes to keep.
    idx = scores[COL_SEED].index[:n_models]
    # Return the indexes.
    return idx

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
def main(X_train, y_train, X_test, y_test, name, symbols, estimators, scores, configuration_ini):
    #--------------#
    #--- Filter ---#
    #--------------#
    # Apply a series of filters to identify the index of seeds of models that should be kept.
    idx = filters(name=name, scores=scores, configuration_ini=configuration_ini)
    # Apply the indexes to the list of estimators.
    estimators = [estimators[i] for i in idx]
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