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
from modules.messages import msg_info
from modules.save_model import main as save_model
from modules.save_to_filename import main as save_to_filename

############
### MAIN ###
############
def main(X_train, y_train, X_test, y_test, name, symbols, estimators, configuration_ini):
    # Define the Voting model using unfitted models from every seed.
    model = VotingClassifier(estimators = estimators, voting = 'soft')
    # Fit the $model to the training set.
    model.fit(X = X_train, y = y_train)
    # Check if the test set has been defined.
    if (X_test is not None) and (y_test is not None):
        # Calculate predictions on the testing set.
        y_preds = model.predict(X = X_test)
        # Dynamically load the scoring metric.
        scoring_metric = dynamic_module_load(module_str=configuration_ini.get('ML', 'SCORING_METRIC'))
        # Load extra parameters defined in the configuration.ini.
        scoring_metric_params = loads(configuration_ini.get('ML', 'SCORING_METRIC_PARAMETERS'))
        # Create the scorer.
        scorer = partial(scoring_metric, **scoring_metric_params)
        # Calculate the score.
        score = scorer(y_true = y_test, y_pred = y_preds)
        # Message to stdout.
        msg_info(f"VOTING CONSENSUS SCORE: {round(score * 100, DECIMAL_PLACES)}")
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