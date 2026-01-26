#!/usr/bin/env python
from functools import partial
from json import loads
from pandas import concat
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.pipeline import Pipeline

######################
### CUSTOM MODULES ###
######################
from modules.load_learners import main as load_learners
from modules.dynamic_module_load import main as dynamic_module_load
from modules.feature_selection_wrapper import FeatureSelection
from modules.hyperparameter_optimization import main as hyperparameter_optimization
from modules.messages import msg_info, msg_warn
from modules.set_universal_learner_params import main as set_universal_learner_params

################
### WARNINGS ###
################
from scipy.linalg import LinAlgWarning
from sklearn.exceptions import FitFailedWarning
from warnings import filterwarnings

filterwarnings('ignore', category = FitFailedWarning)
filterwarnings('ignore', category = LinAlgWarning)
filterwarnings('ignore', category = UserWarning)

#################
### FUNCTIONS ###
#################
def build_pipeline(model, random_state, configuration_ini):
    # Dynamically load the normalization method.
    scaler = dynamic_module_load(module_str=configuration_ini.get('NORMALIZATION', 'NORMALIZE_METHOD'))()
    # Define the feature selection step.
    feature_selection = FeatureSelection(configuration_ini=configuration_ini, random_state=random_state)
    # Instantiate the pipeline.
    pipeline = Pipeline(steps=[
        ('replace_missing_values', SimpleImputer(strategy = 'mean')),
        ('normalization', scaler),
        ('feature_selection', feature_selection),
        ('model', model)
    ])
    # Ensure the pipeline outputs DataFrame objects.
    pipeline.set_output(transform = 'pandas')
    # Return the $pipeline.
    return pipeline

def cross_validation(model, X, y, cross_validation_folds, scoring):
    # Pass the scoring object to maintain consistency with HPO.
    score_cv = cross_val_score(model, X, y, cv=TimeSeriesSplit(n_splits=cross_validation_folds), scoring=scoring)
    # Return the mean and standard deviation.
    return score_cv.mean(), score_cv.std()

def train_predict_rolling(model, X_train, y_train, X_test, y_test, retrain_step_frequency, sliding_window_size, scoring_metric):
    # Initialize a list to store model predictions for each step.
    y_preds = []
    # Total length of the training set.
    total_train = len(X_train)
    # Total length of the test set.
    total_test = len(X_test)
    # Loop through the test data using the retraining frequency as the step size.
    for i in range(0, total_test, retrain_step_frequency):
        # Calculate the absolute position in the sequence (train + current test offset).
        current_cutoff = total_train + i
        # Check if the sliding window was specified.
        if sliding_window_size > 0:
            # If so, then calculate the start of the window, ensuring it doesn't go below index 0.
            start_idx = max(0, current_cutoff - sliding_window_size)
        else:
            # Use the entire training set.
            start_idx = 0
        # Determine if the training window spans across both the train and test sets.
        if start_idx < total_train:
            # Create the current window training set by joining the end of X_train with the current X_test progress
            X_curr_train = concat([X_train.iloc[start_idx:], X_test.iloc[:i]])
            # Create the corresponding label set.
            y_curr_train = concat([y_train.iloc[start_idx:], y_test.iloc[:i]])
        else:
            # If the window has moved past the training set, use from the testing set.
            X_curr_train = X_test.iloc[start_idx - total_train:i]
            # Create the corresponding label by taking from the test label set. 
            y_curr_train = y_test.iloc[start_idx - total_train:i]
        # Calculate the end index for the current window.
        end_idx = min(i + retrain_step_frequency, total_test)
        # Create the test set for the current window.
        X_curr_test = X_test.iloc[i:end_idx]
        # Train the model.
        model.fit(X_curr_train, y_curr_train)
        # Calculate predictions and append them to the results list.
        y_preds.extend(model.predict(X_curr_test))
    # Align the true labels with the total number of predictions made.
    y_true = y_test.iloc[:len(y_preds)]
    # Compute the score.
    score = scoring_metric(y_true = y_true, y_pred = y_preds)
    # Return the $score.
    return score

############
### MAIN ###
############
def main(X_train, y_train, X_test, y_test, name, pipeline, random_state, configuration_ini, learners_yaml):
    # Obtain number of cross-validation folds from configuration file.
    cross_validation_folds = configuration_ini.getint('ML', 'CROSS_VALIDATION_FOLDS')
    #--------------#
    #--- Models ---#
    #--------------#
    # Load configuration to populate global dictionaries.
    learners, learners_hyperparameters = load_learners(learners_yaml=learners_yaml)
    # Iterate through each learner to convert hyperparameter keys to Pipeline format.
    for learner in learners_hyperparameters.keys():
        # Add the 'model__' prefix to each hyperparameter for compatibility with the pipeline.
        learners_hyperparameters[learner] = {f'model__{x}': y for x, y in learners_hyperparameters[learner].items() if not x.startswith('model__')}
    #----------------------#
    #--- Scoring Metric ---#
    #----------------------#
    # Dynamically load the scoring metric.
    scoring_metric = dynamic_module_load(module_str=configuration_ini.get('ML', 'SCORING_METRIC'))
    # Load extra parameters defined in the configuration.ini.
    scoring_metric_params = loads(configuration_ini.get('ML', 'SCORING_METRIC_PARAMETERS'))
    # Check if the pipeline is Nonetype, meaning it needs to be defined.
    if pipeline is None:
        #---------------#
        #--- Pipeline --#
        #---------------#
        pipeline = build_pipeline(model=learners[name], random_state=random_state, configuration_ini=configuration_ini)
        #-----------------------------------#
        #--- Hyperparameter Optimization ---#
        #-----------------------------------#
        # Check if HPO was enabled.
        if configuration_ini.getboolean('GENERAL', 'PERFORM_HYPERPARAMETER_OPTIMIZATION') is True:
            # Dislay message to stdout.
            msg_info('Hyperparameter optimization enabled.')
            # Execute HPO only on the training set to prevent leakage from the test set.
            pipeline = hyperparameter_optimization(
                X_train=X_train,
                y_train=y_train,
                pipeline=pipeline,
                name=name,
                learners=learners,
                learners_hyperparameters=learners_hyperparameters,
                cross_validation_folds=configuration_ini.getint('ML', 'CROSS_VALIDATION_FOLDS'),
                scoring=make_scorer(score_func = scoring_metric, **scoring_metric_params)
            )
        else:
            # Display message to stdout.
            msg_warn('SKIP: Hyperparameter optimization was not enabled.')
        # Return the $pipeline containing models with optimized hyperparameters. HPO only needs to be ran once. This `return` works fine if HPO is not enabled.
        return pipeline
    # Apply universal settings like the random seed to the pipeline.
    pipeline = set_universal_learner_params(model=pipeline, random_state=random_state)
    # Create a clone of the $pipeline before the model is fitted.
    pipeline_clone = clone(pipeline)
    #------------------------#
    #--- Cross-Validation ---#
    #------------------------#
    # Set the cross-validation and cross-validation standard deviation (stddev) as Nonetype.
    score_cv, score_cv_stddev = (None, None)
    # Check if the option to perform cross-validation was enabled.
    if configuration_ini.getboolean('GENERAL', 'PERFORM_CROSS_VALIDATION') is True:
        # Perform cross-validation and return its score and stddev.
        score_cv, score_cv_stddev = cross_validation(
            model=pipeline,
            X=X_train,
            y=y_train,
            cross_validation_folds=cross_validation_folds,
            scoring=make_scorer(score_func = scoring_metric, **scoring_metric_params)
        )
    #----------------#
    #--- Training ---#
    #----------------#
    # Check if the test set has been defined.
    if (X_test is not None) and (y_test is not None):
        # If so, then perform walk-forward rolling retraining.
        score = train_predict_rolling(
            model=pipeline,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            retrain_step_frequency=configuration_ini.getint('ML', 'RETRAIN_STEP_FREQUENCY'),
            sliding_window_size=configuration_ini.getint('ML', 'SLIDING_WINDOW_SIZE'),
            scoring_metric=partial(scoring_metric, **scoring_metric_params)
        )
    else:
        # Otherwise, fit the pipeline on the full training set for production use.
        pipeline.fit(X_train, y_train)
        # Set the $score to Nonetype since there will be no prediction on the test set, as it doesn't exist.
        score = None
    # Return the pipeline clone and the scores.
    return pipeline_clone, score, score_cv, score_cv_stddev