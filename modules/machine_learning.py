#!/usr/bin/env python
from functools import partial
from json import loads
from numpy import inf
from pandas import concat
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from yaml import safe_load

######################
### CUSTOM MODULES ###
######################
from modules.dynamic_module_load import main as dynamic_module_load
from modules.feature_selection_wrapper import FeatureSelection
from modules.messages import msg_info, msg_warn

################
### WARNINGS ###
################
from scipy.linalg import LinAlgWarning
from sklearn.exceptions import FitFailedWarning
from warnings import filterwarnings

filterwarnings('ignore', category = FitFailedWarning)
filterwarnings('ignore', category = LinAlgWarning)
filterwarnings('ignore', category = UserWarning)

################
### LEARNERS ###
################
# Global dictionaries to store config
learners = {}
learners_hyperparameters = {}

def load_learners(learners_yaml):
    # Access the global dictionaries so they can be populated for use in other functions.
    global learners, learners_hyperparameters
    # Open the YAML file and load the 'LEARNERS' section into a configuration dictionary.
    with open(learners_yaml, 'r') as f: config = safe_load(f)['LEARNERS']
    # Iterate through each learner defined in the configuration.
    for name, info in config.items():
        # Import the module using the specified string.
        module_class = dynamic_module_load(module_str=info['class'])        
        # Extract the parameters from the configuration.
        params = info.get('params', {})
        # Instantiate the learner and store it in the global $learners dictionary.
        learners[name] = module_class(**params)
        # Extract the optimization parameters.
        optimization_params = info.get('optimization', {})
        # Check if the current learner is Logistic Regression to handle the infinity string conversion.
        if name == 'Logistic Regression' and 'C' in optimization_params:
            # Convert the '.inf' string from YAML into a float infinity value.
            optimization_params['C'] = [inf if x == '.inf' else x for x in optimization_params['C']]
        # Store the search grid in the global $learners_hyperparameters dictionary with the pipeline prefix.
        learners_hyperparameters[name] = {f'model__{k}': v for k, v in optimization_params.items()}

#################
### FUNCTIONS ###
#################
def build_pipeline(name, random_state, configuration_ini):
    # Dynamically load the normalization method.
    scaler = dynamic_module_load(module_str=configuration_ini.get('NORMALIZATION', 'NORMALIZE_METHOD'))()
    # Define the feature selection step.
    feature_selection = FeatureSelection(configuration_ini=configuration_ini, random_state=random_state)
    # Instantiate the pipeline.
    pipeline = Pipeline(steps=[
        ('replace_missing_values', SimpleImputer(strategy = 'mean')),
        ('normalization', scaler),
        ('feature_selection', feature_selection),
        ('model', learners[name])
    ])
    # Ensure the pipeline outputs DataFrame objects.
    pipeline.set_output(transform = 'pandas')
    # Apply universal settings like the random seed to the pipeline.
    pipeline = set_universal_params(pipeline=pipeline, random_state=random_state)
    # Return the $pipeline.
    return pipeline

def set_universal_params(pipeline, random_state):
    # Check if the model step exists to set its seed.
    try:
        pipeline.named_steps['model'].set_params(random_state = random_state)
    except (ValueError, AttributeError):
        pass
    # Return the $pipeline.
    return pipeline

def hyperparameter_optimization(X_train, y_train, pipeline, name, cross_validation_folds, random_state, scoring):
    try:
        # Obtain the parameters from the dictionary defined above.
        params = learners_hyperparameters[name].copy()
    except KeyError:
        # If no hyperparameters were defined for $name, then raise an error.
        msg_warn(f"The following learner name has no entries within the hyperparameter dictionary: '{name}'")
        # Return the default learner instead.
        return pipeline
    # Check if the current model is Bagging to handle its sub-estimators.
    if name == 'Bagging':
        # Retrieve the list of potential base estimators using the pipeline prefix.
        estimator_options = params.pop('model__estimator_options', [])
        # Define a list to store the optimized versions of these base estimators.
        best_base_models = []
        # Iterate through each estimator.
        for estimator_name in estimator_options:
            # Recursively call this function to find the best hyperparameters for the base learner.
            msg_info(f"Optimizing base estimator '{estimator_name}' for the Bagging ensemble...")
            optimized_base = hyperparameter_optimization(
                X_train=X_train,
                y_train=y_train,
                pipeline=pipeline,
                name=estimator_name,
                cross_validation_folds=cross_validation_folds,
                random_state=random_state,
                scoring=scoring
            )
            best_base_models.append(optimized_base)
        # Update the parameter grid to include the optimized model objects.
        params['model__estimator'] = best_base_models
    # Define a time series split object to prevent future data from leaking into the training folds during optimization.
    timeseries_k_fold = TimeSeriesSplit(n_splits = cross_validation_folds)
    # Set universal parameters on the pipeline.
    pipeline = set_universal_params(pipeline=pipeline, random_state=random_state)
    # Define the RandomizedSearchCV object using the TimeSeriesSplit object and the full pipeline.
    # search = RandomizedSearchCV(
    #     cv = timeseries_k_fold,
    #     estimator = pipeline,
    #     n_iter = 50,
    #     n_jobs = -1,
    #     param_distributions = params,
    #     random_state = random_state,
    #     scoring = scoring
    # )
    search = GridSearchCV(
        cv = timeseries_k_fold,
        estimator = pipeline,
        n_jobs = -1,
        param_grid = params,
        scoring = scoring
    )
    # Fit the model to the training data.
    search.fit(X_train, y_train)
    # Identify the model with the best performance.
    best_estimator = search.best_estimator_
    # Return the $best_estimator.
    return best_estimator

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
def main(X_train, y_train, X_test, y_test, name, random_state, configuration_ini, learners_yaml):
    # Obtain number of cross-validation folds from configuration file.
    cross_validation_folds = configuration_ini.getint('ML', 'CROSS_VALIDATION_FOLDS')
    #--------------#
    #--- Models ---#
    #--------------#
    # Load configuration to populate global dictionaries.
    load_learners(learners_yaml=learners_yaml)
    #---------------#
    #--- Pipeline --#
    #---------------#
    pipeline = build_pipeline(name=name, random_state=random_state, configuration_ini=configuration_ini)
    #----------------------#
    #--- Scoring Metric ---#
    #----------------------#
    # Dynamically load the scoring metric.
    scoring_metric = dynamic_module_load(module_str=configuration_ini.get('ML', 'SCORING_METRIC'))
    # Load extra parameters defined in the configuration.ini.
    scoring_metric_params = loads(configuration_ini.get('ML', 'SCORING_METRIC_PARAMETERS'))
    #-----------------------------------#
    #--- Hyperparameter Optimization ---#
    #-----------------------------------#
    # Execute hyperparameter optimization (HPO) only on the training set to prevent leakage from the test set.
    if configuration_ini.getboolean('GENERAL', 'PERFORM_HYPERPARAMETER_OPTIMIZATION') is True:
        # Perform hyperparameter optimization.
        pipeline = hyperparameter_optimization(
            X_train=X_train,
            y_train=y_train,
            pipeline=pipeline,
            name=name,
            cross_validation_folds=configuration_ini.getint('ML', 'CROSS_VALIDATION_FOLDS'),
            random_state=random_state,
            scoring=make_scorer(score_func = scoring_metric, **scoring_metric_params)
        )
    # Create a clone of the unfitted pipeline.
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
    # Return the scores and the pipeline clone.
    return score, score_cv, score_cv_stddev, pipeline_clone