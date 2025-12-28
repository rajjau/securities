#!/usr/bin/env python
from joblib import dump
from importlib import import_module
from numpy import inf
from pathlib import Path
from pandas import concat, Series
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, TimeSeriesSplit
from yaml import safe_load

######################
### CUSTOM MODULES ###
######################
from modules.date_and_time import main as date_and_time
from modules.messages import msg_info,msg_warn

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
        # Split the class string so the learner can be imported.
        module_path, class_name = info['class'].rsplit('.', 1)
        # Dynamically import the required module.
        module = import_module(module_path)
        # Retrieve the model class from the imported module.
        model_class = getattr(module, class_name)
        # Extract the fixed parameters from the configuration.
        fixed_params = info.get('params', {})
        # Instantiate the learner and store it in the global $learners dictionary.
        learners[name] = model_class(**fixed_params)
        # Extract the optimization parameters.
        optimization_params = info.get('optimization', {})
        # Check if the current learner is Logistic Regression to handle the infinity string conversion.
        if name == 'Logistic Regression' and 'C' in optimization_params:
            # Convert the '.inf' string from YAML into a float infinity value.
            optimization_params['C'] = [inf if x == '.inf' else x for x in optimization_params['C']]
        # Store the search grid in the global $learners_hyperparameters dictionary.
        learners_hyperparameters[name] = optimization_params

#################
### FUNCTIONS ###
#################
def set_universal_params(model, random_state):
    try:
        # Use the current seed (if applicable).
        model.set_params(random_state = random_state)
    except ValueError:
        pass
    # Return the $model.
    return model

def hyperparameter_optimization(X_train, y_train, name, cross_validation_folds, random_state):
    try:
        # Obtain the parameters from the dictionary defined above.
        params = learners_hyperparameters[name].copy()
    except KeyError:
        # If no hyperparameters were defined for $name, then raise an error.
        msg_warn(f"The following learner name has no entries within the hyperparameter dictionary: '{name}'")
        # Return the default learner instead.
        return learners[name]
    # Check if the current model is Bagging to handle its sub-estimators.
    if name == 'Bagging':
        # Retrieve the list of potential base estimators.
        estimator_options = params.pop('estimator_options', [])
        # Define a list to store the optimized versions of these base estimators.
        best_base_models = []
        # Iterate through each estimator.
        for est_name in estimator_options:
            # Recursively call this function to find the best hyperparameters for the base learner.
            msg_info(f"Optimizing base estimator '{est_name}' for the Bagging ensemble...")
            optimized_base = hyperparameter_optimization(X_train=X_train, y_train=y_train, name=est_name, cross_validation_folds=cross_validation_folds, random_state=random_state)
            best_base_models.append(optimized_base)
        # Update the parameter grid to include the optimized model objects.
        params['estimator'] = best_base_models
    # Define a time series split object to prevent future data from leaking into the training folds during optimization.
    timeseries_k_fold = TimeSeriesSplit(n_splits = cross_validation_folds)
    # Define the learner.
    model = learners[name]
    # Set universal parameters.
    model = set_universal_params(model=model, random_state=random_state)
    # Define the RandomizedSearchCV object using the TimeSeriesSplit object.
    search = RandomizedSearchCV(cv = timeseries_k_fold, estimator = model, n_jobs = -1, param_distributions = params, random_state = random_state, scoring = 'f1_macro', n_iter = 10)
    # Fit the model to the training data.
    search.fit(X_train, y_train)
    # Identify the model with the best performance.
    best_model = search.best_estimator_
    # Return the $best_model.
    return best_model

def cross_validation(model, X, y, cross_validation_folds):
    """Run cross-validation using chronological folds."""
    # Execute cross-validation using TimeSeriesSplit to ensure no look-ahead bias during validation.
    score_cv = cross_val_score(model, X, y, cv=TimeSeriesSplit(n_splits=cross_validation_folds), scoring='f1_macro')
    # Return the cross-validation mean score and standard deviation.
    return score_cv.mean(), score_cv.std()

def train_predict_rolling(model, X_train, y_train, X_test, y_test, retrain_step_frequency, sliding_window_size):
    # Initialize a list to collect predictions from each rolling step.
    y_preds = []
    # Ensure the y train and test are Series type.
    y_train, y_test = Series(y_train), Series(y_test)
    # Combine training and testing features/labels into single objects for efficient slicing.
    X_full = concat([X_train, X_test], axis = 0)
    y_full = concat([y_train, y_test], axis = 0)
    # Identify the starting point and total length for the walk-forward simulation.
    initial_train_len = len(X_train)
    # Calculate the total number of rows in the full dataset.
    total_len = len(X_full)
    # Iterate through the test data using the defined frequency to simulate periodic re-training.
    for i in range(initial_train_len, total_len, retrain_step_frequency):
        # If a sliding window has been defined, then use that chunk of history rather than all history. This helps with memory and also ensures that data from 2023 isn't treated the same as data from 2026 for example. We keep looking at the most recent N chunks of data.
        if sliding_window_size > 0:
            start_idx = max(0, i - sliding_window_size)
        else:
            # If set to 0 then use all history.
            start_idx = 0
        # Slice historical data from the beginning of the dataset up to the current index (Expanding Window).
        X_curr_train = X_full.iloc[start_idx:i]
        y_curr_train = y_full.iloc[start_idx:i]
        # Calculate the end index for the prediction chunk.
        end_idx = min(i + retrain_step_frequency, total_len)
        # Define the chunk that the model will predict.
        X_chunk = X_full.iloc[i:end_idx]
        # Re-fit the model on the updated historical dataset.
        model.fit(X_curr_train, y_curr_train)
        # Predict the next chunk and append results.
        y_preds.extend(model.predict(X_chunk))
    # Calculate the F1 Macro score comparing rolling predictions against true test labels.
    score = f1_score(y_true = y_test, y_pred = y_preds, average = 'macro')
    # Return the $score.
    return score

def saved_model_filename(name, symbols, random_state):
    """Define the output filename for saving the trained model to."""
    # Define today's date and time for the filename.
    timestamp = date_and_time(n_days_ago = 1, include_time = True).replace(' ', '_').replace(':', '')
    # Prepare the learner name for the filename.
    name = name.replace(' ', '_')
    # Build the filename using symbols (if provided) or 'all'.
    filename = f"{name}_{'_'.join(symbols) if symbols else 'all'}_{timestamp}_seed{random_state}.joblib"
    # Use `pathlib` to return the absolute path.
    return Path(filename).absolute()

def save(saved_model, model, score, save_threshold):
    # Check if saving is enabled and threshold is met.
    if save_threshold == -1: return None
    # Check if the $score is Nonetype, meaning the model is being trained on the entire training set, or if it's greater than or equal to the specified save threshold.
    if (score is None) or (score >= save_threshold):
        # Save the model object.
        dump(model, filename = saved_model)
        # Message to stdout.
        msg_info(f"The model has been saved to: {saved_model}")

############
### MAIN ###
############
def main(X_train, y_train, X_test, y_test, name, learners_yaml, symbols, random_state, perform_hyperparameter_optimization, perform_cross_validation, cross_validation_folds, retrain_step_frequency, sliding_window_size, save_threshold):
    # Load configuration to populate global dictionaries.
    load_learners(learners_yaml=learners_yaml)
    # Execute hyperparameter optimization (HPO) only on the training set to prevent leakage from the test set.
    if perform_hyperparameter_optimization is True:
        # Perform hyperparameter optimization.
        model = hyperparameter_optimization(
            X_train=X_train,
            y_train=y_train,
            name=name,
            cross_validation_folds=cross_validation_folds,
            random_state=random_state
        )
    else:
        # Retrieve the learner with default parameters.
        model = learners[name]
    # Apply universal settings like the random seed.
    model = set_universal_params(model=model, random_state=random_state)
    # Set the cross-validation and cross-validation standard deviation (stddev) as Nonetype.
    score_cv, score_cv_stddev = (None, None)
    # Check if the option to perform cross-validation was enabled.
    if perform_cross_validation is True:
        # Perform cross-validation and return its score and stddev.
        score_cv, score_cv_stddev = cross_validation(model=model, X=X_train, y=y_train, cross_validation_folds=cross_validation_folds)
    # Check if the test set has been defined.
    if (X_test is not None) and (y_test is not None):
        # If so, then perform walk-forward rolling retraining.
        score = train_predict_rolling(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            retrain_step_frequency=retrain_step_frequency,
            sliding_window_size=sliding_window_size
        )
    else:
        # Otherwise, fit the model on the full training set for production use.
        model.fit(X_train, y_train)
        # Set the $score to Nonetype since there will be no prediction on the test set, as it doesn't exist.
        score = None
    # Generate the filename for saving the model to an output file.
    saved_model = saved_model_filename(name=name, symbols=symbols, random_state=random_state)
    # Save the model if performance requirements are met.
    save(saved_model=saved_model, model=model, score=score, save_threshold=save_threshold)
    # Return the scores.
    return score, score_cv, score_cv_stddev