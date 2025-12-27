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
        # Split the class string so the learner can be imported. For example, sklearn.ensemble.BaggingClassifier -> [sklearn.ensemble, BaggingClassifier].
        module_path, class_name = info['class'].rsplit('.', 1)
        # Dynamically import the required module.
        module = import_module(module_path)
        # Retrieve the model class from the imported module.
        model_class = getattr(module, class_name)
        # Extract the fixed parameters from the configuration.
        fixed_params = info.get('params', {})
        # Instantiate the learner using the fixed parameters and store it in the global $learners dictionary.
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
        # We use .copy() to prevent modifying the global dictionary during recursion.
        params = learners_hyperparameters[name].copy()
    except KeyError:
        # If no hyperparameters were defined for $name, then raise an error.
        msg_warn(f"The following learner name has no entries within the hyperparameter dictionary: '{name}'")
        # Return the default learner instead.
        return learners[name]
    # Check if the current model is Bagging to handle its sub-estimators.
    if name == 'Bagging':
        # Retrieve the list of potential base estimators (e.g., ['Decision Tree', 'Logistic Regression']).
        estimator_options = params.pop('estimator_options', [])
        # Define a list to store the optimized versions of these base estimators.
        best_base_models = []
        for est_name in estimator_options:
            # Recursively call this function to find the best hyperparameters for the base learner.
            msg_info(f"Optimizing base estimator '{est_name}' for the Bagging ensemble...")
            optimized_base = hyperparameter_optimization(
                X_train=X_train, 
                y_train=y_train, 
                name=est_name, 
                cross_validation_folds=cross_validation_folds, 
                random_state=random_state
            )
            best_base_models.append(optimized_base)
        # Update the parameter grid to include the optimized model objects under the 'estimator' key.
        params['estimator'] = best_base_models
    # Define a time series split object.
    timeseries_k_fold = TimeSeriesSplit(n_splits = cross_validation_folds)
    # Define the learner.
    model = learners[name]
    # Set universal parameters.
    model = set_universal_params(model=model, random_state=random_state)
    # Define the RandomizedSearchCV object.
    search = RandomizedSearchCV(
        cv = timeseries_k_fold, 
        estimator = model, 
        n_jobs = -1, 
        param_distributions = params, 
        random_state = random_state, 
        scoring = 'f1_macro',
        n_iter = 10
    )
    # Fit the model with all combinations of hyperparameters to the training data.
    search.fit(X_train, y_train)
    # Identify the model with the best performance.
    best_model = search.best_estimator_
    # Return the $best_model.
    return best_model

def cross_validation(model, X, y, cross_validation_folds):
    """Run cross-validation."""
    score_cv = cross_val_score(model, X, y, cv=TimeSeriesSplit(n_splits=cross_validation_folds), scoring='f1_macro')
    # Return the cross-validation mean score and standard deviation.
    return score_cv.mean(), score_cv.std()

def train_predict_rolling(model, X_train, y_train, X_test, y_test, retrain_step_frequency):
    # Initialize a list to collect predictions from each rolling step.
    y_preds = []
    # Ensure the y train and test are Series type.
    y_train, y_test = Series(y_train), Series(y_test)
    # Combine training and testing features into a single dataframe for efficient indexing.
    X_full = concat([X_train, X_test], axis = 0)
    # Combine training and testing labels into a single series for efficient indexing.
    y_full = concat([y_train, y_test], axis = 0)
    # Identify the starting point for the rolling window based on the original training set size.
    initial_train_len = len(X_train)
    # Determine the total length of the combined dataset.
    total_len = len(X_full)
    # Iterate through the test data using the defined frequency to simulate periodic re-training.
    for i in range(initial_train_len, total_len, retrain_step_frequency):
        # Slice the features from the beginning of time up to the current index for training.
        X_curr_train = X_full.iloc[:i]
        # Slice the labels from the beginning of time up to the current index for training.
        y_curr_train = y_full.iloc[:i]
        # Calculate the end index for the prediction chunk, ensuring it does not exceed the data bounds.
        end_idx = min(i + retrain_step_frequency, total_len)
        # Define the feature chunk that the model will predict in this iteration.
        X_chunk = X_full.iloc[i:end_idx]
        # Fit the model using the current accumulated historical data.
        model.fit(X_curr_train, y_curr_train)
        # Execute predictions on the current chunk and extend the results to the master list.
        y_preds.extend(model.predict(X_chunk))
    # Calculate the score by comparing the rolling predictions against the true test labels.
    score = f1_score(y_true = y_test, y_pred = y_preds, average = 'macro')
    # Return the $score.
    return score

def saved_model_filename(name, symbols, random_state):
    """Define the output filename for saving the trained model to."""
    # Define today's date and time, replacing all spaces with underscores and removing colons in the time.
    timestamp = date_and_time(n_days_ago = 1, include_time = True).replace(' ', '_').replace(':', '')
    # Replace all spaces with underscores.
    name = name.replace(' ', '_')
    # Check if the user has opted to train for all symbols.
    if not symbols:
        # If so, then set the filename for the current learner.
        filename = f"{name}_{timestamp}_seed{random_state}.joblib"
    else:
        # Otherwise, if there are a list if symbols the learner is specific to, then join the symbols via underscores.
        symbols = '_'.join(symbols)
        # Add the symbols as part of the filename.
        filename = f"{name}_{symbols}_{timestamp}_seed{random_state}.joblib"
    # Use `pathlib` to determine the absolute path.
    saved_model = Path(filename).absolute()
    # Return the filename.
    return saved_model

def save(saved_model, model, score, save_threshold):
    """Save the model if its score on the test set was greater than or equal to the save_threshold. If the model was trained on the entire training set then save the model is the save_threshold was set to any number."""
    # Check if the saving of models is disabled.
    if save_threshold == -1: return None
    # If the $accuracy is greater than or equal to the set threshold, save the model to an output file in the current working directory. If the $score is Nonetype, then the the model is ready for production so go ahead as well.
    if (score is None) or (score >= save_threshold):
        # Save the model.
        dump(model, filename = saved_model)
        # Display a message to stdout.
        msg_info(f"The model has been saved to: {saved_model}")

############
### MAIN ###
############
def main(X_train, y_train, X_test, y_test, name, learners_yaml, symbols, random_state, perform_hyperparameter_optimization, perform_cross_validation, cross_validation_folds, retrain_step_frequency, save_threshold):
    # Load the machine learning configuration from the $learners_yaml file to populate the global dictionaries.
    load_learners(learners_yaml=learners_yaml)
    # Check if the $perform_hyperparameter_optimization toggle is enabled.
    if perform_hyperparameter_optimization is True:
        # Perform optimization to find the best model parameters using the $X_train and $y_train sets.
        model = hyperparameter_optimization(
            X_train=X_train,
            y_train=y_train,
            name=name,
            cross_validation_folds=cross_validation_folds,
            random_state=random_state
        )
    else:
        # Otherwise, retrieve the default model settings for the specified $name.
        model = learners[name]
    # Apply the $random_state and other universal settings to the $model.
    model = set_universal_params(model=model, random_state=random_state)
    # Initialize cross-validation variables to None to handle cases where validation is skipped.
    score_cv = None
    score_cv_stddev = None
    # If the $perform_cross_validation toggle is enabled, execute time-series validation on the training data.
    if perform_cross_validation is True:
        score_cv, score_cv_stddev = cross_validation(
            model=model,
            X=X_train,
            y=y_train,
            cross_validation_folds=cross_validation_folds
        )
    # Ensure both features and labels exist for the test set before proceeding to prediction.
    if (X_test is not None) and (y_test is not None):
        # Calculate the rolling performance score by incrementally updating the $model across the test period.
        score = train_predict_rolling(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            retrain_step_frequency=retrain_step_frequency
        )
    else:
        # Train the $model on the entire training set if no test data is available.
        model.fit(X_train, y_train)
        # Set the $score to None as no evaluation was performed.
        score = None
    # Generate a unique filename for the $model based on its $name, $symbols, and $random_state.
    saved_model = saved_model_filename(name=name, symbols=symbols, random_state=random_state)
    # Save the $model to disk if the rolling performance meets or exceeds the $save_threshold.
    save(saved_model=saved_model, model=model, score=score, save_threshold=save_threshold)
    # Return the final rolling $score, the cross-validation mean, and the standard deviation.
    return score, score_cv, score_cv_stddev