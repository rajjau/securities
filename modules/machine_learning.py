#!/usr/bin/env python
from joblib import dump
from numpy import arange, inf
from pathlib import Path
from pandas import concat, Series
from scipy.linalg import LinAlgWarning
from sklearn.exceptions import FitFailedWarning
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from warnings import filterwarnings

#-------------------#
#--- CLASSIFIERS ---#
#-------------------#
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.svm import SVC

######################
### CUSTOM MODULES ###
######################
from modules.date_and_time import main as date_and_time
from modules.messages import msg_info,msg_warn

################
### SETTINGS ###
################
# Maximum number of iterations.
MAX_ITER = 99999999

################
### WARNINGS ###
################
# Filter warnings that occurs when using hyperparameter optimization.
filterwarnings('ignore', category = FitFailedWarning)
filterwarnings('ignore', category = LinAlgWarning)
filterwarnings('ignore', category = UserWarning)

###################
### CLASSIFIERS ###
###################
# Define a dictionary that contains all learners and their hyperparameters to use.
learners = {}
learners['Bagging'] = BaggingClassifier(n_jobs = -1)
learners['Decision Tree'] = DecisionTreeClassifier()
learners['K Nearest Neighbor'] = KNeighborsClassifier(n_jobs = -1)
learners['Logistic Regression'] = LogisticRegression(max_iter = MAX_ITER, n_jobs = None)
learners['Random Forest'] = RandomForestClassifier(n_jobs = -1)
learners['Ridge CV'] = RidgeClassifierCV()
learners['SVC'] = SVC()

###################################
### HYPERPARAMETER OPTIMIZATION ###
###################################
# Define a dictionary that contains all learners to perform hyperparameter optimization with.
learners_hyperparameters = {}

learners_hyperparameters['Bagging'] = {
    'bootstrap': [True, False],
    'bootstrap_features': [True, False],
    'estimator': {'Decision Tree': learners['Decision Tree'],
                  'Logistic Regression': learners['Logistic Regression']
                  },
    'n_estimators': [10]
}

learners_hyperparameters['Decision Tree'] = {
    'class_weight': ['balanced', None],
    'criterion': ['entropy', 'gini', 'log_loss'],
    'max_depth': [5, 10, 20, 50, 100, None],
    'max_features': ['log2', 'sqrt', None],
    'max_leaf_nodes': [5, 10, 20, 50, 100, None],
    'min_samples_leaf': [1, 2, 5, 10],
    'min_samples_split': [2, 5, 10, 20],
    'splitter': ['best', 'random']
}

learners_hyperparameters['K Nearest Neighbor'] = {
    'algorithm': ['ball_tree', 'brute', 'kd_tree'],
    'n_neighbors': range(1, 25),
    'weights': ['distance', 'uniform', None]
}

learners_hyperparameters['Logistic Regression'] = {
    'C': [arange(0.01, 1.01, 0.01), inf],
    'class_weight': ['balanced', None],
    'l1_ratio': arange(0, 1.05, 0.05),
    'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
    'tol': [0.1, 0.01, 0.001, 0.0001, 0.00001]
}

learners_hyperparameters['Random Forest'] = {
    'bootstrap': [True, False],
    'class_weight': ['balanced', 'balanced_subsample'],
    'criterion': ['entropy', 'gini', 'log_loss'],
    'max_depth': [10, 50, 100, None],
    'max_features': ['log2', 'sqrt', None],
    'max_leaf_nodes': [10, 100, None],
    'n_estimators': [100, 200, 250]
}

learners_hyperparameters['Ridge CV'] = {
    'alphas': arange(0.01, 1.01, 0.01),
    'class_weight': ['balanced', None],
    'scoring': ['f1_macro']
}

learners_hyperparameters['SVC'] = {
    'C': arange(0.01, 1.01, 0.01),
    'class_weight': ['balanced', None],
    'coef0': arange(0.0, 10.01, 0.01),
    'decision_function_shape': ['ovo', 'ovr'],
    'degree': arange(1, 10, 1),
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'shrinking': [True, False],
    'tol': [0.1, 0.01, 0.001, 0.0001, 0.00001]
}

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
        params = learners_hyperparameters[name]
    except KeyError:
        # If no hyperparameters were defined for $name, then raise an error.
        msg_warn(f"The following learner name has not entries within the hyperparameter dictionary defined in this script: '{name}'")
        # Return the defult learner instead.
        return learners[name]
    # Check if the current model is the Bagging, which is a meta-estimator.
    if name == 'Bagging':
        # Define all estimators.
        estimators = params['estimator']
        # Define a list that will replace the current dictionary of estimators. This will include only the estimators with the best parameters that are chosen from hyperparameter optimization.
        best_estimators = [
            hyperparameter_optimization(
                X_train=X_train, 
                y_train=y_train, 
                name=estimator, 
                cross_validation_folds=cross_validation_folds, 
                random_state=random_state
            ) for estimator in estimators
        ]
        # Replace the old dictionary with the new list of chosen models.
        learners_hyperparameters[name]['estimator'] = best_estimators
    # Define a time series split object.
    timeseries_k_fold = TimeSeriesSplit(n_splits = cross_validation_folds)
    # Define the learner.
    model = learners[name]
    # Set universal parameters.
    model = set_universal_params(model=model, random_state=random_state)
    # Define the GridSearchCV object.
    # search = GridSearchCV(cv=timeseries_k_fold, estimator=model, param_grid=params, scoring='f1_macro')
    search = RandomizedSearchCV(cv=timeseries_k_fold, estimator=model, n_jobs=-1, param_distributions=params, random_state=random_state, scoring='f1_macro')
    # Fit the model with all combinations of hyperparameters to the training data.
    search.fit(X_train, y_train)
    # Identify the model with the best performance.
    best_model = search.best_estimator_
    # Return the model.
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
def main(X_train, X_test, y_train, y_test, name, symbols, perform_cross_validation, cross_validation_folds, perform_hyperparameter_optimization, random_state, save_threshold, retrain_step_frequency):
    # Check if hyperparameter optimization is enabled in the configuration.
    if perform_hyperparameter_optimization is True:
        # Perform optimization to find the best model parameters using the initial training set.
        model = hyperparameter_optimization(X_train=X_train,
                                            y_train=y_train,
                                            name=name,
                                            cross_validation_folds=cross_validation_folds,
                                            random_state=random_state
                                            )
    else:
        # Otherwise, retrieve the default model settings for the specified learner.
        model = learners[name]
    # Apply the current random seed and other universal settings to the model.
    model = set_universal_params(model=model, random_state=random_state)
    # Initialize cross-validation scores to None to handle cases where validation is skipped.
    score_cv = None
    # Initialize cross-validation standard deviation to None.
    score_cv_stddev = None
    # If enabled, execute time-series cross-validation on the training data to measure historical stability.
    if perform_cross_validation is True:
        score_cv, score_cv_stddev = cross_validation(model=model,
                                                     X=X_train,
                                                     y=y_train,
                                                     cross_validation_folds=cross_validation_folds
                                                     )
    # Ensure both features and labels exist for the test set before proceeding to prediction.
    if (X_test is not None) and (y_test is not None):
        # Calculate the rolling performance score by incrementally updating the model across the test period.
        score = train_predict_rolling(model=model,
                                      X_train=X_train,
                                      y_train=y_train,
                                      X_test=X_test,
                                      y_test=y_test,
                                      retrain_step_frequency=retrain_step_frequency
                                      )
    else:
        # Train the model on the entire training set.
        model.fit(X_train, y_train)
        # Set the score to None if no test data was provided for evaluation.
        score = None
    # Generate a unique filename for the model based on its name, symbols, and seed.
    saved_model = saved_model_filename(name=name, symbols=symbols, random_state=random_state)
    # Save the model to disk if its rolling performance score meets or exceeds the required threshold.
    save(saved_model=saved_model, model=model, score=score, save_threshold=save_threshold)
    # Return the final rolling score, the cross-validation mean, and the cross-validation standard deviation.
    return score, score_cv, score_cv_stddev