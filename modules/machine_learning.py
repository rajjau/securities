#!/usr/bin/env python
from joblib import dump
from numpy import arange
from pathlib import Path
from sklearn.exceptions import FitFailedWarning
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold, TimeSeriesSplit
from warnings import filterwarnings
#-------------------#
#--- CLASSIFIERS ---#
#-------------------#
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier

######################
### CUSTOM MODULES ###
######################
from modules.border import main as border
from modules.date_and_time import main as date_and_time
from modules.messages import msg_info,msg_warn

################
### SETTINGS ###
################
# Total number of folds to use in cross-validation.
CROSS_VALIDATION_FOLDS = 7
#
#------------------#
#-- RANDOM STATE --#
#------------------#
# Set the random state for all models.
RANDOM_MODEL = None
#
# Set the random state for the `train_test_split` function.
RANDOM_TRAIN_TEST_SPLIT = None
#
# The accuracy of the model must be greater than or equal to the set threshold in order for the model to be saved.
SAVE_THRESHOLD = 0.70
#
#--------------#
#-- WARNINGS --#
#--------------#
# Filter warnings that occurs when using GridSearchCV.
filterwarnings('ignore', category = FitFailedWarning)
filterwarnings('ignore', category = UserWarning)

###################
### CLASSIFIERS ###
###################
# Define a dictionary that contains all learners and their hyperparameters to use.
learners = {}
# learners['Bagging'] = BaggingClassifier(bootstrap = True, estimator = RandomForestClassifier(n_estimators = 250, random_state = RANDOM_MODEL), n_estimators = 10)
# learners['Logistic Regression'] = LogisticRegression()
# learners['Naive Bayes'] = GaussianNB()
# learners['Random Forest'] = RandomForestClassifier()
learners['Ridge'] = RidgeClassifier()

####################
### GRIDSEARCHCV ###
####################
# Define a dictionary that contains all learners to perform GridSearchCV with.
learners_gridsearch = {}

learners_gridsearch['Logistic Regression'] = {
    'C': arange(0.05, 1.05, 0.05),
    'class_weight': ['balanced', None],
    'l1_ratio': arange(0.05, 1.05, 0.05),
    'max_iter': [10000],
    'penalty': ['elasticnet', 'l1', 'l2', None],
    'random_state': [RANDOM_MODEL],
    'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
    'tol': [0.1, 0.01, 0.001, 0.0001, 0.00001]
}

learners_gridsearch['Random Forest'] = {
    'bootstrap': [True, False],
    'class_weight': ['balanced', 'balanced_subsample'],
    'criterion': ['entropy', 'gini', 'log_loss'],
    'max_depth': [50, 100, None],
    'max_features': ['log2', 'sqrt', None],
    'max_leaf_nodes': [10, 100, None],
    'n_estimators': [100, 250],
    'random_state': [RANDOM_MODEL]
}

learners_gridsearch['Ridge'] = {
    'alpha': arange(0.05, 1.05, 0.05),
    'class_weight': ['balanced', None],
    'copy_X': [True],
    'max_iter': [None],
    'random_state': [RANDOM_MODEL],
    'solver':['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'],
    'tol': [0.1, 0.01, 0.001, 0.0001, 0.00001],
}

#################
### FUNCTIONS ###
#################
def hyperparameter_optimization(X_train, y_train, name):
        # Display message to stdout.
        msg_info('Hyperparameter optimization is enabled.')
        try:
            # Obtain the parameters from the dictionary defined above.
            params = learners_gridsearch[name]
        except KeyError:
            # If no hyperparameters were defined for $name, then raise an error.
            msg_warn(f"The following learner name has not entries within the GridSearchCV dictionary defined in this script: '{name}'")
            # Return the defult learner instead.
            return(learners[name])
        # Define a stratified K-fold cross-validation object. Here, the $random_state is set so every time this script is run the same exact data is used, so all models can be properly compared
        # strat_k_fold = StratifiedKFold(n_splits = CROSS_VALIDATION_FOLDS, random_state = RANDOM_TRAIN_TEST_SPLIT, shuffle = True)
        # Define a time series split object.
        timeseries_k_fold = TimeSeriesSplit(n_splits = CROSS_VALIDATION_FOLDS)
        # Define the learner.
        model = learners[name]
        # Define the GridSearchCV object.
        model_gscv = GridSearchCV(estimator = model, param_grid = params, cv = timeseries_k_fold, scoring = 'f1_macro')
        # Fit the model with all combinations of hyperparameters to the training data.
        model_gscv.fit(X_train, y_train)
        # Identify the model with the best performance.
        best_model = model_gscv.best_estimator_
        # Return the model.
        return(best_model)

def cross_validation(model, X, y):
    # Run cross-validation.
    # score_cv = cross_val_score(model, X, y, cv = StratifiedKFold(n_splits = CROSS_VALIDATION_FOLDS, random_state = RANDOM_TRAIN_TEST_SPLIT), scoring = 'f1_macro')
    score_cv = cross_val_score(model, X, y, cv = TimeSeriesSplit(n_splits = CROSS_VALIDATION_FOLDS), scoring = 'f1_macro')
    # Return the cross-validation mean score and standard deviation.
    return(score_cv.mean(), score_cv.std())

def predict(model, X_test, y_test):
    # Make predictions on the test set.
    y_pred = model.predict(X_test)
    # Calculate the performance of the model.
    score = f1_score(y_true = y_test, y_pred = y_pred, average = 'macro')
    # Return the score.
    return(score)

def saved_model_filename(name, symbols):
    # Define today's date and time, replacing all spaces with underscores and removing colons in the time.
    timestamp = date_and_time(n_days_ago = 1, include_time = True).replace(' ', '_').replace(':', '')
    # Replace all spaces with underscores.
    name = name.replace(' ', '_')
    # Check if the user has opted to train for all symbols.
    if symbols is False:
        # If so, then set the filename for the current learner.
        filename = f"{name}_{timestamp}.joblib"
    else:
        # Otherwise, if there are a list if symbols the learner is specific to, then join the symbols via underscores.
        symbols = '_'.join(symbols)
        # Add the symbols as part of the filename.
        filename = f"{name}_{symbols}_{timestamp}.pkl"
    # Use `pathlib` to determine the absolute path.
    saved_model = Path(filename).absolute()
    # Return the filename.
    return(saved_model)

def save(saved_model, model, score):
    # If the $accuracy is greater than or equal to the set threshold, save the model to an output file in the current working directory.
    if score >= SAVE_THRESHOLD:
        # Save the model.
        dump(model, filename = saved_model)
        # Display a message to stdout.
        msg_info(f"The model has been saved to: {saved_model}")

############
### MAIN ###
############
def main(X_train, X_test, y_train, y_test, symbols, perform_hyperparameter_optimization = True, perform_cross_validation = True):
    # Iterate through each model within the learners dictionary.
    for name in learners:
        # Display the name of the current model to stdout.
        border(f"Machine Learning: {name}", '><')
        # Display a message to stdout about the threshold for saving a model.
        msg_info(f"The threshold has been set to {SAVE_THRESHOLD}. Any model with greater or equal accuracy will be saved.")
        # Check if the variable that controls hyperparameter optimization is enabled.
        if perform_hyperparameter_optimization is True:
            # If so, then use GridSearchCV to obtain the model with the best performing hyperparameters.
            model = hyperparameter_optimization(X_train = X_train, y_train = y_train, name = name)
        else:
            # Otherwise, define the current model using the $name.
            model = learners[name]
        # Set the cross-validated score and corresponding standard deviation to None by default. These variables are only useful if the variable to perform cross-validation is set to bool True.
        score_cv = None
        score_cv_stddev = None
        # If the parameter to perform cross-validaton is set to bool True, then do so.
        if perform_cross_validation is True:
            # Perform cross-validation and obtain the average score along with the standard deviation.
            [score_cv, score_cv_stddev] = cross_validation(model = model, X = X_train, y = y_train)
            # Display the average score and standard deviation to stdout.
            msg_info(f"The cross-validation score is {score_cv} with a standard deviation of {score_cv_stddev}.")
        # Fit the $model to the training data.
        model.fit(X_train, y_train)
        # Calculate the accuracy of the trained $model on the test dataset.
        score = predict(model = model, X_test = X_test, y_test = y_test)
        # Display the score to stdout.
        msg_info(f"Score: {score}")
        # Define the name of the file to save the model to, if applicable.
        saved_model = saved_model_filename(name = name, symbols = symbols)
        # If the $score is greater than or equal to the set threshold, save the model to an output file in the current working directory.
        save(saved_model = saved_model, model = model, score = score)