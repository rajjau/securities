#!/usr/bin/env python
from json import loads
from numpy import ravel
from pandas import Index
from sklearn.feature_selection import mutual_info_classif, RFECV, SelectKBest, VarianceThreshold
from sklearn.metrics import make_scorer
from sklearn.model_selection import TimeSeriesSplit
#-------------------#
#--- CLASSIFIERS ---#
#-------------------#
from sklearn.ensemble import RandomForestClassifier

######################
### CUSTOM MODULES ###
######################
from modules.dynamic_module_load import main as dynamic_module_load
from modules.messages import msg_info, msg_warn

#################
### FUNCTIONS ###
#################
def days_of_the_week(selected_features):
    # Search for any feature name that contains the specified string.
    matches = selected_features[selected_features.str.contains('day_of_week_')]
    # If there are no matches, then return the selected features as-is.
    if matches.empty: return(selected_features)
    # Display a warning message to user.
    msg_warn('One or more day-of-the-week features have been selected. Adding the other days of the week to avoid bias.')
    # Otherwise, define the prefix for each day of the week feature (e.g., 'lagged_day_of_week').
    prefixes = set(item.rsplit('_', 1)[0] for item in matches)
    # Define stock market days of the week.
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    # Define the missing days of the week.
    other_days_of_week = Index([f"{prefix}_{day}" for prefix in prefixes for day in days])
    # Return the union of the existing selected features Index and the one above. This will ensure there's no duplicates.
    return selected_features.union(other_days_of_week)

def apply_selected_features(X_train, X_test, selected_features):
    """Apply the selected features to both the training and testing datasets."""
    # If a specific day of the week is selected, then add the other days of the week.
    selected_features = days_of_the_week(selected_features=selected_features)
    # Define the current number of features.
    total_features = len(X_train.columns)
    # Update the training data to only include the selected features.
    X_train = X_train[selected_features]
    try:
        # Do the same for the test set.
        X_test = X_test[selected_features]
    except TypeError:
        # If the test set has not been defined, then do nothing.
        pass
    # Display number of selected features.
    print(f"\tSelected {len(selected_features)} of {total_features} features.")
    # Return the modified training and testing datasets.
    return X_train, X_test, selected_features

def variance_threshold(X_train, X_test, feature_names, threshold):
    """Perform VarianceThreshold feature selection to remove features with low variance."""
    # If the $threshold was set to -1, then VarianceThreshold was disabled.
    if threshold == -1: return X_train, X_test, feature_names
    # Display threshold to stdout.
    msg_info(f"VARIANCE THRESHOLD")
    # Initialize the function. Threshold is lowered to 0.0001 to support stationary/percentage features.
    selector = VarianceThreshold(threshold = threshold)
    # Fit and transform the training data.
    selector.fit_transform(X = X_train)
    # Define the selected features.
    selected_features = feature_names[selector.get_support()]
    # Update the training and testing data to only include the $selected_features identified by variance.
    return apply_selected_features(X_train=X_train, X_test=X_test, selected_features=selected_features)

def select_k_best(X_train, y_train, X_test, feature_names, k):
    """Perform SelectKBest feature selection to select features based on mutual information."""
    # If $k was set to -1, then SelectKBest was disabled.
    if k == -1: return X_train, X_test, feature_names
    # Display k to stdout.
    msg_info(f"SELECTKBEST")
    # Initialize the function.
    selector = SelectKBest(mutual_info_classif, k = k)
    # Fit and transform the training data.
    selector.fit_transform(X = X_train, y = ravel(y_train))
    # Define the selected features.
    selected_features = selector.get_feature_names_out(input_features = feature_names)
    # Convert the ndarray to an Index. This keeps the output type consistent.
    selected_features = Index(selected_features)
    # Update the training and testing data to only include the $selected_features identified by variance.
    return apply_selected_features(X_train=X_train, X_test=X_test, selected_features=selected_features)

def recursive_feature_elimination(X_train, y_train, X_test, feature_names, is_enabled, cross_validation_folds, scoring):
    """Perform Recursive Feature Elimination with Cross-Validation (RFECV) to select features."""
    # If the scoring was set to -1 (str) then RFECV was disabled.
    if not is_enabled: return X_train, X_test, feature_names
    # Display message to stdout.
    msg_info(f"RECURSIVE FEATURE ELIMINATION")
    # Initialize the function. 
    selector = RFECV(estimator = RandomForestClassifier(n_estimators = 25, max_depth = None, random_state = 0), cv = TimeSeriesSplit(n_splits = cross_validation_folds), n_jobs = -1, scoring = scoring)
    # Fit and transform the training data.
    selector.fit_transform(X = X_train, y = ravel(y_train))
    # Define the selected features.
    selected_features = feature_names[selector.support_]
    # Update the training and testing data to only include the $selected_features identified by variance.
    return apply_selected_features(X_train=X_train, X_test=X_test, selected_features=selected_features)

############
### MAIN ###
############
def main(X_train, y_train, X_test, feature_names, configuration_ini):
    # Ensure the $feature_names are of type `Index` from pandas so boolean arrays can be applied.
    feature_names = Index(feature_names)
    # Use VarianceThreshold to remove features that are stagnant or nearly constant.
    X_train, X_test, selected_features = variance_threshold(
        X_train=X_train,
        X_test=X_test,
        feature_names=feature_names,
        threshold=configuration_ini.getfloat('FEATURE SELECTION', 'VARIANCE_THRESHOLD')
    )
    # Perform feature selection using SelectKBest based on mutual information. k is set to 25 to allow more depth.
    X_train, X_test, selected_features = select_k_best(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        feature_names=X_train.columns,
        k=configuration_ini.getint('FEATURE SELECTION', 'SELECTKBEST')
    )
    # Dynamically load the scoring metric.
    scoring_metric = dynamic_module_load(module_str=configuration_ini.get('ML', 'SCORING_METRIC'))
    # Load extra parameters defined in the configuration.ini.
    scoring_metric_params = loads(configuration_ini.get('ML', 'SCORING_METRIC_PARAMETERS'))
    # Perform feature selection using RFECV.
    X_train, X_test, selected_features = recursive_feature_elimination(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        feature_names=X_train.columns,
        is_enabled=configuration_ini.getboolean('FEATURE SELECTION', 'RFECV'),
        cross_validation_folds=configuration_ini.getint('ML', 'CROSS_VALIDATION_FOLDS'),
        scoring=make_scorer(score_func = scoring_metric, **scoring_metric_params)
    )
    # Return the modified $X_train and $X_test sets along with the list of final $selected_features.
    return X_train, X_test, selected_features