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

#################
### FUNCTIONS ###
#################
def variance_threshold(X, feature_names, is_enabled, threshold):
    """Perform VarianceThreshold feature selection to remove features with low variance."""
    # Check if VarianceThreshold was disabled.
    if not is_enabled: return X, feature_names
    # Initialize the function. Threshold is lowered to 0.0001 to support stationary/percentage features.
    selector = VarianceThreshold(threshold = threshold)
    # Fit and transform the training data.
    selector.fit_transform(X = X)
    # Define the selected features.
    selected_features = feature_names[selector.get_support()]
    # Update the training and testing data to only include the $selected_features identified by variance.
    return X[selected_features], selected_features

def select_k_best(X, y, feature_names, is_enabled, k):
    """Perform SelectKBest feature selection to select features based on mutual information."""
    # Check if SelectKBest was disabled.
    if not is_enabled: return X, feature_names
    # Initialize the function.
    selector = SelectKBest(mutual_info_classif, k = k)
    # Fit and transform the training data.
    selector.fit_transform(X = X, y = ravel(y))
    # Define the selected features.
    selected_features = selector.get_feature_names_out(input_features = feature_names)
    # Convert the ndarray to an Index. This keeps the output type consistent.
    selected_features = Index(selected_features)
    # Update the training and testing data to only include the $selected_features identified by variance.
    return X[selected_features], selected_features

def recursive_feature_elimination(X, y, feature_names, random_state, is_enabled, cross_validation_folds, min_features_to_select, scoring, step_size):
    """Perform Recursive Feature Elimination with Cross-Validation (RFECV) to select features."""
    # Check if RFECV was disabled.
    if not is_enabled: return X, feature_names
    # Convert the step size to type int if greater than 0, otherwise it will error out.
    if step_size > 0: step_size = int(step_size)
    # Initialize the function. 
    selector = RFECV(
        estimator = RandomForestClassifier(n_estimators = 250, max_depth = 3, min_samples_leaf = 10, random_state = random_state),
        cv = TimeSeriesSplit(n_splits = cross_validation_folds),
        min_features_to_select = min_features_to_select,
        n_jobs = -1,
        scoring = scoring,
        step = step_size
    )
    # Fit and transform the training data.
    selector.fit_transform(X = X, y = ravel(y))
    # Define the selected features.
    selected_features = feature_names[selector.support_]
    # Update the training and testing data to only include the $selected_features identified by variance.
    return X[selected_features], selected_features

############
### MAIN ###
############
def main(X, y, feature_names, random_state, configuration_ini):
    # Ensure the $feature_names are of type `Index` from pandas so boolean arrays can be applied.
    feature_names = Index(feature_names)
    # Use VarianceThreshold to remove features that are stagnant or nearly constant.
    X, selected_features = variance_threshold(
        X=X,
        feature_names=feature_names,
        is_enabled=configuration_ini.getboolean('FEATURE SELECTION', 'ENABLE_VARIANCE_THRESHOLD'),
        threshold=configuration_ini.getfloat('FEATURE SELECTION', 'VARIANCE_THRESHOLD_THRESHOLD')
    )
    # Perform feature selection using SelectKBest based on mutual information. k is set to 25 to allow more depth.
    X, selected_features = select_k_best(
        X=X,
        y=y,
        feature_names=X.columns,
        is_enabled=configuration_ini.getboolean('FEATURE SELECTION', 'ENABLE_SELECTKBEST'),
        k=configuration_ini.getint('FEATURE SELECTION', 'SELECTKBEST_K')
    )
    # Dynamically load the scoring metric.
    scoring_metric = dynamic_module_load(module_str=configuration_ini.get('ML', 'SCORING_METRIC'))
    # Load extra parameters defined in the configuration.ini.
    scoring_metric_params = loads(configuration_ini.get('ML', 'SCORING_METRIC_PARAMETERS'))
    # Perform feature selection using RFECV.
    X, selected_features = recursive_feature_elimination(
        X=X,
        y=y,
        feature_names=X.columns,
        random_state=random_state,
        is_enabled=configuration_ini.getboolean('FEATURE SELECTION', 'ENABLE_RFECV'),
        cross_validation_folds=configuration_ini.getint('ML', 'CROSS_VALIDATION_FOLDS'),
        min_features_to_select=configuration_ini.getint('FEATURE SELECTION', 'RFECV_MIN_FEATURES'),
        scoring=make_scorer(score_func = scoring_metric, **scoring_metric_params),
        step_size=configuration_ini.getfloat('FEATURE SELECTION', 'RFECV_STEP_SIZE')
    )
    # Return the $selected_features.
    return selected_features