#!/usr/bin/env python
from numpy import ravel
from pandas import Index
from sklearn.feature_selection import mutual_info_classif, RFECV, SelectKBest, VarianceThreshold
from sklearn.model_selection import TimeSeriesSplit
#-------------------#
#--- CLASSIFIERS ---#
#-------------------#
from sklearn.tree import DecisionTreeClassifier

#################
### FUNCTIONS ###
#################
def apply_selected_features(X_train, X_test, selected_features):
    """Apply the selected features to both the training and testing datasets."""
    # Update the training data to only include the selected features.
    X_train = X_train[selected_features]
    try:
        # Do the same for the test set.
        X_test = X_test[selected_features]
    except TypeError:
        # If the test set has not been defined, then do nothing.
        pass
    # Return the modified training and testing datasets.
    return X_train, X_test

def variance_threshold(X, feature_names):
    """Perform VarianceThreshold feature selection to remove features with low variance."""
    # Initialize the function. Threshold is lowered to 0.0001 to support stationary/percentage features.
    selector = VarianceThreshold(threshold = 0.0001)
    # Fit and transform the training data.
    selector.fit_transform(X = X)
    # Define the selected features.
    selected_features = feature_names[selector.get_support()]
    # Return the selected list of features.
    return selected_features

def select_k_best(X, y, feature_names, k):
    """Perform SelectKBest feature selection to select features based on mutual information."""
    # Initialize the function.
    selector = SelectKBest(mutual_info_classif, k = k)
    # Fit and transform the training data.
    selector.fit_transform(X = X, y = ravel(y))
    # Define the selected features.
    selected_features = selector.get_feature_names_out(input_features = feature_names)
    # Convert the ndarray to an Index. This keeps the output type consistent.
    selected_features = Index(selected_features)
    # Return the selected list of features.
    return selected_features

def recursive_feature_elimination(X, y, feature_names):
    """Perform Recursive Feature Elimination with Cross-Validation (RFECV) to select features."""
    # Initialize the function.
    selector = RFECV(estimator = DecisionTreeClassifier(), cv = TimeSeriesSplit(n_splits = 5), n_jobs = -1, scoring = 'f1_macro')
    # Fit and transform the training data.
    selector.fit_transform(X = X, y = ravel(y))
    # Define the selected features.
    selected_features = feature_names[selector.support_]
    # Return the selected list of features.
    return selected_features

############
### MAIN ###
############
def main(X_train, y_train, X_test, feature_names):
    # Ensure the $feature_names are of type `Index` from pandas so boolean arrays can be applied.
    feature_names = Index(feature_names)
    # Use VarianceThreshold to remove features that are stagnant or nearly constant.
    selected_features = variance_threshold(X=X_train, feature_names=feature_names)
    # Update the training and testing data to only include the $selected_features identified by variance.
    X_train, X_test = apply_selected_features(X_train, X_test, selected_features)
    # Perform feature selection using SelectKBest based on mutual information. k is set to 25 to allow more depth.
    selected_features = select_k_best(X=X_train, y=y_train, feature_names=X_train.columns, k=25)
    # Update the training and testing data to only include the final $selected_features.
    X_train, X_test = apply_selected_features(X_train, X_test, selected_features)
    # Return the modified $X_train and $X_test sets along with the list of final $selected_features.
    return X_train, X_test, selected_features