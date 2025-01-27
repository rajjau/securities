#!/usr/bin/env python
from numpy import ravel
from pandas import Index
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import TimeSeriesSplit
#-------------------#
#--- CLASSIFIERS ---#
#-------------------#
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV

#################
### FUNCTIONS ###
#################
def select_k_best(X, y, feature_names):
    # Initialize the function.
    selector = SelectKBest()
    # Fit and transform the training data.
    selector.fit_transform(X = X, y = ravel(y))
    # Define the selected features.
    selected_features = selector.get_feature_names_out(input_features = feature_names)
    # Return the selected list of features.
    return(selected_features)

def recursive_feature_elimination(X, y, feature_names):
    # Initialize the function.
    selector = RFECV(estimator = RandomForestClassifier(max_depth = 50, max_features = 'sqrt', n_estimators = 100), cv = TimeSeriesSplit(n_splits = 5), scoring = 'f1_macro')
    # Fit and transform the training data.
    selector.fit_transform(X = X, y = ravel(y))
    # Define the selected features.
    selected_features = feature_names[selector.support_]
    # Return the selected list of features.
    return(selected_features)

############
### MAIN ###
############
def main(X_train, X_test, y_train, feature_names):
    # Ensure the feature names are of type `Index` from pandas so boolean arrays can be applied.
    feature_names = Index(feature_names)
    # Recursive Feature Elimination.
    selected_features = recursive_feature_elimination(X = X_train, y = y_train, feature_names = feature_names)
    # Apply the selected features to the training and test sets.
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]
    # Return the modified X training and X test sets along with the list of selected feature column names.
    return(X_train, X_test, selected_features)