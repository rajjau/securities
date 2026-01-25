#!/usr/bin/env python
from sklearn.base import BaseEstimator, TransformerMixin

######################
### CUSTOM MODULES ###
######################
from modules.feature_selection import main as feature_selection

############
### MAIN ###
############
class FeatureSelection(BaseEstimator, TransformerMixin):
    def __init__(self, configuration_ini, random_state = None):
        # Make input variables class global variables.
        self.configuration_ini = configuration_ini
        self.random_state = random_state

    def fit(self, X, y):
        # Perform feature selection and store the selected features.
        self.selected_features = feature_selection(
            X=X, 
            y=y,
            feature_names=X.columns, 
            random_state=self.random_state, 
            configuration_ini=self.configuration_ini
        )
        # Return self to allow method chaining.
        return self

    def transform(self, X):
        # Ensure we only return the columns selected during fitting.
        return X[self.selected_features]

    def get_feature_names_out(self, input_features=None):
        # Return the names of the selected features.
        return self.selected_features