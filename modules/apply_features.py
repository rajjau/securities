#!/usr/bin/env python
from pandas import Index

######################
### CUSTOM MODULES ###
######################
from modules.messages import msg_info, msg_warn

#################
### FUNCTIONS ###
#################
def days_of_the_week(selected_features, verbose):
    # Search for any feature name that contains the specified string.
    matches = selected_features[selected_features.str.contains('day_of_week_')]
    # If there are no matches, then return the selected features as-is.
    if matches.empty: return(selected_features)
    # Display a warning message to user.
    if verbose is True: msg_warn('One or more day-of-the-week features have been selected. Adding the other days of the week to avoid bias.')
    # Otherwise, define the prefix for each day of the week feature (e.g., 'lagged_day_of_week').
    prefixes = set(item.rsplit('_', 1)[0] for item in matches)
    # Define stock market days of the week.
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    # Define the missing days of the week.
    other_days_of_week = Index([f"{prefix}_{day}" for prefix in prefixes for day in days])
    # Return the union of the existing selected features Index and the one above. This will ensure there's no duplicates.
    return selected_features.union(other_days_of_week)

############
### MAIN ###
############
def main(X_train, X_test, selected_features, verbose = False):
    """Apply the selected features to both the training and testing datasets."""
    # If a specific day of the week is selected, then add the other days of the week.
    selected_features = days_of_the_week(selected_features=selected_features, verbose=False)
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
    if verbose is True: msg_info(f"Selected {len(selected_features)} of {total_features} features.")
    # Return the modified training and testing datasets.
    return X_train, X_test, selected_features