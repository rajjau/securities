#!/usr/bin/env python
from pandas import concat, DataFrame

######################
### CUSTOM MODULES ###
######################
from modules.add_features import main as add_features
from modules.border import main as border
from modules.feature_selection import main as feature_selection
from modules.messages import msg_info

############
### MAIN ###
############
def main(X_train, y_train, X_test, columns_x, random_seeds, configuration_ini):
    # Create a border to denote a process.
    border('FEATURE SELECTION', border_char='><')
    # Check if feature selection was enabled. If not, then return the training and testing sets as-is.
    if not configuration_ini.getboolean('GENERAL', 'PERFORM_FEATURE_SELECTION'): return X_train, X_test
    # List to hold all selected feature sets from every random seed.
    all_selected_features = []
    # Iterate through each random seed.
    for seed in random_seeds:
        # Message to stdout.
        msg_info(f"Seed {seed}")
        # Perform feature selection.
        selected_features_seed, _, _ = feature_selection(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            feature_names=columns_x,
            random_state=seed,
            configuration_ini=configuration_ini
        )
        # Add the selected features from the current seed to the master list. This puts a value of 1 (int) for every feature.
        all_selected_features.append(DataFrame(1, index = selected_features_seed, columns = [f"seed_{seed}"]))
    # Concatenate all features column-wise, where each column represents a seed. If a feature was selected in one seed but not another, it's value in the seed it's missing in will be 0.
    selected_features = concat(all_selected_features, axis = 1).fillna(0).astype(int)
    # Calculate the mean for each feature. A value of 1.0 means that feature was selected for every seed. Higher is better (more robust and stable). Additionally, round to a single decimal place. This allows for easy filtering.
    selected_features['rate'] = selected_features.mean(axis = 1).round(decimals = 1)
    # Keep only features that are equal to or above the threshold specified within the configuration.ini.
    selected_features = selected_features[selected_features['rate'] >= configuration_ini.getfloat('FEATURE SELECTION', 'SELECTED_FEATURES_THRESHOLD')]
    # Apply the selected features to the training and test sets.
    X_train, X_test, _ = add_features(X_train=X_train, X_test=X_test, selected_features=selected_features.index, verbose=True)
    # Diplay message to stdout regarding selected features.
    msg_info(f"Selected features: {selected_features.index.values}")
    msg_info(f"Kept {len(selected_features)} out of {len(columns_x)} total features.")
    # Return the training and testing data after feature selection.
    return X_train, X_test