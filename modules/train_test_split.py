#!/usr/bin/env python
from numpy import ravel,unique
from pandas import DataFrame
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

######################
### CUSTOM MODULES ###
######################
from modules.messages import msg_info
from modules.one_hot_encoding import main as one_hot_encoding

#############################
### NORMALIZATION METHODS ###
#############################
NORMALIZE_METHODS = {
    'MinMaxScaler': MinMaxScaler,
    'RobustScaler': RobustScaler,
    'StandardScaler': StandardScaler
}

#################
### FUNCTIONS ###
#################
def holdout(data, days):
    # Check if there are more than 0 days specified for holdout.
    if days > 0:
        # Define the final X dates that will be kept as the test set.
        holdout = unique(data['t_d'].values)[-days:]
        # Define the training dataset based on the opposite of the holdout.
        data_train = data[~data['t_d'].isin(holdout)]
        # Define the testing dataset based on the holdout for the final X days.
        data_test = data[data['t_d'].isin(holdout)]
    else:
        # If not, then return $data as the training set and Nonetype for the testing set.
        return data, None
    # Return the training and test datasets.
    return data_train, data_test

def split_X_y(data, columns_x, columns_y):
    # If $data is Nonetype then return Nonetype for both $X and $y.
    if data is None: return None, None
    # Define the feature columns as a DataFrame.
    X = data[columns_x]
    # Define the label column(s), and use the `ravel` command to ensure there are no warnings during training later on.
    y = ravel(data[columns_y])
    # Return the feature and label columns for the $data.
    return X, y

def one_hot_encode_data(data, columns_x, columns_one_hot_encoding):
    # If $data is Nonetype then return Nonetype for both $data and $columns_x.
    if data is None: return None, None
    # Check if the global variable has been defined for a list of columns to perform one-hot encoding (OHE) to.
    if columns_one_hot_encoding:
        # Iterate through each column name in the list.
        for name in columns_one_hot_encoding:
            # Message to stdout.
            msg_info(f"One-hot encoding column: '{name}'")
            # Perform OHE for the current $column, remove the original column, and add the new OHE columns to $data.
            data, names_ohe = one_hot_encoding(data=data, name=name, drop_and_replace=True)
            # Define the list of feature columns.
            columns_x = sorted([entry for entry in columns_x if entry != name] + names_ohe)
    # Return $data and $columns_X. This works if one-hot encoding was not performed as well.
    return data, columns_x

def apply_normalize(X, columns_x, normalize_method, scaler = None):
    # If $X is Nonetype then return Nonetype for both $X and $scaler (which has already been set to Nonetype).
    if X is None: return None, scaler
    # Check if the $scaler has not been fitted already.
    if not scaler:
        # Initialize the scaler.
        scaler = NORMALIZE_METHODS[normalize_method]()
        # Fit the scaler to $X.
        scaler = scaler.fit(X)
    # Transform $X using the fitted scaler.
    X = scaler.transform(X)
    # Convert $X back into a DataFrame.
    X = DataFrame(X, columns = columns_x)
    # Return $X
    return X, scaler

############
### MAIN ###
############
### MAIN ###
def main(data, columns_x, columns_y, columns_one_hot_encoding, holdout_days, normalize_X, normalize_method):
    # Ensure the $data is sorted from past to present based on the time column.
    data = data.sort_values(by = 't', ascending = True)
    # Define the last $holdout_days as a holdout split. This will divide the data into training and test datasets.
    data_train, data_test = holdout(data, days=holdout_days)
    # Split the training datasets into feature and label columns using $columns_x and $columns_y.
    X_train, y_train = split_X_y(data=data_train, columns_x=columns_x, columns_y=columns_y)
    # Perform one-hot-encoding on the training features based on the $columns_one_hot_encoding list.
    X_train, columns_x_ohe = one_hot_encode_data(data=X_train, columns_x=columns_x, columns_one_hot_encoding=columns_one_hot_encoding)
    # Split the test datasets into feature and label columns.
    X_test, y_test = split_X_y(data=data_test, columns_x=columns_x, columns_y=columns_y)
    # Check if the test sets have been defined. If $holdout_days is 0 then these variables will be None.
    if (X_test is not None) and (y_test is not None):
        # Perform one-hot-encoding on the testing features.
        X_test, _ = one_hot_encode_data(data=X_test, columns_x=columns_x, columns_one_hot_encoding=columns_one_hot_encoding)
        # Reindex the testing features to match the training features to ensure column alignment.
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
        # Check if any NaNs exist in the testing set after the reindexing process.
        if X_test.isnull().values.any():
            # Identify the boolean mask of rows that do not contain any NaNs.
            mask = X_test.notnull().all(axis=1)
            # Keep only the rows in the testing features that do not contain NaNs.
            X_test = X_test[mask]
            # Align the testing labels to match the rows kept in the testing features.
            y_test = y_test[mask.values]
    # Check if the $normalize_X toggle is set to True.
    if normalize_X is True:
        # If so, then normalize the training set and obtain the fitted scaler.
        X_train, scaler = apply_normalize(X=X_train, columns_x=columns_x_ohe, normalize_method=normalize_method, scaler=None)
        # Use the fitted scaler from the training set to normalize the testing features.
        X_test, _ = apply_normalize(X=X_test, columns_x=columns_x_ohe, normalize_method=normalize_method, scaler=scaler)
    # Return the training and testing features, labels, and the updated $columns_x_ohe list.
    return X_train, X_test, y_train, y_test, columns_x_ohe