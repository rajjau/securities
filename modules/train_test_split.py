#!/usr/bin/env python
from numpy import ravel,unique
from pandas import DataFrame
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

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
    # Define the final X dates that will be kept as the test set.
    holdout = unique(data['t_d'].values)[-days:]
    # Define the training dataset based on the opposite of the holdout.
    data_train = data[~data['t_d'].isin(holdout)]
    # Define the testing dataset based on the holdout for the final X days.
    data_test = data[data['t_d'].isin(holdout)]
    # Return the training and test datasets.
    return(data_train, data_test)

def split_X_y(data, columns_x, columns_y, normalize_X, normalize_method):
    # Define the feature columns as a DataFrame.
    X = data[columns_x]
    # Check if the variable to standardize the data was set to bool True.
    if normalize_X is True:
        # Initialize the scaler.
        scaler = NORMALIZE_METHODS[normalize_method]()
        # Fit and transform the features data.
        X = scaler.fit_transform(X)
        # Convert $X back into a DataFrame.
        X = DataFrame(X, columns = columns_x)
    # Define the label column(s), and use the `ravel` command to ensure there are no warnings during training later on.
    y = ravel(data[columns_y])
    # Return the feature and label columns for the $data.
    return(X, y)

############
### MAIN ###
def main(data, columns_x, columns_y, holdout_days, normalize_X, normalize_method):
    # Ensure the $data is sorted from past to present.
    data = data.sort_values(by = 't', ascending = True)
    # Define the last X days as a holdout. This will split the data into training and test datasets.
    [data_train, data_test] = holdout(data, days = holdout_days)
    # Split the training datasets into feature and label columns.
    [X_train, y_train] = split_X_y(data = data_train, columns_x = columns_x, columns_y = columns_y, normalize_X = normalize_X, normalize_method = normalize_method)
    # Split the test datasets into feature and label columns.
    [X_test, y_test] = split_X_y(data = data_test, columns_x = columns_x, columns_y = columns_y, normalize_X = normalize_X, normalize_method = normalize_method)
    # Return the training and testing data.
    return(X_train, X_test, y_train, y_test)




