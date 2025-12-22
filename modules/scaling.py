#!/usr/bin/env python3
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

############
### FUNCTIONS ###
############
def UseMinMaxScaler(X, min_value = 0, max_value = 1):
    # Define the MinMaxScaler with the specified range.
    scaler = MinMaxScaler(feature_range = (min_value, max_value))
    # Fit the scaler to the training and testing data, then transform the values.
    X_scaled = scaler.fit_transform(X)
    # Convert the scaled data back to a DataFrame using the same columns and index as the input DataFrame.
    X_scaled = DataFrame(X_scaled, columns = X.columns, index = X.index)
    # Return the scaled data.
    return(X_scaled)

def UseRobustScaler(X, min_value = 0, max_value = 1):
    # Define the RobustScaler with the specified range.
    scaler = RobustScaler()
    # Fit the scaler to the training and testing data, then transform the values.
    X_scaled = scaler.fit_transform(X)
    # Convert the scaled data back to a DataFrame using the same columns and index as the input DataFrame.
    X_scaled = DataFrame(X_scaled, columns = X.columns, index = X.index)
    # Return the scaled data.
    return(X_scaled)

def UseStandardScaler(X):
    # Define the Standard Scaler.
    scaler = StandardScaler()
    # Fit the scaler to the training and testing data, then transform the values.
    X_scaled = scaler.fit_transform(X)
    # Convert the scaled data back to a DataFrame using the same columns and index as the input DataFrame.
    X_scaled = DataFrame(X_scaled, columns = X.columns, index = X.index)
    # Return the scaled data.
    return(X_scaled)
