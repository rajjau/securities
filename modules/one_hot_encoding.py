#!/usr/bin/env python
from pandas import concat,DataFrame
from sklearn.preprocessing import OneHotEncoder

############
### MAIN ###
############
def main(data, name, drop_and_replace = False):
    # Define the column to perform one-hot encoding (OHE) one.
    column = data[[name]]
    # Initialize the OHE object. Set sparse output to bool False to see the values rather than an object.
    ohe = OneHotEncoder(sparse_output = False)
    # Perform OHE with the specified $column name. Here, the column must be a DataFrame rather than a Series.
    column_ohe = ohe.fit_transform(DataFrame(column))
    # Create the DataFrame that contains the OHE columns, where descriptive names (based on the original values) are used.
    column_ohe = DataFrame(column_ohe, columns = ohe.get_feature_names_out([name]))
    # Define the column names to be returned later on.
    names_ohe = column_ohe.columns.to_list()
    # Check if the variable to drop the original column and add the new OHE columns was set to bool True.
    if drop_and_replace is True:
         # Remove the original $column.
        data = data.drop(labels = [name], axis = 1)
        # Now add the new OHE columns to $data.
        data = concat([data, column_ohe], axis = 1)
        # Return the full $data.
        return(data, names_ohe)
    else:
        # Otherwise, return only the OHE columns.
        return(column_ohe, names_ohe)