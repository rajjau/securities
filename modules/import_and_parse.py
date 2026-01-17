#!/usr/bin/env python
from pandas import read_parquet

######################
### CUSTOM MODULES ###
######################
from modules.border import main as border

############
### MAIN ###
############
def main(filename, columns_x, columns_y):
    # Create a border to denote a process.
    border('IMPORT DATA', border_char='><')
    # Read the data from $filename and import it as a DataFrame.
    data = read_parquet(path = filename)
    # If a wildcard was passed to define the feature set columns, then use all columns except for $columns_y.
    if columns_x[0] == '*': columns_x = data.columns.difference(columns_y).to_list()
    # Return the processed $data and the updated $columns_x.
    return data, columns_x