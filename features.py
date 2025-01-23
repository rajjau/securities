#!/usr/bin/env python
from pandas import read_csv
from pathlib import Path
from sys import argv

######################
### CUSTOM MODULES ###
######################
from modules.add_features import main as add_features
from modules.is_file import main as is_file

#################
### FUNCTIONS ###
#################
def args():
    try:
        filename = Path(argv[1]).absolute()
    except IndexError:
        raise IndexError('Argument 1: CSV containing the combined stock data to add features to.')
    try:
        # Argument 2: Output filename.
        filename_output = Path(argv[2]).absolute()
    except IndexError:
        raise IndexError('Argument 2 : CSV output filename for data with features.')
    # Return the user-defined variable(s).
    return(filename, filename_output)

############
### MAIN ###
############
def main(filename, filename_output):
    # Ensure the specified file exists.
    is_file(filename = filename, exit_on_error = True)
    # Read the data.
    data = read_csv(filepath_or_buffer = filename)
    # Remove all duplicate rows.
    data = data.drop_duplicates(keep = 'first')
    # Remove all rows that contain any NaNs.
    data = data.dropna(axis = 0)
    # Sort the DataFrame by the symbol name and timestamp. This groups stocks and ensures the data for a given stock is in order from earliest to most recent.
    data = data.sort_values(by = ['T', 't'])
    # Add various extra features to the $data.
    data = add_features(data)
    # Sort the columns in alphabetical order.
    data = data.sort_index(axis = 1)
    # Again, remove all rows that contain any NaNs after adding features.
    data = data.dropna(axis = 0)
    # # Save the $data with new features to the output file.
    data.to_csv(filename_output, header = True, index = False, quoting = 1)

#############
### START ###
#############
if __name__ == '__main__':
    [filename, filename_output] = args()
    main(filename = filename, filename_output = filename_output)