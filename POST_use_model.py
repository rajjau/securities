#!/usr/bin/env python
from joblib import load
from pandas import DataFrame
from pathlib import Path
from sys import argv

############
### DATA ###
############


######################
### CUSTOM MODULES ###
######################
from modules.is_file import main as is_file

#################
### FUNCTIONS ###
#################
def args():
    try:
        # Argument 1: Path to the saved model.
        filename_model = Path(argv[1]).absolute()
    except IndexError:
        # If argument 1 was not specified, then raise an error.
        raise IndexError('Argument 1: Path to the saved model. It should be a joblib file.')
    # Return the user-defined variable(s).
    return filename_model

############
### MAIN ###
############
def main(filename_model):
    # Check if the specified model file exists.
    is_file(filename = filename_model)
    # Load the model in.
    with open(filename_model, 'rb') as f: model = load(f)
    print(model.feature_names_in_)

#############
### START ###
#############
if __name__ == '__main__':
    # Parse the command line arguments.
    filename_model = args()
    # Start the main function.
    main(filename_model = filename_model)