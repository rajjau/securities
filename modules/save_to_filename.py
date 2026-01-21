#!/usr/bin/env python
from pathlib import Path

######################
### CUSTOM MODULES ###
######################
from modules.date_and_time import main as date_and_time

############
### MAIN ###
############
def main(dir_data_saved, name, tickers, extension, random_state = False, timestamp = False):
    """Define the filename of the output file"""
    # If not passed, then define today's date and time for the filename.
    if timestamp is False: timestamp = date_and_time(n_days_ago = 0, include_time = True).replace(' ', '_').replace(':', '')
    # Prepare the name for the filename (e.g., 'DecisionTree', 'FittedScaler', 'SelectedFeatures', etc.).
    name = name.replace(' ', '_')
    # Build the filename using tickers.
    filename = Path(f"{name}_{tickers}_{timestamp}.{extension}")
    # If the random state was defined, add it to the filename.
    if random_state: filename = f"{filename.stem}_seed{random_state}.{extension}"
    # Use `pathlib` to return the absolute path.
    return Path(dir_data_saved, filename).absolute()