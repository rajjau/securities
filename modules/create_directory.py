#!/usr/bin/env python
from pathlib import Path

############
### MAIN ###
############
def main(directory, fail_if_exists = False):
    try:
        # Create the specified directory.
        Path(directory).mkdir()
    except FileExistsError:
        # Check if the parameter to error out if the directory already exists has been set to bool True.
        if fail_if_exists is True:
            # If so, then raise an error
            raise FileExistsError(f"The specified directory already exists: '{directory}'")
    # Return bool True.
    return True
