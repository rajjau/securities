#!/usr/bin/env python
from pathlib import Path

############
### MAIN ###
############
def main(directory, exit_on_error = True):
    # Check if the specified directory exists.
    if Path(directory).is_dir():
        # If so, then return bool True.
        return True
    else:
        # Otherwise, check if the exit on error variable was set to bool True.
        if exit_on_error is True:
            # If so, then raise an error.
            raise FileNotFoundError(f"Unable to locate directory: '{directory}")
        else:
            # Return bool False.
            return False