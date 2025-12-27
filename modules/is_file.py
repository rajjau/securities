#!/usr/bin/env python
from pathlib import Path

############
### MAIN ###
############
def main(filename, exit_on_error = True):
    # Check if the specified file exists.
    if Path(filename).is_file():
        # If so, then return bool True.
        return True
    else:
        # Otherwise, check if the exit on error variable was set to bool True.
        if exit_on_error is True:
            # If so, then raise an error.
            raise FileNotFoundError(f"Unable to locate file: '{filename}")
        else:
            # Return bool False.
            return False