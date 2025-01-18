#!/usr/bin/env python
from pathlib import Path

############
### MAIN ###
############
def main(data, filename):
    # Check if the output file already exists.
    if Path(filename).is_file() is True:
        # If so, then the header will not need to be written.
        header = False
        # Set the mode to append rather than write.
        mode = 'a'
    else:
        # If the output file will be created, then add the header.
        header = True
        # Set the mode to write.
        mode = 'w'
    # Save the combined $data to the output file.
    data.to_csv(filename, header = header, index = False, mode = mode, quoting = 1)