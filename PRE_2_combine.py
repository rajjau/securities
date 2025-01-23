#!/usr/bin/env python
from pathlib import Path
from sys import argv

######################
### CUSTOM MODULES ###
######################
from modules.combine_json import main as combine_json
from modules.messages import msg_info

#################
### FUNCTIONS ###
#################
def args():
    try:
        # Argument 1: Path to the directory that contains all data.
        directory = Path(argv[1]).absolute()
    except IndexError:
        raise IndexError('Argument 1: Path to the directory containing all saved JSON data.')
    try:
        # Argument 2: Output filename.
        filename_output = Path(argv[2]).absolute()
    except IndexError:
        raise IndexError(f"Argument 2: Output CSV filename for the combined data from: '{directory}'")
    # Return the user-defined variable(s).
    return(directory, filename_output)

############
### MAIN ###
############
def main(dir_data, filename_output):
    # Find all JSON files located within the $dir_data directory.
    filenames = sorted([filename for filename in Path(dir_data).iterdir() if filename.is_file() and filename.suffix == '.json'])
    # Combine the data for all filenames.
    combine_json(filenames = filenames, filename_output = filename_output)
    # Display informational message to stdout.
    msg_info(f"Done. All data has been written to the output file: '{filename_output}'")

#############
### START ###
#############
if __name__ == '__main__':
    # Obtain user-defined arguments
    [directory, filename_output] = args()
    # Start the script
    main(dir_data = directory, filename_output = filename_output)