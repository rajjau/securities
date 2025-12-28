#!/usr/bin/env python
from argparse import ArgumentParser
from pathlib import Path

######################
### CUSTOM MODULES ###
######################
from modules.combine_json import main as combine_json
from modules.messages import msg_info

#################
### FUNCTIONS ###
#################
def args():
    """Parse and return command-line arguments."""
    # Create an ArgumentParser object.
    parser = ArgumentParser(description='Combine all JSON files into a single CSV.')
    # Add arguments.
    parser.add_argument('directory', type=Path, help='Path to the directory containing all saved JSON data.')
    parser.add_argument('filename_output', type=Path, help='Output CSV filename for the combined data.')
    # Parse the arguments.
    args = parser.parse_args()
    # Return the arguments.
    return args.directory.absolute(), args.filename_output.absolute()

############
### MAIN ###
############
def main(dir_data, filename_output):
    # Find all JSON files located within the $dir_data directory.
    filenames = sorted([filename for filename in Path(dir_data).iterdir() if filename.is_file() and filename.suffix == '.json'])
    # Combine the data for all filenames.
    combine_json(filenames=filenames, filename_output=filename_output)
    # Display informational message to stdout.
    msg_info(f"Done. All data has been written to the output file: '{filename_output}'")

#############
### START ###
#############
if __name__ == '__main__':
    # Obtain user-defined arguments.
    directory, filename_output = args()
    # Start the script.
    main(directory=directory, filename_output=filename_output)