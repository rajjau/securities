#!/usr/bin/env python
from argparse import ArgumentParser
from json import load
from pandas import concat, DataFrame
from pathlib import Path

######################
### CUSTOM MODULES ###
######################
from modules.convert_to_list import main as convert_to_list
from modules.messages import msg_info, msg_warn
from modules.write_csv import main as write_csv

#################
### FUNCTIONS ###
#################
def args():
    """Parse and return command-line arguments."""
    # Create an ArgumentParser object.
    parser = ArgumentParser(description='Combine the data from all JSON files into a single CSV file.')
    # Add arguments.
    parser.add_argument('filenames', type=str, help='Comma separated list of files to combine.')
    parser.add_argument('filename_output', type=Path, help='Comma separated list of files to combine.')
    # Parse the arguments.
    args = parser.parse_args()
    # Convert the $filenames into a list from a string, and additionally, define the absolute path for every filename.
    filenames = [Path(entry).absolute() for entry in convert_to_list(string=args.filenames, delimiter=',')]
    # Return the arguments.
    return filenames, args.filename_output.absolute()

def parse(filename):
    # Display current filename.
    msg_info(f"Parsing: '{filename}'")
    # Open $filename as read-only and obtain its JSON contents.
    with open(filename, 'r') as f: raw = load(f)
    try:
        # The JSON contains meta-data as well as the actual data we queried under the specified key.
        results = raw['results']
    except KeyError:
        # If no results were found in the JSON file, then display a warning message to stdout.
        msg_warn(f"No results found for {filename}, skipping.")
        # Return bool False.
        return False
    # Convert the JSON dictionaries for each timepoint into a pandas DataFrame.
    data = [DataFrame(entry, index = [0]) for entry in results]
    # Concatenate all DataFrames into one.
    data = concat(data, axis = 0).reset_index(drop = True)
    # Add the filename as another column.
    data['f'] = filename.stem
    # Remove duplicate rows, if any exist.
    data = data.drop_duplicates(keep = 'first').reset_index(drop = True)
    # Return the DataFrame.
    return data

############
### MAIN ###
############
def main(filenames, filename_output):
    # Display informational message to stdout.
    msg_info('Combining all data from every file. This may take some time, please be patient.')
    # Iterate through each file.
    for filename in filenames:
        # Convert its JSON contents to a pandas DataFrame.
        data = parse(filename=filename)
        # If the data was found, then write it to the output file. This read and write process is better for memory.
        if data is not False: write_csv(data = data, filename = filename_output)

#############
### START ###
#############
if __name__ == '__main__':
    # Obtain user-defined arguments
    filenames, filename_output = args()
    # Start the script
    data = main(filenames=filenames, filename_output=filename_output)
    # Display the data to stdout.
    print(data.to_csv())