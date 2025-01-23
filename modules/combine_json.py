#!/usr/bin/env python
from json import load
from pandas import concat,DataFrame
from pathlib import Path
from sys import argv

######################
### CUSTOM MODULES ###
######################
from modules.listify import main as listify
from modules.messages import msg_info,msg_warn
from modules.write_csv import main as write_csv

#################
### FUNCTIONS ###
#################
def args():
    try:
        # Argument 1: Comma separated list of files to combine.
        filenames = argv[1]
    except IndexError:
        # If argument 1 was not specified, then raise an error.
        raise IndexError('Argument 1: Comma separated list of files to combine.')
    # Convert the $filenames into a list from a string, and additionally, define the absolute path for every filename.
    filenames = [Path(entry).absolute() for entry in listify(filenames)]
    # Return the user-defined variable(s).
    return(filenames)

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
        return(False)
    # Convert the JSON dictionaries for each timepoint into a pandas DataFrame.
    contents = [DataFrame(entry, index = [0]) for entry in results]
    # Concatenate all DataFrames into one.
    contents = concat(contents, axis = 0).reset_index(drop = True)
    # Add the filename as another column.
    contents['f'] = filename.stem
    # Return the $contents DataFrame.
    return(contents)

############
### MAIN ###
############
def main(filenames, filename_output):
    # Display informational message to stdout.
    msg_info('Combining all data from every file. This may take some time, please be patient.')
    # Iterate through each file.
    for filename in filenames:
        # Convert its JSON contents to a pandas DataFrame.
        data = parse(filename = filename)
        # If the data was found, then write it to the output file. This read and write process is better for memory.
        if data is not False: write_csv(data = data, filename = filename_output)

#############
### START ###
#############
if __name__ == '__main__':
    # Obtain user-defined arguments
    filenames = args()
    # Start the script
    data = main(filenames = filenames)
    # Display the data to stdout.
    print(data.to_csv())