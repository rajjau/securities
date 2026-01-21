#!/usr/bin/env python
from pathlib import Path
from time import sleep
from urllib.request import urlretrieve
from urllib.error import HTTPError

######################
### CUSTOM MODULES ###
######################
from modules.messages import msg_info,msg_warn

##################
### POLYGON.IO ###
##################
# Define the Polygon.io API Key.
API_KEY = '_1xf0pxcVM3Hzu4lX7BFj2rvrkSh_HEf'

# Define the Polygon.io URL.
URL = f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/DATE?adjusted=true&apikey={API_KEY}" # Entire market

#################
### FUNCTIONS ###
#################
def define_url(date):
    # Replace the starting and ending dates with their corresponding variables from the $dates tuple.
    url_ = URL.replace('DATE', date)
    # Return the modified $url that's ready to be used.
    return url_

def download(url, filename, stdout_ticker_date):
    try:
        # Download the data for the current $ticker to the $filename.
        urlretrieve(url, filename)
    except HTTPError:
        # Display a warning message to stdout.
        msg_warn(f"HTTP Error for {stdout_ticker_date}. Reasons include API limit(s) or closed markets.")
        # Return bool False.
        return False
    # Display an informational message to stdout.
    msg_info(f"Successfully downloaded the data for {stdout_ticker_date}.")
    # Return bool True.
    return True

############
### MAIN ###
############
def main(dir_data, dates):
    # Define a list of all output filenames.
    filenames_output = []
    # Iterate through each $date in the $dates tuple.
    for date in dates:
        # Edit the URL to replace all placeholders with the current $date, $ticker, etc.
        url_ = define_url(date = date)
        # Define the full path to the output JSON file.
        filename = Path(dir_data, f"market_{date}.json")
        # Use the filename to create an informational message that will be used in messages to stdout later.
        stdout_ticker_date = filename.stem.replace('_', ' ')
        # Check if the $filename already exists.
        if filename.is_file():
            # If so, then display a message to stdout.
            msg_info(f"SKIP: The data for {stdout_ticker_date} has already been downloaded.")
            # Continue to the next $ticker.
            continue
        else:
            # Download the data.
            is_success = download(url = url_, filename = filename, stdout_ticker_date = stdout_ticker_date)
            # Add the output filename to the list if the data was able to be downloaded.
            if is_success is True: filenames_output.append(filename)
            # If there is more than one date in the $dates tuple, wait the specified time to avoid API flood errors.
            if len(dates) > 1: sleep(15)
    # Return the list of filenames that contain newly downloaded data.
    return filenames_output