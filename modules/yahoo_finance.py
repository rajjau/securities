#!/usr/bin/env python
from pathlib import Path
from datetime import datetime
from urllib.request import urlretrieve

######################
### CUSTOM MODULES ###
######################
from modules.messages import msg_info

#################
### FUNCTIONS ###
#################
def time_from_unix(timestamp):
    # Convert the UNIX $timestamp into YYYY-MM-DD HH:MM format
    yyy_mm_dd_time = datetime.fromtimestamp(timestamp)
    # Return the converted time, but only keep the YYYY-MM-DD
    return(yyy_mm_dd_time.date())

############
### MAIN ###
############
def main(names, timestamp_today, timestamp_previous, dir_data):
    # Define the Yahoo! Finance URL and use the timestamps to specify the interval
    url = f"https://query1.finance.yahoo.com/v7/finance/download/STOCK_NAME?period1={timestamp_previous}&period2={timestamp_today}&interval=1d&events=history&includeAdjustedClose=true"
    # Define a dictionary that will contain the stock names as keys and the values will be their corresponding filenames
    data = {}
    # Iterate through each stock name within the $names list
    for stock in names:
        # Convert the current $stock to uppercase
        stock = stock.upper()
        # Replace the placeholder string with the name of the current $stock. It doesn't need to be uppercase
        url_stock = url.replace('STOCK_NAME', stock)
        # Define the full path to the output CSV file
        filename = Path(dir_data, f"RAW_{stock}_{timestamp_previous}-{timestamp_today}.csv")
        # Add the current $stock to the dictionary, where its value will be the full path to its saved data $filename
        data[stock] = filename
        # Check if the $filename already exists
        if filename.is_file():
            # If so, then display a message to stdout
            msg_info(f"SKIP: The data for '{stock}' has already been downloaded for the specified period: {time_from_unix(timestamp_previous)} -> {time_from_unix(timestamp_today)}")
            # Continue to the next $stock
            continue
        # Download the data for the current $stock to the $filename
        urlretrieve(url_stock, filename)
    # Return the $data dictionary
    return(data)