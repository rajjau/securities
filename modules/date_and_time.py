#!/usr/bin/env python
from datetime import datetime, timedelta

############
### MAIN ###
############
def main(n_days_ago = 0, include_time = False):
    # Full timestamp of the current date and time
    timestamp = datetime.now()
    # Check if the user wants to calculate the date that occured N days ago (e.g., 5, 10, 15 days ago)
    if n_days_ago > 0:
        # If so, then subtract the number of days from the current date
        date = timestamp - timedelta(days = n_days_ago)
    # Check if the time was set to be included or not.
    if include_time is False:
        # Define only the YYYY-MM-DD.
        date = date.date()
    else:
        # Define the date in YYYY-MM-DD and time (24-hour) format.
        date = date.strftime('%Y-%m-%d %R')
    # Return the $date
    return(date)