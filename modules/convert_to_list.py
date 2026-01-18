#!/usr/bin/env python

############
### MAIN ###
############
def main(string, delimiter):
    """Split a string representation of a list into an actual list based on the provided $delimiter."""
    return [item.strip() for item in string.split(delimiter)]