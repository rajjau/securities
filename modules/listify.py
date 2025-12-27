#!/usr/bin/env python3

############
### MAIN ###
############
def main(lst):
    # Split the string via the comma delimiter to create a list.
    lst = lst.split(',')
    # Remove all extra whitespace.
    lst = [name.strip() for name in lst]
    # Remove empty entries.
    lst = [name for name in lst if name]
    # Return the Python list.
    return lst