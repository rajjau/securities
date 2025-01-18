#!/usr/bin/env python

############
### MAIN ###
############
def main(message, border_char = '='):
    # Create the top and bottom borders for the $message.
    borders = border_char * (len(message) + 5)
    # Display the top and bottom $borders with the $message to stdout.
    print(f"\n{borders}\n{message}\n{borders}")