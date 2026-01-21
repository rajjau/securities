#!/usr/bin/env python

################
### SETTINGS ###
################
# Number of extra characters to add to the border length.
BUFFER = 5

############
### MAIN ###
############
def main(message, border_char = '='):
    # Total length of the message plus the buffer.
    total = len(message) + BUFFER
    # Create the top and bottom borders for the $message.
    borders = border_char * total
    # Ensure the borders match the length of the message.
    borders = borders[:total]
    # Display the top and bottom $borders with the $message to stdout.
    print(f"\n{borders}\n{message}\n{borders}")