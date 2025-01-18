#!/usr/bin/env bash

################
### SETTINGS ###
################
# The company symbol(s)
SYMBOLS=('AAPL' 'MSFT')

# The year(s) to obtain data for
YEARS=(2023 2024);

#################
### FUNCTIONS ###
#################
add_leading_zero() {
	if [[ ${1} -lt 10 ]]; then
		echo "0${1}";
	else
		echo "${1}";
	fi;
};

#############
### START ###
#############
# Iterate through all symbols.
for symbol in ${SYMBOLS[@]}; do
	# Iterate through all years for the current $symbol.
	for year in ${YEARS[@]}; do
		# Iterate through all months in the year for the current $symbol.
		for month_ in {1..12}; do
			# Add a leading zero to the month number if it's less than 10. This ensures the API calls work properly when passing the dates.
			month=$(add_leading_zero ${month_});
			# Execute the specified Python script.
			python3 main.py "${symbol}" "${year}-${month}-01" "${year}-${month}-30";
			# We need to wait to ensure API limits aren't being hit.
			sleep 60;
		done;
	done;
done;
