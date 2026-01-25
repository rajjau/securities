#!/usr/bin/env python3
from numpy import inf
from yaml import safe_load

######################
### CUSTOM MODULES ###
######################
from modules.dynamic_module_load import main as dynamic_module_load

#################
### FUNCTIONS ###
#################
def special_handling(optimization_params):
    # Convert the '.inf' string from YAML into a float infinity value.
    for param, values in optimization_params.items():
        # Itererate through each value in the list and replace '.inf' strings with float infinity.
        optimization_params[param] = [inf if (isinstance(val, str) and val.strip().lower() == '.inf') else val for val in values]
    # Return the updated optimization parameters.
    return optimization_params

############
### MAIN ###
############
def main(learners_yaml):
    # Define dictionaries to hold learners and their hyperparameters.
    learners = {}
    learners_hyperparameters = {}
    # Open the YAML file and load the 'LEARNERS' section into a configuration dictionary.
    with open(learners_yaml, 'r') as f: config = safe_load(f)['LEARNERS']
    # Iterate through each learner defined in the configuration.
    for name, info in config.items():
        # Import the module using the specified string.
        module_class = dynamic_module_load(module_str=info['class'])        
        # Extract the parameters from the configuration.
        params = info.get('params', {})
        # Instantiate the learner and store it in the global $learners dictionary.
        learners[name] = module_class(**params)
        # Extract the optimization parameters.  
        optimization_params = info.get('optimization', {})
        # Make any modifications needed for special handling of certain parameters.
        optimization_params = special_handling(optimization_params)
        # Store the optimization parameters in the $learners_hyperparameters dictionary.
        learners_hyperparameters[name] = optimization_params
    # Return the dictionaries.
    return learners, learners_hyperparameters