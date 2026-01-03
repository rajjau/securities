#!/usr/bin/env python
from importlib import import_module

############
### MAIN ###
############
def main(module_str):
    # Split the class string so the learner can be imported.
    module_path, class_name = module_str.rsplit('.', 1)
    # Dynamically import the required module.
    module = import_module(module_path)
    # Retrieve the model class from the imported module.
    module_class = getattr(module, class_name)
    # Return the loaded module.
    return module_class