#!/usr/bin/env python3
from sklearn.pipeline import Pipeline

############
### MAIN ###
############
def main(model, random_state):
    # Check if the $model is a pipeline.
    if isinstance(model, Pipeline):
        # If so, then set target to the 'model' step within the pipeline.
        target = model.named_steps['model']
    else:
        # Otherwise, set target as the input $model.
        target = model
    try:
        # Set the random state if applicable.
        target.set_params(random_state = random_state)
    except ValueError:
        pass
    # Return the $model object, which can be a standalone model or a pipeline.
    return model