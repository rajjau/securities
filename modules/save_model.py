#!/usr/bin/env python
from joblib import dump

######################
### CUSTOM MODULES ###
######################
from modules.messages import msg_info

############
### MAIN ###
############
def main(saved_model, model, score, save_threshold, is_production):
    # Check if saving is enabled and threshold is met.
    if (save_threshold == -1) and (is_production is False): return None
    # Check if the $score is Nonetype, meaning the model is being trained on the entire training set, OR if it's greater than or equal to the specified save threshold, OR if the current run is a production run.
    if (score is None) or (score >= save_threshold) or (is_production is True):
        # Save the model object.
        dump(model, filename = saved_model)
        # Message to stdout.
        msg_info(f"SAVED: Trained model saved to: {saved_model}")