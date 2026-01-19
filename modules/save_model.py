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
    # Check if the $score is Nonetype, meaning the model is being trained on the entire training set, OR if it's greater than or equal to the specified save threshold.
    score_meets_threshold = (score is None) or (score >= save_threshold)
    # Check if either: 1) the current run is for production AND the score meets thereshold or 2) the score is not negative AND it meets the threshold.
    if ((is_production is True) and (score_meets_threshold is True)) or ((save_threshold >= 0) and (score_meets_threshold is True)):
        # Save the model object.
        dump(model, filename = saved_model)
        # Message to stdout.
        msg_info(f"SAVED: Trained model saved to: {saved_model}")
        