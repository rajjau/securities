#!/usr/bin/env python3
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit

######################
### CUSTOM MODULES ###
######################
from modules.set_universal_learner_params import main as set_universal_learner_params
from modules.messages import msg_error, msg_info, msg_warn

################
### SETTINGS ###
################
# Set the name of the Bagging learner used within the learners.yaml file.
BAGGING_LEARNER_NAME = 'Bagging'

# Set the total number of iterations to use.
N_ITER = 50

# Set the random state for hyperparameter optimization.
RANDOM_STATE = 0

#################
### FUNCTIONS ###
#################
def handle_bagging(X_train, y_train, pipeline, learners, learners_hyperparameters, cross_validation_folds, scoring):
    try:
        # Retrieve the list of potential base estimators using the pipeline prefix.
        estimator = learners_hyperparameters[BAGGING_LEARNER_NAME]['model__estimator'][0]
    except IndexError:
        msg_error('Unable to locate any estimators. Please check the learners YAML file.')
    # Message to stdout.
    msg_info(f"Optimizing base estimator '{estimator}' for the Bagging ensemble.")
    # Build a base pipeline for the current estimator. Start with cloning the existing $pipeline.
    base_pipeline = clone(pipeline)
    # Obtain the index of the 'model' step within the pipeline.
    idx = [i for i, item in enumerate(base_pipeline.steps) if item[0] == 'model'][0]
    # Replace Bagging with the current estimator as the model.
    base_pipeline.steps[idx] = ('model', learners[estimator]) 
    #------------------------------#
    # Call the main function to find the best hyperparameters for the current $estimator_name.
    base_pipeline = main(
        X_train=X_train,
        y_train=y_train,
        pipeline=base_pipeline,
        name=estimator,
        learners=learners,
        learners_hyperparameters=learners_hyperparameters,
        cross_validation_folds=cross_validation_folds,
        scoring=scoring
    )
    # Obtain the optimized model for the current $estimator_name.
    optimized_estimator = base_pipeline.steps[idx][1]
    # Return the optimized base estimator.
    return optimized_estimator

def param_search_random(timeseries_k_fold, pipeline, params, scoring):
    # Define the RandomizedSearchCV object using the TimeSeriesSplit object and the full pipeline.
    search = RandomizedSearchCV(
        cv = timeseries_k_fold,
        estimator = pipeline,
        n_iter = N_ITER,
        n_jobs = -1,
        param_distributions = params,
        random_state = RANDOM_STATE,
        scoring = scoring
    )
    # Return the search object.
    return search

def param_search_grid(timeseries_k_fold, pipeline, params, scoring):
    # Define the GridSearch object.
    search = GridSearchCV(
        cv = timeseries_k_fold,
        estimator = pipeline,
        n_jobs = -1,
        param_grid = params,
        scoring = scoring
    )
    # Return the search object.
    return search

############
### MAIN ###
############
def main(X_train, y_train, pipeline, name, learners, learners_hyperparameters, cross_validation_folds, scoring):
    try:
        # Obtain the parameters from the dictionary defined above.
        params = learners_hyperparameters[name].copy()
    except KeyError:
        # If no hyperparameters were defined for $name, then raise an error.
        msg_warn(f"The following learner name has no entries within the hyperparameter dictionary: '{name}'")
        # Return the default learner instead.
        return pipeline
    # Check if the current model is Bagging to handle its sub-estimators.
    if name == BAGGING_LEARNER_NAME:
        # Create a separate training set for optimizing the Bagging estimators. Keep the number of splits small since we only have the training set to split.
        outer_cv = TimeSeriesSplit(n_splits = 2)
        # Use next and reversed to efficiently grab only the most recent (last) split.
        estimator_idx, val_idx = next(reversed(list(outer_cv.split(X_train))))
        # Optimize the base estimators for Bagging. Use the training indexes for the estimator.
        optimized_estimator = handle_bagging(
            X_train=X_train.iloc[estimator_idx],
            y_train=y_train.iloc[estimator_idx],
            pipeline=pipeline,
            learners=learners,
            learners_hyperparameters=learners_hyperparameters,
            cross_validation_folds=cross_validation_folds,
            scoring=scoring
        )
        # Obtain the index of the model within the pipeline process.
        idx = [i for i, item in enumerate(pipeline.steps) if item[0] == 'model'][0]
        # Add the hyperparameter optimized base estimator to Bagging.
        pipeline.steps[idx] = ('model', learners[name].set_params(estimator = optimized_estimator))
        # Return the Bagging classifer fitted on the validation set indexes defined above. It's important to return here and not fit again because that will overfit.
        return pipeline.fit(X=X_train.iloc[val_idx], y=y_train.iloc[val_idx])
    # Define a time series split object to prevent future data from leaking into the training folds during optimization.
    timeseries_k_fold = TimeSeriesSplit(n_splits = cross_validation_folds)
    # Set universal parameters on the pipeline.
    pipeline = set_universal_learner_params(model=pipeline, random_state=RANDOM_STATE)
    #----------------------------#
    #--- Randomized Search CV ---#
    #----------------------------#
    # search = param_search_random(
    #     timeseries_k_fold=timeseries_k_fold,
    #     pipeline=pipeline,
    #     params=params,
    #     scoring=scoring
    # )
    #--------------------------#
    #--- Grid Search CV -------#
    #--------------------------#
    search = param_search_grid(
        timeseries_k_fold=timeseries_k_fold,
        pipeline=pipeline,
        params=params,
        scoring=scoring
    )
    # Fit the model to the training data.
    search.fit(X_train, y_train)
    # Identify the model with the best performance.
    best_estimator = search.best_estimator_
    # Return the $best_estimator.
    return best_estimator