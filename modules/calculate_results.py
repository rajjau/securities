#!/usr/bin/env python
from pandas import concat, DataFrame

#####################
### CUSTOM MODULES ###
######################
from modules.messages import msg_info, msg_warn

################
### SETTINGS ###
################
DECIMAL_PLACES = 1

#################
### FUNCTIONS ###
#################
def convert_to_dataframe(learner, random_seeds, total_score, total_cv_score, total_cv_std):
    """Convert the lists of scores into a single Pandas DataFrame."""
    # Define header.
    header = ['Seed', learner, 'CrossVal', 'CrossValStddev']
    # Concatenate the seeds, scores for the current $learner, cross-validation scores, and cross-validation standard deviations into a single DataFrame.
    df_scores = concat([DataFrame(random_seeds), DataFrame(total_score), DataFrame(total_cv_score), DataFrame(total_cv_std)], axis = 1)
    # Add the $header to the DataFrame.
    df_scores.columns = header
    # Convert all values to percentages from decimals.
    for col in header[1:]: df_scores[col] = (df_scores[col]* 100).round(decimals = DECIMAL_PLACES)
    # Return the DataFrame.
    return df_scores

def rank_scores(df_scores, learner):
    """Rank scores using a composite score and tier system tailored for financial prediction."""
    # Create a copy of the DataFrame to work with.
    df = df_scores.copy()
    # Compute a generalization score by subtracting the cross-validation standard deviation from the cross-validation score.
    df['GeneralizationScore'] = df['CrossVal'] - df['CrossValStddev']
    # Compute a composite score that prioritizes cross-validation performance, stability, and then test-set performance.
    df['CompositeScore'] = (0.70 * df['CrossVal']) + (0.20 * df['GeneralizationScore']) + (0.10 * df[learner])
    # Create a tier flag to ensure higher tiers are sorted above lower ones. Tier 1: CrossVal >= 50. Tier 0: CrossVal < 50.
    df['Tier'] = (df['CrossVal'] >= 50).astype(int)
    # Sort by tier, then by composite score, and finally by generalization score.
    df = df.sort_values(by=['Tier', 'CompositeScore', 'GeneralizationScore'], ascending=[False, False, False])
    # Apply the index from the new $df DataFrame to the original DataFrame. This ensures that we keep the original columns, just reordered.
    df_scores = df_scores.loc[df.index].reset_index(drop = True)
    # Return the sorted DataFrame.
    return df_scores

def calculate_average_scores(df_scores):
    """Calculate and display the average score for the current learner across all random seeds. Includes the cross-validation score if applicable."""
    # Obtain the header.
    header = df_scores.columns
    # Calculate average scores, skipping the first 'Seed' column.
    averages = ['AVERAGE'] + [df_scores[col].mean().round(decimals = DECIMAL_PLACES) for col in header[1:]]
    # Convert the list to a Pandas DataFrame.
    averages = DataFrame([averages], columns = header)
    # Add the average scores to the existing scores DataFrame.
    df_scores = concat([df_scores, averages], ignore_index = True)
    # Return the DataFrame.
    return df_scores

############
### MAIN ###
############
def main(learner, random_seeds, total_score, total_cv_score, total_cv_std, save_results_to_file, output_filename):
    # Check if the total score list is empty.
    if all(total_score) is False:
        # Display warning message to user.
        msg_warn('There are no performance scores on the test set.')
        # Return Nonetype
        return None
    # Convert the lists of scores to a single Pandas DataFrame.
    df_scores = convert_to_dataframe(
                    learner=learner,
                    random_seeds=random_seeds,
                    total_score=total_score,
                    total_cv_score=total_cv_score,
                    total_cv_std=total_cv_std
                    )
    # Sort the DataFrame from best performance to worst.
    df_scores = rank_scores(df_scores=df_scores, learner=learner)
    # Calculate the averages for the scores columns.
    df_scores = calculate_average_scores(df_scores=df_scores)
    # Check if the results should be saved to a file.
    if save_results_to_file:
        # Display message to stdout regarding output filename.
        msg_info(f"Saving results to: {output_filename}")
        # Save the scores to a CSV file.
        df_scores.to_csv(output_filename, index=False, quoting=1)
    # Display the scores to stdout.
    print(df_scores.to_string(index = False))