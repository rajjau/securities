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
    # Place the seeds, scores for the current $learner, cross-validation scores, and cross-validation standard deviations into a single DataFrame.
    df_scores = DataFrame({
        'Seed': random_seeds,
        f"{learner} %": total_score,
        'CrossVal %': total_cv_score,
        'CrossValStddev %': total_cv_std
    })
    # Define columns that will be turned into decimals.
    col_scores = df_scores.columns[1:]
    try:
        # Convert all values in the specified columns from decimals to percentages.
        df_scores[col_scores] = (df_scores[col_scores]* 100).round(decimals = DECIMAL_PLACES)
    except TypeError:
        # If there are no scores (e.g., training was performed on the entire dataset), then return bool False.
        return False, False
    # Return the DataFrame and columns.
    return df_scores, col_scores

def calculate_average_scores(df_scores, col_scores):
    """Calculate and display the average score for the current learner across all random seeds. Includes the cross-validation score if applicable."""
    # Calculate the average for all of the columns containing scores.
    df_averages = df_scores[col_scores].mean(numeric_only = True).round(decimals = DECIMAL_PLACES)
    # Convert the Series above to a single-row DataFrame.
    df_averages = df_averages.to_frame().T
    # Add the 'AVERAGE' label to the first ('Seed') column.
    df_averages.insert(0, df_scores.columns[0], 'AVERAGE')
    # Add the average scores to the existing scores DataFrame as a new row. Return the modified DataFrame.
    return concat([df_scores, df_averages], ignore_index = True)

############
### MAIN ###
############
def main(learner, total_score, total_cv_score, total_cv_std, random_seeds, save_results_to_file, filename_output):
    # Convert the lists of scores to a single Pandas DataFrame.
    df_scores, col_scores = convert_to_dataframe(
        learner=learner,
        random_seeds=random_seeds,
        total_score=total_score,
        total_cv_score=total_cv_score,
        total_cv_std=total_cv_std
    )
    # If the conversion to a DataFrame 
    if df_scores is False:
        # Display warning message to user.
        msg_warn('There are no performance scores on the test set.')
        # Return Nonetype
        return None
    # Calculate the averages for the scores columns.
    df_scores = calculate_average_scores(df_scores=df_scores, col_scores=col_scores)
    # Check if the results should be saved to a file.
    if save_results_to_file:
        # Display message to stdout regarding output filename.
        msg_info(f"Saving results to: {filename_output}")
        # Save the scores to a CSV file.
        df_scores.to_csv(filename_output, index=False, quoting=1)
    # Display the scores to stdout.
    print(df_scores.to_string(index = False))