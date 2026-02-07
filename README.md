# Goal

Develop a machine learning (ML) model to predict whether the closing price of a given stock will be greater or less than the opening price.

# Workflow

1. Download ticker data from Massive.com.
2. Engineer features:
    
    - Time and date features (i.e., day of week, day of month, month, is start of month, is end of month, quarter of year)
    - Overnight features (i.e., overnight gap between today's opening price and yesterday's closing price)
    - Lagged features (adding all existing features from the previous day)

3. Save the data for the user-specified tickers (e.g., 'SPY', 'QQQI') into a Parquet file.
4. Perform ML.

    A. Load the data for the specific ticker from its Parquet file (e.g., 'SPY.parquet').
    
    B. Split the data into training and testing datasets.

    C. Iterate through each learner specified in the configuration file. For a given learner:

    - Perform hyperparameter optimization (if enabled) on a single random seed.

    - Iterate through each random seed specified in the configuration INI file and train the learner. A rolling window walk-forward validation method is used. Here, the `sliding_window_size` variable determines the size of the window, while the `retrain_step_frequency` tells the script how many days are used for prediction before retraining.

        - For example, a dataset contains 1,000 total days where the initial training data ranges from days 1- 100 and the initial test dataset ranges from days 101-200. The `sliding_window_size` = 100 and `retrain_step_frequency` = 10. The model will first be trained on days 1-100 and predicts on days 101-110. Then, it will retrain on days 11-110 and predicts on days 111-120. It continues by retraining on days 21-120 and predicting on days 121-130. This process continues until the prediction range hits day 1,000.

        - The Matthews Correlation Coefficient (MCC) score is used to assess each model's performance within every sliding window. Additionally, if cross-validation (CV) is enabled, then scikit-learn's TimeSeriesSplit method is used.

    - After the MCC score and CV (if applicable) are obtained from every random seed, the final step is to pass this data to the VotingClassifier.

        - Low quality random seed models are filtered out if they contain negative peformance (including within CV). Additionally, a performance gap is calculated: | MCC score - CV score | for each random seed (lower is better). Models whose performance is greater than the median performance gap are filtered as well. The models are sorted based on their performance gap from best to worst, and the top K models are kept.

        - The remaining models are used to create the VotingClassifier, and training is once again performed using the rolling window walk-forward validation method. Note that the final sliding window, the most recent days, are used to train the output model.

        - The MCC scoring metric evaluates the VotingClassifier model's performance across all sliding windows. Even though the final trained model is based on the most recent days, its past performance provides a larger sample to evaluate robustness.
    
    D. Repeat the above steps for the next learner if applicable.

---
---

## configuration.ini

An INI file is used to set options and parameters throughout the workflow. This includes the path to a local directory to store data, the column name that defines the label for ML, and what tickers to train on.

## learners.yaml

A YAML file is used to define the ML classifiers that can be selected within the **configuration.ini**. An example is shown below:

```
LEARNERS:
  Random Forest:
    class: sklearn.ensemble.RandomForestClassifier
    params:
      bootstrap: True
      n_jobs: -1
      n_estimators: 250
    optimization:
      class_weight: [balanced, balanced_subsample]
      criterion: [entropy, gini, log_loss]
      max_depth: [10, 20, 50]
      max_features: [log2, sqrt]
      max_leaf_nodes: [10, 20, 30, 50, 100, 200]
      min_samples_leaf: [1, 2, 5, 10, 20]`
```
The `params` section above is for defining hyperparameters.

The `optimization` section above is for hyperparameter optimization using either GridSearchCV or RandomizedSearchCV.

## Data

All data is obtained from Massive.com, formally Polygon.io, in JSON format. This data is for a single day and contains all tickers within the United States stock market.

1. The data for each weekday is downloaded as JSON. If a given day is a holiday, then the file will not contain any ticker data.

2. All JSON data are combined into a single CSV, which will contain all supported tickers for every weekday. Days with no ticker data are filtered out.

3. Tickers (defined within the **configuration.ini**) are processed where features are engineered and the processed ticker data is saved to a Parquet file (e.g., 'SPY.parquet'). See the following section for details about feature engineering.

### Feature Engineering

By default, the data for a given ticker on a specific day contains (sorted in alphabetical order):

- closing price (`c`)
- daily high (`h`)
- daily low (`l`)
- number of transactions for the current day (`n`)
- opening price (`o`)
- Unix timestamp for the start of the day (`t`)
- daily volume (`v`)
- volume weighted average price (`vw`)

Other features