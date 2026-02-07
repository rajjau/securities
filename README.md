# Goal

Develop a machine learning (ML) model to predict whether the closing price of a given stock will be greater or less than the opening price.

# Workflow

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