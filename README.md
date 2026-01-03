## Workflow
1. **Initialization:** Validates the presence of `configuration.ini` and `learners.yaml`.
   
2. **Data Importing:** Imports raw CSV data, filters by symbols, and handles caching for faster subsequent runs.

   A. **Preprocessing:** Verifies the existence of the cache directory and creates it if it does not already exist.

   B. **Parsing:** If no cache exists, the script reads the source Parquet file and filters the rows to include only the user-specified ticker symbols (column `T`).

   C. **Caching:** Saves the filtered dataset into a local cache file for future runs.

   D. **Dynamic Feature Mapping:** Detects if a wildcard (`*`) is used for feature selection, and if so, automatically identifies and assigns all available columns (excluding the target labels) as the feature set.

3. **Preprocessing:** Split the input data into training and testing sets.

   A. **Sorting:** Sorts the dataset by the time column (`t`), to ensure the model learns from the past to predict on more recent data.

   B. **Time-Based Holdout Split:** Implements a time-series split where the final  days (defined by `t_d`) are used as a testing holdout while all preceding data is used for training.
   
   C. **One-Hot Encoding:** Performs One-Hot Encoding on columns defined within `configuration.ini`.
   
   D. **Cleanup:** Ensure the training and testing datasets contain only numeric data (types `int` and `float`) and removes all `NaN` values.
   
   E. **Normalization:** Fits a scaler (options include **Standard**, **MinMax**, or **Robust**) _only_ on the training data. This fitted scaler is then applied to the test data to prevent "data snooping" or look-ahead bias.
  
4. **Feature Selection:** (Optional) Performs one or more of the following feature selection methods. Variance threshold is almost always performed if enabled.

   A. **Variance Threshold:** Filters out features with low variance (threshold set at 0.05). This eliminates "stagnant" columns that do not change enough.

   B. **SelectKBest:** Uses Mutual Information toselects the top N features that share the most information with the labels, helping to reduce noise.
   
   C. **Recursive Elimination (RFECV):** Performed using the Decision Tree estimator. It uses a `TimeSeriesSplit` cross-validation strategy to iteratively prune the least important features based on the Macro F1-score.


6. **Machine Learning:** Iterates through selected learners (defined in `learners.yaml`) to train a model. This process is performed over 25 random seeds and the final results are an average over all random seeds.

   A. **Load:** Parses the `learners.yaml` file to  import scikit-learn models with their specified hyperparameters. Each model class includes a separate section for hyperparameters to tune during optimization.
   
   B. **TimeSeries:** Executes `RandomizedSearchCV` using `TimeSeriesSplit` to tune hyperparameters. This ensures that the optimization process respects the chronological order of financial data, preventing the model from "cheating" by seeing future information during the tuning phase.
   
   C. **Cross-Validation:** Performs internal validation on the training set using chronological folds. This provides a mean F1-score and standard deviation.
   
   D. **Walk-Forward Rolling Retraining:** Model is periodically retrained as more recent data arrives. It supports:
      * **Retrain Step Frequency:** How often the model updates its parameters.
      * **Sliding Window:** Training on a fixed-size history.

7. **Results Aggregation:** Calculates mean scores and standard deviations across all seeds and exports the final metrics to CSV.

## Feature Engineering
This is the workflow used to engineer features. The Polygon.io service is used to obtain daily data for all tickers and data is obtained from 01/01/2023 to present.

### 1. Data Initialization & Cleaning

* **Sorting:** Orders data by Ticker (`T`) and Timestamp (`t`) to ensure temporal integrity.
* **Logical Flags:** Benchmarks the current session (e.g., `open_to_close` price action).

### 2. Temporal & Cyclical Encoding

* **Date Normalization:** Extracts days, months, quarters, and month-end flags.
* **Cyclical Mapping:** Uses Sine and Cosine transformations for "Day of Week" and "Day of Month" to help machine learning models understand time-based loops (e.g., Monday being close to Friday).

### 3. Price Action & Momentum

* **Overnight Gaps:** Calculates the percentage difference between the previous close and current open.
* **Candlestick Anatomy:** Quantifies body size, upper/lower shadows, and relative body strength.
* **Interaction Features:** Combines price volatility with volume (e.g., `spread_volume`) to identify high-conviction moves.

### 4. Historical Lagging (Memory)

* **Multi-Day Lags:** Shifts core OHLCV data back up to 15 days (configurable in `configuration.ini`).
* **Derived Lags:** Calculates historical daily returns and high-low spreads for every lag interval.

### 5. Technical Indicators

Computes window-based statistics across multiple horizons (5, 10, 15, 20, 50 days):

* **Moving Averages:** Distance from simple moving average (SMA) and exponential moving average (EMA) to identify trends.
* **Volatility:** Rolling Standard Deviation of returns.
* **Momentum:** Rate of Change (ROC) and Normalized Rolling Ranges.
