## Workflow

1. **Initialization:** Validates the presence of `configuration.ini` and `learners.yaml`.
   
2. **Data Importing:** Imports data from Parquet file for a given symbol (e.g., NVDA).

   A. **Dynamic Feature Mapping:** Detects if a wildcard (`*`) is used for feature selection, and if so, automatically identifies and assigns all available columns (excluding the target labels) as the feature set.

3. **Preprocessing:** Split the input data into training and testing sets.

   A. **Sorting:** Sorts the dataset by the time column (`t`), to ensure the model learns from the past to predict on more recent data.

   B. **Time-Based Holdout Split:** Implements a time-series split where the final  days (defined by `t_d`) are used as a testing holdout while all preceding data is used for training.
   
   C. **One-Hot Encoding:** Performs One-Hot Encoding on columns defined within `configuration.ini`.
   
   D. **Cleanup:** Ensure the training and testing datasets contain only numeric data (types `int` and `float`) and removes all `NaN` values.
   
   E. **Normalization:** Fits a scaler (options include **Standard**, **MinMax**, or **Robust**) _only_ on the training data. This fitted scaler is then applied to the test data to prevent "data snooping" or look-ahead bias.
  
4. **Feature Selection:** (Optional) Performs one or more of the following feature selection methods. Variance threshold is almost always performed if enabled.

   A. **Variance Threshold:** Filters out features with low variance. This eliminates columns whose values don't change enough.

   B. **SelectKBest:** Uses Mutual Information to select the top N features that share the most information with the labels, helping to reduce noise.
   
   C. **Recursive Elimination (RFECV):** Performed using the Decision Tree estimator. It uses a `TimeSeriesSplit` cross-validation strategy to iteratively prune the least important features based on the Matthews Correlation Coefficient (MCC).

6. **Machine Learning:** Iterates through selected learners (defined in `learners.yaml`) to train a model. This process is performed over 25 random seeds and the final results are an average over all random seeds.

   A. **Load:** Parses the `learners.yaml` file to  import scikit-learn models with their specified hyperparameters. Each model class includes a separate section for hyperparameters to tune during optimization.
   
   B. **TimeSeries:** Executes `RandomizedSearchCV` using `TimeSeriesSplit` to tune hyperparameters. This ensures that the optimization process respects the chronological order of financial data, preventing the model from "cheating" by seeing future information during the tuning phase.
   
   C. **Cross-Validation:** Performs internal validation on the training set using chronological folds. This provides a mean MCC score and standard deviation.
   
   D. **Walk-Forward Rolling Retraining:** Model is periodically retrained as more recent data arrives. It supports:
      * **Retrain Step Frequency:** How often the model updates its parameters.
      * **Sliding Window:** Training on a fixed-size history.

7. **Results Aggregation:** Calculates mean scores and standard deviations across all seeds and exports the final metrics to CSV.

## Feature Engineering
This is the workflow used to engineer features. The Polygon.io service is used to obtain daily data for all tickers and data is obtained from 01/01/2023 to present.

### 1. Temporal & Cyclical Features
* **Calendar Units:** Extracts the day of the week, day of the month, month, quarter, and a binary flag for the end of the month.
* **Cyclical Encoding:** Uses Sine and Cosine transformations for days of the week and month.
> This allows the model to understand that "Sunday" (6) and "Monday" (0) are close to each other, rather than far apart numerically.

### 2. Price Action & Momentum (Lagged)
Looks back at the previous 15 trading days to capture historical context. Note that the number of days is configurable within the `configuration.ini`:

* **Historical OHLCV:** Creates 15 columns for each core metric (Open, High, Low, Close, Volume, VWAP, etc.).
* **Historical Volatility:** Calculates the High-Low spread and the daily return for each of the 15 lagged days.
* **Overnight Gaps:** Measures the percentage difference between the previous day's Close and the current day's Open, including a "Significant Gap" flag if the move is > 0.5%.

### 3. Technical Indicators
It calculates several standard technical analysis metrics across multiple time windows (5, 10, 15, 20, and 50 days):

* **Moving Average Distance:** Measures how far the current price has "stretched" away from its SMA and EMA (Simple and Exponential Moving Averages).
* **Rate of Change (ROC):** The percentage change in price over the specific window.
* **Rolling Volatility:** The standard deviation of returns over the window.
* **Normalized Range:** The high-to-low range over the window, scaled by the moving average.

### 4. Candlestick Anatomy
* **Body Size:** The absolute distance between Open and Close.
* **Shadows (Wicks):** The length of the upper and lower wicks (representing intraday price rejection).
* **Relative Body:** The ratio of the body to the total range (identifies "marubozu" vs "doji" type candles).

### 5. Interaction & Volume Features
* **Dollar Volume:** Close price multiplied by volume.
* **Return-Volume Interaction:** Multiplies the daily return by volume to identify high-conviction moves.
* **Spread-Volume:** Multiplies the High-Low spread by volume to detect volatile "churning" or exhaustion.