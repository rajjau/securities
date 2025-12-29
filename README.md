# Project Summary: Financial ML Pipeline

### 1. Data Acquisition & Pre-processing (The `PRE_` Layer)
* **A. Ingestion (`PRE_1_download.py` & `polygon_io.py`)**
    * i. Automates market data downloads via Polygon.io API.
    * ii. Handles date expansion and trading day validation.
* **B. Consolidation (`PRE_2_combine.py` & `combine_json.py`)**
    * i. Aggregates daily JSON market snapshots into unified CSV files.
    * ii. Efficiently flattens nested API responses into tabular formats.
* **C. Transformation (`PRE_3_features.py` & `add_features.py`)**
    * i. Performs data cleaning (duplicate/null removal) and chronological sorting.
    * ii. Prepares the "Golden Dataset" for feature enrichment.

### 2. Feature Engineering
* **A. Technical Indicators**
    * i. **Moving Averages:** SMA/EMA over windows ranging from 5 to 50 days.
    * ii. **Momentum:** Rate of Change (ROC) and Relative Body Size of candles.
* **B. Temporal & Lagged Logic**
    * i. **Lagged Data:** Creates historical look-back windows (up to 15 days) for OHLCV data.
    * ii. **Time Features:** Extracts cyclical patterns from timestamps (Day of Week, Day of Month).
* **C. Encoding**
    * i. Implements One-Hot Encoding for categorical data to make it machine-learning ready.

### 3. Model Training & Evaluation (`PERFORM_ml.py`)
* **A. Pipeline Operations**
    * i. **Normalization:** Scales data using `RobustScaler`, `StandardScaler`, or `MinMaxScaler`.
    * ii. **Feature Selection:** Reduces noise using `VarianceThreshold` and `SelectKBest`.
* **B. Machine Learning Strategy (`machine_learning.py`)**
    * i. **Model Orchestration:** Manages multiple learners (RF, SVC, LogReg) via `learners.yaml`.
    * ii. **Optimization:** Uses `RandomizedSearchCV` with `TimeSeriesSplit` for leakage-free tuning.
* **C. Validation & Scoring**
    * i. **Walk-Forward Retraining:** Implements a rolling-window training schedule to simulate real trading.
    * ii. **Performance Metrics:** Ranks models using F1-Macro scores and generalization stability.

### 4. Infrastructure & Maintenance
* **A. System Control**
    * i. **`configuration.ini`:** Centralized control for symbols, holdout periods, and ML toggles.
    * i. **`learners.yaml`:** Centralized control for defining learners.
* **B. Lifecycle Management (`PERFORM_update.py`)**
    * i. Incremental update logic that identifies data gaps and fetches only new market days.
