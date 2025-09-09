# Data Science Assignment: Trader Behavior & Market Sentiment Analysis

**Candidate:** Hrutik Adsare\
**Submission For:** Web3 Trading Team

## 📁 Repository Structure

This repository is structured as per the assignment instructions:

    ds_Hrutik_Adsare/
    │
    ├── notebook_1.ipynb          # Main Google Colab notebook (View on Colab)
    ├── csv_files/                # Processed data & model artifacts
    │   ├── engineered_dataset.csv
    │   ├── metrics.csv
    │   ├── model_best_random_forest.pkl
    │   ├── scaler.pkl
    │   ├── best_info.json
    │   ├── feature_importance_*.csv
    │   └── classification_report_best.csv
    ├── outputs/                  # All visualizations & charts
    │   ├── sentiment_time.png
    │   ├── pnl_distribution.png
    │   ├── trade_size_vs_sentiment.png
    │   ├── roc_curve.png
    │   ├── precision_recall_curve.png
    │   ├── confusion_matrix.png
    │   └── feature_importance_*.png
    ├── ds_report.pdf             # Summary report of insights and conclusions
    └── README.md                 # This file

## 🚀 Quick Links

-   **Google Colab Notebook:** [Open notebook_1.ipynb in
    Colab](https://colab.research.google.com/drive/1N--eMl3-BNy8thEsz-VuiJMoknGaYQZa?usp=sharing) *(Set to 'Anyone with the link can
    view')*
-   **Fear & Greed Index Data:**
    [Download](https://drive.google.com/file/d/1PgQC0tO8XN-wqkNyghWC_mnrYv_nh5f/view?usp=sharing)
-   **Historical Trader Data:**
    [Download](https://drive.google.com/file/d/1IAfLZwu6rizyWKgBTogwSmmVYU6VbjVs/view?usp=sharing)

## 🎯 Objective

This project analyzes the relationship between on-chain trader behavior
(from Hyperliquid) and overall market sentiment (Bitcoin Fear & Greed
Index) to identify patterns and signals that could inform smarter
trading strategies.

## 🔧 Setup & Installation

The entire analysis is built in **Google Colab** and requires no local
setup. To run it:

1.  Open the main notebook: [notebook_1.ipynb](https://colab.research.google.com/drive/1N--eMl3-BNy8thEsz-VuiJMoknGaYQZa?usp=sharing)
2.  Ensure you are logged into a Google account.
3.  Runtime -\> Run all
    -   The notebook will automatically mount Google Drive, install
        necessary packages, and run the entire pipeline.

### Dependencies

All packages are installed within the notebook. Key libraries used: -
`pandas`, `numpy`, `matplotlib`, `seaborn` - `scikit-learn` -
`lightgbm`, `xgboost` - `joblib` (for saving models)

## 📊 Methodology & Workflow

### 1. Data Loading & Cleaning

-   Loaded the two provided datasets.
-   Parsed and standardized timestamps to UTC.
-   Handled missing values and incorrect data types.

### 2. Feature Engineering (Key to High Performance)

This was the most critical step for capturing the relationship between
behavior and sentiment.

**A. Sentiment Features:** - Mapped the `Classification` string to a
numerical score (`Extreme Fear: -2` ... `Extreme Greed: +2`). - Created
lagged (`value_lag1`) and rolling average (`value_3d_mean`,
`value_7d_mean`) features from the Fear & Greed Index to capture trends
and momentum.

**B. Trader Behavior Features:** - **Rolling Statistics per Account:**
For each trader, calculated rolling window (last 50 trades) metrics: -
`roll50_winrate`: Recent success rate. - `roll50_avg_size_usd`: Recent
average trade size. - `roll50_buy_ratio`: Recent bias towards long
positions. - `roll50_pnl_std`: Recent volatility/risk in performance. -
`roll50_pnl_mean`: Recent average profitability. -
`roll50_intertrade_mean`: Recent average time between trades (activity
frequency). - **Normalized Features:** Created z-scores for
`Execution_Price` and `Size_USD` within each coin to identify
outliers. - `log_size_usd`: Log-transformed trade size to handle its
skewed distribution.

**C. Interaction Features:** - `sentiment_x_size`: Interaction between
market sentiment and trade size. - `sentiment_x_rollwin`: Interaction
between market sentiment and the trader's recent win rate.

**Target Variable:** `target_profit` - A binary label indicating if a
trade resulted in a profit (`Closed_PnL > 0`).

### 3. Modeling & Evaluation

-   **Time-Based Split:** Data was split chronologically (80% train, 20%
    validation) to prevent look-ahead bias and simulate a real-world
    backtest.
-   **Models Trained & Compared:**
    -   Logistic Regression (baseline)
    -   Random Forest
    -   LightGBM
    -   XGBoost
-   **Threshold Tuning:** The classification threshold was optimized for
    each model to maximize accuracy on the validation set, as the
    dataset was not perfectly balanced.
-   **Comprehensive Evaluation:** Models were evaluated using Accuracy,
    Precision, Recall, F1-Score, ROC-AUC, and Precision-Recall AUC.

## 🏆 Results & Key Findings

### Best Model Performance

The **Random Forest** model performed best on the validation set: -
**Accuracy:** 89.8% - **Precision:** 89.0% - **Recall:** 82.8% -
**F1-Score:** 85.8% - **ROC-AUC:** 0.963

*(See `csv_files/metrics.csv` for full results across all models)*

### Key Insights

1.  **Trader's Recent Performance is King:** The most important feature
    was consistently the **rolling 50-trade winrate
    (`roll50_winrate`)**. A trader's recent history is the strongest
    predictor of their next trade's outcome.
2.  **Sentiment Modulates Behavior:** The interaction features
    (`sentiment_x_size`, `sentiment_x_rollwin`) were among the top
    predictors. This proves that market sentiment doesn't act alone but
    **amplifies or dampens existing trader behavior** (e.g., traders may
    trade with larger sizes during "Greed" phases, but not necessarily
    more profitably).
3.  **High Predictive Power:** The extremely high ROC-AUC score (0.96)
    demonstrates a very strong signal and a clear relationship between
    the engineered features (behavior + sentiment) and trade
    profitability.

## 📈 How to Use This Analysis

The saved best model (`model_best_random_forest.pkl`) and scaler
(`scaler.pkl`) can be loaded into a Python environment to make
predictions on new, unseen data. The feature engineering pipeline
defined in the notebook must be applied to any new data first.

``` python
import joblib
model = joblib.load('csv_files/model_best_random_forest.pkl')
scaler = joblib.load('csv_files/scaler.pkl')
# ... Apply the same feature engineering steps to new data ...
# predictions = model.predict_proba(new_data)
```

## 🔮 Future Work

-   **SHAP Analysis:** For deeper model interpretability
    (computationally expensive, demonstrated in a separate notebook
    variant).
-   **Predicting PnL Magnitude:** Transition from a classification
    problem (profit/loss) to a regression problem (predicting exact
    PnL).
-   **Live Strategy Integration:** The model could be used as a signal
    within a larger trading strategy to evaluate the quality of one's
    own trades or those of others on the network.

------------------------------------------------------------------------

**Note for Reviewers:** Thank you for your time and consideration. The
complete end-to-end pipeline is contained in `notebook_1.ipynb`. All
outputs(except model file) are saved in the designated folders.
