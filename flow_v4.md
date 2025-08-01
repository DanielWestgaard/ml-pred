## Quick Intro
Read `README.md` for and overview of the project, and `hypothesis/spread_compression_patterns/README.md` for information about the hypothesis I'm trying to make work.

I've really been struggling to make my model profitable (backtesting showed consistently "0% win rate and always negative returns, with extremly low confidence on BUY/SELL"). 

# Round 1

**Training data period**: `2023-01-01_2025-07-01`
**Backtesting data period**: `2025-07-02_2025-07-31`

## 🔧 KEY FIXES IMPLEMENTED:

1. **Target Logic**: Now predicts future movements instead of labeling past patterns
2. **Temporal Features**: Model now understands market direction and momentum
3. **Lookahead Prevention**: All features strictly use historical data only
4. **Signal Processing**: Fixed filtering bugs that blocked 99% of predictions
5. **Feature Quality**: Added 19+ new predictive features with market microstructure
6. **Memory efficient feature generation and feature importance analysis**

## Actual testing flow and paths
1. Processing datasets: Command `python main.py --hyp 'hypothesis/spread_compression_patterns/config_process_data_v4_enhanced.yml'`
    - Training dataset: logfile "logs/data/logging_01082025_133755.log", results from feature imbalance: "storage/data_engineering_stats/feature_importance/2025-08-01_13-44-03"
    - Backtest dataset: logfile "logs/data/logging_01082025_134518.log", results from feature imbalance: "storage/data_engineering_stats/feature_importance/2025-08-01_13-45-32"
2. Training new model on those features (without SMOTE): `python main.py --hyp 'hypothesis/spread_compression_patterns/config_model_training_v4_enhanced.yml'`
    - Logfile: "logs/training/logging_01082025_134621.log"
    - Saved model path: "storage/models/2025-08-01_13:46:57"
3. Backtesting trained model on testing dataset: `python main.py --hyp 'hypothesis/spread_compression_patterns/config_backtest_v4_enhanced.yml'`
    - Logfile: "logs/backtesting/logging_01082025_134803.log"
    - Last line: "Return: 0.00%, Trades: 0, Win Rate: 0.0%"


## Results
The OOB score of 1.0000 during training is a classic sign of overfitting - the model has memorized
the training data perfectly but fails to generalize.

Analysis Summary:

The Problem:
1. Model is completely broken - predicting ONLY class 0 (HOLD) for every single prediction
2. Perfect overfitting - OOB score of 1.0 indicates the model memorized training data
3. No trading signals - All predictions get 0.05 confidence (hardcoded for class 0) which is below
the 0.15 threshold
4. Zero trades executed - Result: 0% return, 0 trades, 0% win rate

Why this happened:
1. Feature engineering went wrong - Added 19+ new features that likely introduced data leakage or
noise
2. Model complexity mismatch - 500 trees with max_depth=12 on possibly limited/corrupted training
data
3. Possible data leakage - Despite claims of "lookahead prevention", the perfect OOB score
suggests the model learned spurious patterns
4. Class imbalance issues - Model defaulting to majority class (HOLD) exclusively

The model hasn't just "gone worse" - it's completely non-functional. Your previous iterations were
likely much better than this, even if they had low profitability.

---

# Round 2
1. Processing datasets: Command `python main.py --hyp 'hypothesis/spread_compression_patterns/config_process_data_v4_enhanced.yml'`
    - Training dataset: logfile "logs/data/logging_01082025_151128.log", results from feature imbalance: "storage/data_engineering_stats/feature_importance/2025-08-01_15-12-24"
    - Backtest dataset: logfile "logs/data/logging_01082025_151506.log", results from feature imbalance: "storage/data_engineering_stats/feature_importance/2025-08-01_15-15-10"
2. Training new model on those features (without SMOTE): `python main.py --hyp 'hypothesis/spread_compression_patterns/config_model_training_v4_enhanced.yml'`
    - Logfile: "logs/training/logging_01082025_151620.log"
    - Saved model path: "storage/models/2025-08-01_15:16:41"
3. Backtesting trained model on testing dataset: `python main.py --hyp 'hypothesis/spread_compression_patterns/config_backtest_v4_enhanced.yml'`
    - Logfile: "logs/backtesting/logging_01082025_151716.log"
    - Last line: ""

## Results
...

---


# Template for 
1. Processing datasets: Command `python main.py --hyp 'hypothesis/spread_compression_patterns/config_process_data_v3.yml'`
    - Training dataset: logfile "", results from feature imbalance: ""
    - Backtest dataset: logfile "", results from feature imbalance: ""
2. Training new model on those features (without SMOTE): `python main.py --hyp 'hypothesis/spread_compression_patterns/config_model_training_v3_balanced.yml'`
    - Logfile: ""
    - Saved model path: ""
3. Backtesting trained model on testing dataset: `python main.py --hyp hypothesis/spread_compression_patterns/config_backtest_v3_ultra_risk_optimized.yml`
    - Logfile: ""
    - Last line: ""