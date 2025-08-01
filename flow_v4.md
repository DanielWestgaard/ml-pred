# Round 1

**Training data period**: `2020-01-01_2025-07-01`
**Backtesting data period**: `2025-07-02_2025-07-31`

1. Processing datasets: Command `python main.py --hyp 'hypothesis/spread_compression_patterns/config_process_data_v4_enhanced.yml'`
    - Training dataset: logs "logs/data/logging_01082025_114010.log", results from feature imbalance: "storage/data_engineering_stats/feature_importance/2025-08-01_11-44-55"
    - Backtest dataset: logs "", results from feature imbalance: ""
2. Training new model on those features (without SMOTE): `python main.py --hyp 'hypothesis/spread_compression_patterns/config_model_training_v3_balanced.yml'`
    - Logfile: ""
    - Saved model path: ""
3. Backtesting trained model on testing dataset: `python main.py --hyp hypothesis/spread_compression_patterns/config_backtest_v3_ultra_risk_optimized.yml`
    - Logfile: ""
    - Last line: ""


## Results
...


---


# Template for 
1. Processing datasets: Command `python main.py --hyp 'hypothesis/spread_compression_patterns/config_process_data_v3.yml'`
    - Training dataset: logs "", results from feature imbalance: ""
    - Backtest dataset: logs "", results from feature imbalance: ""
2. Training new model on those features (without SMOTE): `python main.py --hyp 'hypothesis/spread_compression_patterns/config_model_training_v3_balanced.yml'`
    - Logfile: ""
    - Saved model path: ""
3. Backtesting trained model on testing dataset: `python main.py --hyp hypothesis/spread_compression_patterns/config_backtest_v3_ultra_risk_optimized.yml`
    - Logfile: ""
    - Last line: ""