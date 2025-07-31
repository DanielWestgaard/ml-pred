# Round 1
`validate_balanced_targets.py`

1. Processing datasets: Command `python main.py --hyp 'hypothesis/spread_compression_patterns/config_process_data_v3.yml'`
    - Training dataset: logs "logs/data/logging_31072025_173426.log", results from feature imbalance: "storage/data_engineering_stats/feature_importance/2025-07-31_17-34-31"
    - Backtest dataset: logs "logs/data/logging_31072025_173351.log", results from feature imbalance: "storage/data_engineering_stats/feature_importance/2025-07-31_17-33-54"
2. Training new model on those features (without SMOTE): `python main.py --hyp 'hypothesis/spread_compression_patterns/config_model_training_v3_balanced.yml'`
    - Logfile: "logs/training/logging_31072025_173529.log"
    - Saved model path: "storage/models/2025-07-31_17:35:55"
3. Backtesting trained model on testing dataset: `python main.py --hyp hypothesis/spread_compression_patterns/config_backtest_v3_ultra_risk_optimized.yml`
    - Logfile: "logs/backtesting/logging_31072025_173646.log"
    - Last line: "Return: -2.08%, Trades: 465, Win Rate: 0.0%"
        - This is actually an improvement on "Return" as, last time, before changing some of the imbalanced classes, it was on "-2.22%", but still 0% win
