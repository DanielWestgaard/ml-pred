# Round 1
`validate_balanced_targets.py`

1. Processing datasets: Command `python main.py --hyp 'hypothesis/spread_compression_patterns/config_process_data_v3.yml'`
    - Training dataset: logs "", results from feature imbalance: ""
    - Backtest dataset: logs "", results from feature imbalance: ""
2. Training new model on those features (without SMOTE): `python main.py --hyp 'hypothesis/spread_compression_patterns/config_model_training_v3_balanced.yml'`
    - Logfile: ""
    - Saved model path: ""
3. Backtesting trained model on testing dataset: `python main.py --hyp hypothesis/spread_compression_patterns/config_backtest_v3_ultra_risk_optimized.yml`
    - Logfile: ""
    - Last line: ""
