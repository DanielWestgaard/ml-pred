After implementing better `depth_stability` feature and better BUY Signal Accuracy, I will try a new round of processing, training, backtesting:

Note:


# Round 1
1. Processing datasets: Command `python main.py --hyp 'hypothesis/spread_compression_patterns/config_process_data_v3.yml'`
    - Training dataset: logs "logs/data/logging_31072025_162420.log", results from feature imbalance: "storage/data_engineering_stats/feature_importance/2025-07-31_16-24-26"
    - Backtest dataset: logs "logs/data/logging_31072025_162543.log", results from feature imbalance: "storage/data_engineering_stats/feature_importance/2025-07-31_16-25-46"
2. Training new model on those features (without SMOTE): `python main.py --hyp 'hypothesis/spread_compression_patterns/config_model_training_v3_balanced.yml'`
    - Logfile: "logs/training/logging_31072025_162738.log"
    - Saved model path: "storage/models/2025-07-31_16:28:03"
3. Backtesting trained model on testing dataset: `python main.py --hyp hypothesis/spread_compression_patterns/config_backtest_v3_ultra_risk_optimized.yml`
    - Logfile: "logs/backtesting/logging_31072025_162905.log"
