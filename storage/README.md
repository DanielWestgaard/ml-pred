# About /storage
This folder is used for storage of different purposes.

## Structure
```
├── data/                       # Storing datasets
│   ├── capital_com/            # Datasets from Capital.com
│   │   ├── BTCUSD/             # Bitcoin to USD data
│   │   │   ├── raw/            # Raw data from API
│   │   │   │   ├── 1m/         # 1-minute timeframe
│   │   │   │   ├── 5m/         # 5-minute timeframe
│   │   │   │   └── 1h/         # 1-hour timeframe
│   │   │   └── processed/      # Processed data through pipeline
│   │   │       ├── 1m/
│   │   │       ├── 5m/
│   │   │       └── 1h/
│   │   ├── GBPUSD/             # British Pound to USD data
│   │   │   ├── raw/
│   │   │   │   ├── 1m/
│   │   │   │   ├── 5m/
│   │   │   │   └── 1h/
│   │   │   └── processed/
│   │   │       ├── 1m/
│   │   │       ├── 5m/
│   │   │       └── 1h/
│   │   └── saved_responses/    # API responses for debugging
│   ├── alpha_vantage/          # Datasets from Alpha Vantage
│   │   ├── processed/
│   │   ├── raw/
│   │   └── saved_responses/
│   └── ...
│
├── data_engineering_stats/     # Statistics and plots during data engineering pipeline
│   ├── cleaned_data/           # Plots of dataset after cleaning
│   ├── fe_generation/
│   ├── fe_selection/
│   └── feature_importance/     # Feature importance results

│
├── models/                     # Trained models and metadata
│   ├── <datetimenow>/          # Name of directory for model will be of now: Eg. '2025-06-09_20:19:29/'
│   │   ├── trained_model.pkl   # Actual trained model. Eg. 'xgboost.pkl'
│   │   ├── metadata.yml        # Information (metadata) about model training, location, parameters, etc.
│   │   ├── evaluation/         # Evaluation of model and plots of its predictions. Created in evaluation.py (plot method).
│   │   │   ├── plot.pdf        # eg.
│   │   │   └── evaluation.pdf  # eg.
│   │   └── backtesting/        # Backtesting results done on that selected model
│   │       ├── backtest_results_<datetime>
│   │       └── trades_log_<datetime>
│   │
│   └── ...
│
├── multi_data_sources/         # TODO: Storing data from different sources (sentiment analysis, weather, news, etc.)
│   ├── calender_events/        # TODO: Calendar/Events
│   │   ├── ?
│   │   └── ?
│   └── ...
│
└── README.md                   # Documentation related to storage/
```

## File Naming Convention for historical data storage
Data files follow the pattern: `YYYY-MM-DD_YYYY-MM-DD.csv` (start_date_end_date.csv)

Example: `2025-05-01_2025-07-18.csv` contains data from May 1, 2025 to July 18, 2025.

## Organization Benefits
- **Symbol-based**: Easy to find data for specific trading pairs
- **Timeframe separation**: Clear organization by data granularity
- **Scalable**: Simple to add new symbols and timeframes
- **Consistent naming**: Standardized date format for easy sorting
- **Intuitive navigation**: Mirrors how traders think about market data
