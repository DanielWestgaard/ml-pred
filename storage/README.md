# About /storage
This folder is used for storage of different purposes.

## Structure
```
├── data/                       # Storing datasets
│   ├── capital_com             # Datasets from Capital.com
│   │   ├── processed           # Processed datasets gone through the Data Pipeline
│   │   ├── raw                 # Raw data
│   │   └── saved_responses     # Responses from API
│   ├── alpha_vantage           # Datasets from Alpha Vantage
│   │   ├── processed
│   │   ├── raw
│   │   └── saved_responses
│   └── ...
├── data_engineeering_stats/    # Statistics and plots during data engineering pipeline
│   ├── fe_generation/
│   └── fe_selection/
├── models/                     # Trained models and metadata
├── backtesting_results/        # Results from backtesting
└── README.md                   # Documentation related to storage/
```