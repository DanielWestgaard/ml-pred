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
│
├── data_engineeering_stats/    # Statistics and plots during data engineering pipeline
│   ├── cleaned_data/           # Plots of dataset after cleaning
│   ├── fe_generation/
│   └── fe_selection/
│
├── models/                     # Trained models and (TODO:) metadata
│   ├── <datetimenow>/          # Name of directory for model will be of now: Eg. '2025-06-09_20:19:29/'
│   │   ├── trained_model.pkl   # Actual trained model. Eg. 'xgboost.pkl'
│   │   ├── metadata.yml        # Information (metadata) about model training, location, parameters, etc.
│   │   └── evaluation/         # Evaluation of model and plots of its predictions. Created in evaluation.py (plot method).
│   │       ├── plot.pdf        # eg.
│   │       └── evaluation.pdf  # eg.
│   └── ...
│
├── backtesting_results/        # TODO: Results from backtesting
│
└── README.md                   # Documentation related to storage/
```