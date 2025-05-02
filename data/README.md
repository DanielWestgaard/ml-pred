# About data/
This folder will be solely responsible for engineering data that a ML model will use for training.

## Focus points
- **Quality data**: Make sure the data is reliable.
- **A lot of historical data**: The more data, the better.
- **Cleaning**: Make sure data is accurate, consistent, and free from errors or irrelevant information.
- **Feature generation**: Generate as many features as possible to help a model predict better.
- **Feature selection**: Select the very best features with low correlation to each other and high prediction value.
- **Normalization**: Rescaling numeric data to a standard range (eg. between 0 and 1), to ensure that features contribute equally to distance based algorithms.
- ...

## Structure
```
├── providers                   # Different providers/sources to get/retrieve historical data
│   ├── base_provider.py        # Interface for different providers
│   ├── capital_com.py          # Using the broker Capital.com's API to fetch historical data, very simple
│   ├── alpha_vantage.py        # Stock Market Data API, 30 days free intraday market data
│   ├── ...
│   └── ...
└── README.md
```