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
Will also use this for planning.
```
├── providers                   # Different providers/sources to get/retrieve historical data
│   ├── base_provider.py        # Interface for different providers
│   ├── capital_com.py          # Using the broker Capital.com's API to fetch historical data, very simple
│   └── alpha_vantage.py        # Stock Market Data API, 30 days free intraday market data
│
├── processing/                 # Data transformation
│   ├── cleaning.py
│   ├── validation.py           # Don't know if this is relevant but
│   ├── normalization.py        # Rescaling numeric data to a standard range
│
├── features/                   # Feature engineering
│   └── ...
│
├── pipeline/                   # Pipeline orchestration
│   └── ...
│
└── README.md
```

## Important notes:
- Cleaning, timezone: As I understand it, capital.com API uses "snapshotTimeUTC", so that's what I'll use (temporary). - _timestamp_alignment(). Should be improved when using other providers using another timezone.
- The library pandas_lib could really help with calculating features, but the following error appears on (on any Python version) ``` ImportError: cannot import name 'NaN' from 'numpy' ```. After some reasearch, this appears after numpy version 2.0. Considering I'll use this project and the whole pipeline on other servers/machines one day, I will create my own functions/calculations as to not rely on older libraries.

## About pipelines/
...

## About processing/
All code related to actual processing, like cleaning, validation, normalizing, etc. will be placed here.
### 1. Cleaning
Simple cleaning class that performs the following: 
- **Handles missing values** (detailed description in class for the different columns).
- **Removes duplicates** (of dates).
- **Timestamp alignement** (eg. Timezone normalization to UTC, converting date to datetime, etc.).
- **Handles outliers**, first with IQR method, and then with Z-Score.
- Ensures **concistency in datatypes** (timestamps, prices as floats, volume as integers).
- **OHLC validity**.
### 2. Validation 
Simple validation class that performs validation of data after cleaning, to verify data integrity and ensure loical consistency. These are the current validations done:
- **OHLC Relationships**, similar to the one in cleaning, but using this to validate that work.
- **Timestamps** for integrity and consistency.
- **Volume** 
- **Price Movements** for extreme or suspicous patterns.
- **Data Completeness**
### 4. Normalization
NSY...

## About features/
Code related to creating and selecting features with prediction power to help a ML model train and predict.
### Feature Generation
Planned features to generate: WIP
- <ins>Price Action Features</ins>
  - **Moving Averages**: Simple and exponential moving averages (SMAs/EMAs) over different time periods (5, 10, 20, 50, 200 periods)
  - **Price Momentum**: Rate of change (ROC) over various lookback periods
  - **Volatility Indicators**: Average True Range (ATR), Bollinger Band width
  - **Support/Resistance**: Distance from recent highs/lows
- <ins>Volume-Based Features</ins>
  - **Volume Moving Averages**: To identify unusual volume spikes
  - **Volume Rate of Change**: How rapidly volume is increasing/decreasing
  - On-Balance Volume (OBV): Cumulative indicator that relates volume to price changes
  - Volume Profile: Distribution of volume at different price levels
- <ins>Technical Indicators</ins>
  - RSI (Relative Strength Index): Measures momentum and overbought/oversold conditions
  - MACD (Moving Average Convergence Divergence): Trend-following momentum indicator
  - ADX (Average Directional Index): Measures trend strength
  - Stochastic Oscillator: Compares current price to range over time period
- Time-Based Features
  - Time of day/week/month features: Markets often exhibit cyclical patterns
  - Seasonality components: Extracted using decomposition methods
- Market Regime Features
  - Volatility regimes: High vs. low volatility periods
  - Trend strength indicators: To identify trending vs. ranging markets
- Feature Transformations
  - Log returns: Instead of raw price changes (more normally distributed)
  - Z-score normalization: Standardizing indicators for better model performance
  - Fourier transforms: For extracting cyclical components

## About providers/
Files under this are used solely used for fetching raw historical data. This data is intended to engineer and used for model training.<br>
Default behaviour is to store the files under storage/data and under the respective provider.
### Capital.com
Only uses the broker/capitalcom.py implementation to only get historical data. For more information on the broker itself, see broker/ documentation.<br>
The broker itself has relatively restricted with historical data, especially for 1min intraday timeframe, but useful in early stages. 5min has a little more.
#### How to use
Only one way to use this, no matter timeframe, symbol, etc. Example:
```python
from data.providers.capital_com import ProviderCapitalCom

provider = ProviderCapitalCom()
provider.fetch_and_save_historical_data(symbol="GBPUSD", timeframe="MINUTE_5",
                                        from_date="2024-04-15T00:00:00", to_date="2025-05-01T01:00:00",
                                        print_answer=False)
```
#### Useful links:
- [API documentation](https://open-api.capital.com/)
- [Requirements to use their API](https://open-api.capital.com/#section/Getting-started)
### Alpha Vantage
Different restrictions on free api token, but seems quite good when you have the paid subscription.<br>
Currently (for this use), intraday can be used for (eg.) "core stocks", but not intraday Forex as it's not a part of free plan.<br>
This is also why methods like "saving the name of raw datafile" is very proprietary and can be seen as "not finished", as I'm planning to use this provider 
(or someone else) when this project has come further down.
#### How to use
This one is a little different. Current implementation is only tested, and implemented, with free api key and used on "Core Stock APIs". So if you're using this on forex or others, you will need to change fetch_historical_data() according to the selected "type of API" you need. Documentation is fairly good, so should be okay.<br>
Example of current use:
```python
from data.providers.alpha_vantage import ProviderAlphaVantage

provider = ProviderAlphaVantage()
provider.fetch_and_save_historical_data(symbol="NVDA", timeframe="1min",
                                        month="2025-04", print_answer=False,
                                        store_answer=True)
```
#### Useful links:
- [API documentation](https://www.alphavantage.co/documentation/)
- [Get free API token](https://www.alphavantage.co/support/#api-key)
- [Premium API key](https://www.alphavantage.co/premium/) for actually utilizing the providers' possibilities large datasets.