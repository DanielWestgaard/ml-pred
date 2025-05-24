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
│   ├── base_provider.py        # Interface for different providers. I think, currently, almost all files inherits from this - could be moved up "one level" (../)
│   ├── capital_com.py          # Using the broker Capital.com's API to fetch historical data, very simple
│   └── alpha_vantage.py        # Stock Market Data API, 30 days free intraday market data
│
├── processing/                 # Data transformation
│   ├── cleaning.py             # Cleaning data (missing values, duplicates, alignement, etc.)
│   └── validation.py           # Validating cleaned data
│
├── features/                   # Feature engineering
│   ├── generation.py           # Generating features with simplistic error handling
│   ├── normalization.py        # Rescaling numeric data to a standard range
│   └── selection.py            # Various techniques (3) to select the "most efficient" features
│
├── pipeline/                   # Pipeline orchestration
│   └── engineer_pipeline.py    # Main orchestrator and pipeline for engineering historical data
│
└── README.md
```


## Important notes:
- Cleaning, timezone: As I understand it, capital.com API uses "snapshotTimeUTC", so that's what I'll use (temporary). - _timestamp_alignment(). Should be improved when using other providers using another timezone.
- The library pandas_lib could really help with calculating features, but the following error appears on (on any Python version) ``` ImportError: cannot import name 'NaN' from 'numpy' ```. After some reasearch, this appears after numpy version 2.0. Considering I'll use this project and the whole pipeline on other servers/machines one day, I will create my own functions/calculations as to not rely on older libraries.
- Fetching historical data using providers/capital_com.py: It uses (naturally) an API that returns both bid and ask prices, but the code will "Calculate mid prices (average of bid and ask) with rounding" so its easier to train. I don't know if that's the best solution, but currently the one in place.
  - Under the ```broker/``` component, you can see that the actual ```broker/capital_com/rest_api/markets_info.py``` and the function called ```fetch_and_save_historical_prices()```, uses ```convert_json_to_ohlcv_csv``` method from ```utils/broker``` to "calculate" these "new prices" from the actual response.


## About pipelines/
...


## About processing/
All code related to actual processing, like cleaning, validation, normalizing, etc. will be placed here.

### Cleaning
Simple cleaning class that performs the following: 
- **Handles missing values** (detailed description in class for the different columns).
- **Removes duplicates** (of dates).
- **Timestamp alignement** (eg. Timezone normalization to UTC, converting date to datetime, etc.).
- **Handles outliers**, first with IQR method, and then with Z-Score.
- Ensures **concistency in datatypes** (timestamps, prices as floats, volume as integers).
- **OHLC validity**.

### Validation 
There are two validation classes in ```validation.py```.<br>
Simple validation class that performs validation of data after cleaning, to verify data integrity and ensure loical consistency. These are the current validations done:
- **OHLC Relationships**, similar to the one in cleaning, but using this to validate that work.
- **Timestamps** for integrity and consistency.
- **Volume** 
- **Price Movements** for extreme or suspicous patterns.
- **Data Completeness**
The second class is responsible for validating the features after feature generation.


## About features/
Code related to creating and selecting features with prediction power to help a ML model train and predict.

### Feature Generation
The feature generator generates a lot of features (100+). I have tried to implement and use a lot of the major technical indicators and features types relevant for financial time series. The easiest way to get a quick glance on the features are through ```run()``` in the ```FeatureGenerator``` class.<br>
To handle errors, but also keep a more simplistic code, have I implemented ```safely_execute()``` that I run/calculate all features through. This way error handling is improved (not the best) and code is still simplistic.<br>
> [!NOTE]
> The feature generator creates several categorical features. Some ML Models requires only numerical input, so this should be handled accordingly for the models needs.

### Transformation - transformation.py
#### Preparation of features before normalization
- **Handling missing values**: Some steps to handle missing values, as this can mess up normalization and selection later.
  1. Dropping whole features (technically columns) that contain more empty values (```Null```, ```NaN```) than the threshold (default is 50%).
  2. Based on a fixed/static list with feature names, extracts the maximum window size (eg. ```n```) and drops the first ```n``` rows. This can also be fixed (TODO).
    - **<ins>Note</ins>**: This fix _assumes_ all generated features (that requires past data to calculate _and_ the features manually added in the ```high_feature_windows``` list) starts with the abbreviation or one-word of the feature name, followed by "_" and the window size. Eg. For Simple Moving Average with window size 50: ```sma_50```.
- **Handles duplicated features**/columns.
#### Normalization
Simple Normalization class that uses **two Normalization Techniques**; [Z-Score](https://www.investopedia.com/terms/z/zscore.asp) for _Unbounded_ (no boundaries of certainty) features and [MinMax Scaker](https://medium.com/@iamkamleshrangi/how-min-max-scaler-works-9fbebb9347da) for _Bounded_ (field that has limited scope) features.<br>
The ```run()``` has some functionality that excludes some features from normalization:
- **String Categorical Features**:
  - ```market_state_desc``` (e.g., 'low_vol_ranging', 'high_vol_strong_trend')
  - ```adx_trend_regime``` (e.g., 'trending', 'ranging')
  - ```trend_regime``` (e.g., 'strong_trend', 'weak_trend', 'ranging')
  - ```volatility_regime``` (e.g., 'high', 'medium', 'low')
  - ```session``` (e.g., 'asian_session', 'european_session', 'us_session')
  - ```month_period``` (e.g., 'start_of_month', 'mid_month', 'end_of_month')
- **Binary/Integer Features**:
  - ```is_holiday```, ```is_weekend```
  - ```in_value_area```
  - One-hot encoded regime columns with prefixes like ```vol_regime_```, ```adx_regime_```, ```trend_regime_```, ```state_```
- **Ordinal Numeric Codes**:
  - ```market_state``` (0-8 representing different market states)

### Feature Selection - selection.py
The feature selection file performs _very simple_ Feature Selection, by using different techniques/methods, to then select the features that two or more techniques agree about:
1. **Correlation-based Feature Selection**: Selects subsets of features that are highly correlated with the target variable (```close```), but have low correlation with each other.
2. **XGB Regression**: "Ranks" features based on importance, and selects features above ```threshold```-variable.
3. **Recursive Feature Elimination**: Fits  a model (XGB Regressor) and removes the weakest feature (or features) until the specified number of features is reached.

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