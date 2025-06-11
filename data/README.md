# About data/ 💽
This folder will be solely responsible for engineering data that a ML model will use for training.

## Focus points (not updated - TODO) 🎯
- **Quality data**: Make sure the data is reliable.
- **A lot of historical data**: The more data, the better.
- **Cleaning**: Make sure data is accurate, consistent, and free from errors or irrelevant information.
- **Feature generation**: Generate as many features as possible to help a model predict better.
- **Feature selection**: Select the very best features with low correlation to each other and high prediction value.
- **Normalization**: Rescaling numeric data to a standard range (eg. between 0 and 1), to ensure that features contribute equally to distance based algorithms.
- ...


## Structure 🗂️
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
│   ├── target_hypothesis.py    # Functions realated to getting/calculating- and setting target to the selected hypothesis (based on 'name' var in the hypotheses config.yml-file)
│   └── selection.py            # Various techniques (3) to select the "most efficient" features
│
├── pipeline/                   # Pipeline orchestration
│   └── engineer_pipeline.py    # Main orchestrator and pipeline for engineering historical data
│
├── plot/                       # Functionality for plotting datasets
│   └── plot_data.py            # Simple plotting
│
└── README.md
```


## Important notes: 📝
- Cleaning, timezone: As I understand it, capital.com API uses "snapshotTimeUTC", so that's what I'll use (temporary). - _timestamp_alignment(). Should be improved when using other providers using another timezone.
- The library pandas_lib could really help with calculating features, but the following error appears on (on any Python version) ``` ImportError: cannot import name 'NaN' from 'numpy' ```. After some reasearch, this appears after numpy version 2.0. Considering I'll use this project and the whole pipeline on other servers/machines one day, I will create my own functions/calculations as to not rely on older libraries.
- Fetching historical data using providers/capital_com.py: It uses (naturally) an API that returns both bid and ask prices, but the code will "Calculate mid prices (average of bid and ask) with rounding" so its easier to train. I don't know if that's the best solution, but currently the one in place.
  - Under the ```broker/``` component, you can see that the actual ```broker/capital_com/rest_api/markets_info.py``` and the function called ```fetch_and_save_historical_prices()```, uses ```convert_json_to_ohlcv_csv``` method from ```utils/broker``` to "calculate" these "new prices" from the actual response.


# About pipelines/ 👨‍🔧

## Data Pipeline 💽 🪠
This is the main orchestrator for processing and engineering raw data into clean, validated, feature rich, processed data - ready to be used for training Machine Learning Model/-s.<br>

### How to use 🙋‍♂️
There are 2 ways of running the Data Pipeline on your raw data:
1. **Default/Standard**: The Pipeline will do the _usual flow_ of (simplified) `Cleaning` -> `Validation` -> `Feature Generation` -> `Transformation` -> `Feature Selection` -> `Save`.
    - Use `main.py` to run the Data Pipeline on your raw data: (flag-name might change in the future)
        ```bash
        python main.py -eg 'storage/data/capital_com/raw/raw_GBPUSD_m5_20240101_20250101.csv'
        ```
    - Or, implement it yourself. Eg.:
        ```python
        from data.pipelines.engineer_pipeline import EngineeringPipeline

        data_pipeline = EngineeringPipeline(raw_dataset = "storage/data/capital_com/raw/raw_GBPUSD_m_20250501_20250501.csv",
                                            save_cleaned_data = True)
        
        data_pipeline.run()
        ```
    
2. **Focusing on a Hypothesis**: The pipeline will have a somewhat different flow: `Cleaning` -> `Validation` -> `Generation` (only generating the specified features from given hypothesis config-file) -> `Target Hypothesis` (creating/calculating target column representing the Hypothesis - what the Model will predict) -> `Transformation` -> `Save`.<br>
This really is at **the heart of this project**. Check out the documentation under `hypothesis/` and `Target Hypothesis` (chapter _here_) to see what this is, and how to create one to focus on (`config.yml`, `README.md`, new method in `target_hypothesis.py`).<br>
When this is done, you can:
    - Use `main.py`and run the Pipeline like this (eg.):
        ```bash
        python main.py --engineer-data 'storage/data/capital_com/raw/raw_GBPUSD_m_20250101_20250523.csv' --hypothesis "hypothesis/1_session_based_tod_anomalies/config.yml"
        ```
    - Or, implement it yourself:
        ```python
        from data.pipelines.engineer_pipeline import EngineeringPipeline

        data_pipeline = EngineeringPipeline(raw_dataset = "storage/data/capital_com/raw/raw_GBPUSD_m_20250501_20250501.csv",
                                            save_cleaned_data = True,
                                            hypothesis_source = "hypothesis/1_session_based_tod_anomalies/config.yml")
        data_pipeline.run()
        ```

> [!NOTE]
> Before using Hypothesis on the Pipeline, make sure you have created the Hypothesis properly. See the documentation under `hypothesis/` and under `Processing`/`Target Hypothesis` in this document. Also, use the existing functionality as examples. This is currently a quick run-down:
> 1. `config.yml`-file for the hypothesis in the given format. Also recommended to have a `README.md` under the directory of your hypothesis for a more detailed description, but not needed for this functionality to work.
> 2. Create a method in the (`data/features/`)`target_hypothesis.py` to calculate your `target` (the hypothesis). Eg. like Hypothesis 1 ("session_based_tod_anomalies").<br>


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
Furthermore, there is also implemented a helper method, `_add_features_efficiently()`, that does excactly what the name suggests. The main goal of this function is to eliminate DataFrame fragmentation, improve performance, reduce memory usage, and better code organization. Hence, when improving Generation with new features, remember this approach:
```python
def some_feature(self):
    # Calculate all features first
    feature1 = calculation1
    feature2 = calculation2  
    feature3 = calculation3
    
    # Add all at once
    new_features = {
        'feature1': feature1,
        'feature2': feature2,
        'feature3': feature3
    }
    self._add_features_efficiently(new_features)
```
> [!NOTE]
> The feature generator creates several categorical features. Some ML Models requires only numerical input, so this should be handled accordingly for the models needs.
#### Future Performance tips (TODO - Note to self):
1. **Vectorize operations** when possible instead of loops
2. **Avoid intermediate DataFrames** - work with Series/arrays
3. **Group related calculations** - do all moving averages at once
4. **Use dropna() strategically** - only when necessary
5. **Pre-allocate arrays** for complex calculations like volume profiles

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
> [!NOTE]
> There are several steps in the normalization process that exludes some features with particular prefixes (eg. like starting with ```target_```) from normalization.

### About Creating Target Column for the Hypothesis - target_hypothesis.py
If the pipeline is being ran with ```hypothesis``` argument, the selection process is skipped and this is ran after generation.<br>
It's basically setting target column to the hypothesis. In the ```taget_hypothesis.py```-file, based on the name of the hypothesis in the ```config.yml```-file.<br>
For a more through and technical description, see the ```README.md``` documentation under the ```hypothesis/``` folder.
> [!IMPORTANT]
> When adding functionality for new hypotheses: Remember to add the prefix ```target_*``` for the _target column to the hypothesis_. This is because the normalization step after generation and target hypothesis will _not normalize_ columns that starts with ```target_```.

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



## About plot/ 📊
Simple functionality to plot candlestick data. Can both be run from main.py, but also used in Data Pipeline, after cleaning.<br>
Currently only using have to methods for using either ```plotly.graph_objects``` or ```mplfinance```.
### How to use
Recommended usage is either plotting raw data, or after cleaning the data (eg. in the Data Pipeline), as the final step in the Data Pipeline is normalization.<br>
To simply plot candlestick data:
```bash
python main.py --plot-data 'storage/data/capital_com/raw/raw_GBPUSD_m_20250515_20250523.csv'
```
### Challenges
- Plotting 1-minute candles the **datasets can get very large**, making the plotting and interaction _slow_, and also incredibly _difficult to see what's going on_. Current solution is a little improved ```plotly.graph_objects``` implementation in the method ```simple_small_go_candle_plot()```. Though this can easily be even more improved. Perhaps splitting up the plotting?
- **Plotting engineered data**: As most of the data has been _normalized_ (even ohlcv), I can't really check to see if the cleaning was done correctly. Solution is to _plot and save dataset after cleaning_ it, as part of the Data Pipeline.