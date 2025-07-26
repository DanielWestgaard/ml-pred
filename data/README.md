# About data/ 💽
This folder is responsible for all data engineering operations, supporting both **historical data processing** for model training and **real-time live data streaming** for strategy execution. The data system provides a complete pipeline from raw market data ingestion to feature-rich datasets ready for ML model training and live trading.

## Key Capabilities 🚀
- **Historical Data Processing**: Complete pipeline for cleaning, validating, and engineering historical market data
- **Live Data Streaming**: Real-time market data ingestion via WebSocket and REST APIs
- **Model-Aware Feature Generation**: Intelligent feature generation based on model requirements
- **Quality Assurance**: Multi-layer data validation and cleaning with error handling
- **Hypothesis-Driven Processing**: Configurable feature generation based on specific market hypotheses
- **Real-time Feature Pipeline**: Live feature generation for strategy execution
- **Multi-Provider Support**: Support for various data providers (Capital.com, Alpha Vantage)

## Structure 🗂️
```
data/
├── providers/                          # Data source providers
│   ├── base_provider.py                # Provider interface/abstract base
│   ├── capital_com.py                  # Capital.com API integration
│   └── alpha_vantage.py                # Alpha Vantage API integration
│
├── processing/                         # Data transformation and validation
│   ├── base_processor.py               # Base processor interface
│   ├── cleaning.py                     # Data cleaning (missing values, duplicates, outliers)
│   └── validation.py                   # Data validation and integrity checks
│
├── features/                           # Unified feature engineering system
│   ├── enhanced_generation.py          # ModelAwareFeatureGenerator - unified feature system (200+ features)
│   ├── feature_registry.py             # Feature registration and dependency management
│   ├── transformation.py               # Feature transformation and normalization
│   ├── selection.py                    # Feature selection (CFS, XGB, RFE)
│   └── target_hypothesis.py            # Hypothesis-specific target generation
│
├── pipelines/                          # Data pipeline orchestration
│   └── engineer_pipeline.py            # Main historical data processing pipeline
│
├── live/                               # Real-time data management
│   ├── live_data_manager.py            # LiveDataManager - main live data orchestrator
│   └── websocket_data_manager.py       # WebSocket data streaming management
│
├── plot/                               # Data visualization
│   └── plot_data.py                    # Candlestick and data plotting utilities
│
└── README.md                           # Documentation
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
        python main.py -eg 'storage/data/capital_com/GBPUSD/raw/5m/2024-01-01_2025-01-01.csv'
        ```
    - Or, implement it yourself. Eg.:
        ```python
        from data.pipelines.engineer_pipeline import EngineeringPipeline

        data_pipeline = EngineeringPipeline(raw_dataset = "storage/data/capital_com/GBPUSD/raw/1m/2025-05-01_2025-05-01.csv",
                                            save_cleaned_data = True)
        
        data_pipeline.run()
        ```
        
> [!NOTE]
> The pipeline now uses the unified **ModelAwareFeatureGenerator** system directly, providing consistent feature generation between training and live trading environments.
    
2. **Focusing on a Hypothesis**: The pipeline will have a somewhat different flow: `Cleaning` -> `Validation` -> `Generation` (only generating specified features from config-file) -> `Target Hypothesis` (generating _target column_ representing the Hypothesis) -> `Transformation` -> `Save`.<br>
This really is at **the heart of this project**. Check out the documentation under `hypothesis/` and `Target Hypothesis` (chapter _in this readme file_) to see what this is, and how to create one to focus on (`config.yml`, `README.md`, new method in `target_hypothesis.py`).<br>
When this is done, you can:
    - **Recommended**: Use `main.py`and run the Pipeline, with hypothesis config, like this (eg.):
        ```bash
        python main.py -hyp "hypothesis/1_session_based_tod_anomalies/config.yml"
        ```
    - **Older version of using hypothesis flow**: Use `main.py`and run the Pipeline like this (eg.):
        ```bash
        python main.py --engineer-data 'storage/data/capital_com/GBPUSD/raw/1m/2025-01-01_2025-05-23.csv' --hypothesis "hypothesis/1_session_based_tod_anomalies/config.yml"
        ```
    - Or, implement it yourself:
        ```python
        from data.pipelines.engineer_pipeline import EngineeringPipeline

        data_pipeline = EngineeringPipeline(raw_dataset = "storage/data/capital_com/GBPUSD/raw/1m/2025-05-01_2025-05-01.csv",  # Using hypothesis config-file, this will be dynamically fetched - so no need to provide if so
                                            save_cleaned_data = True,
                                            hypothesis_source = "hypothesis/1_session_based_tod_anomalies/config.yml")
        data_pipeline.run()
        ```

> [!NOTE]
> The pipeline uses the modern list-format for feature specification from hypothesis configs. The unified system automatically resolves dependencies and generates features in the optimal order.

> [!NOTE]
> Before using Hypothesis on the Pipeline, make sure you have created the Hypothesis properly. See the documentation under `hypothesis/` and under `Features`/`Target Hypothesis` in this document. Also, use the existing functionality as examples. This is currently a quick run-down:
> 1. `config.yml`-file for the hypothesis in the given format. Also recommended to have a `README.md` under the directory of your hypothesis for a more detailed description, but not needed for this functionality to work.
> 2. Create a method in the (`data/features/`)`target_hypothesis.py` to calculate your `target` (the hypothesis). Eg. like Hypothesis 1 ("session_based_tod_anomalies").<br>


# About processing/
All code related to actual processing, like cleaning, validation, normalizing, etc. will be placed here.

## Cleaning
Simple cleaning class that performs the following: 
- **Handles missing values** (detailed description in class for the different columns).
- **Removes duplicates** (of dates).
- **Timestamp alignement** (eg. Timezone normalization to UTC, converting date to datetime, etc.).
- **Handles outliers**, first with IQR method, and then with Z-Score.
- Ensures **concistency in datatypes** (timestamps, prices as floats, volume as integers).
- **OHLC validity**.

## Validation 
There are two validation classes in ```validation.py```.<br>
Simple validation class that performs validation of data after cleaning, to verify data integrity and ensure loical consistency. These are the current validations done:
- **OHLC Relationships**, similar to the one in cleaning, but using this to validate that work.
- **Timestamps** for integrity and consistency.
- **Volume** 
- **Price Movements** for extreme or suspicous patterns.
- **Data Completeness**
The second class is responsible for validating the features after feature generation.



# About features/
The unified feature engineering system that creates sophisticated technical indicators and features with prediction power to help ML models train and predict effectively.

## Unified Feature Generation System (`enhanced_generation.py`)
**ModelAwareFeatureGenerator** - A complete, production-ready feature system providing 200+ technical indicators and features:

### Feature Categories
- **Price Action Features**: Moving averages (SMA/EMA), Rate of Change, Bollinger Bands, Support/Resistance levels, Average True Range (ATR)
- **Volume Analysis**: Volume moving averages, Volume Rate of Change, On Balance Volume, Volume Profile with key metrics, Money Flow Index, VWAP
- **Technical Indicators**: RSI, MACD, ADX, Stochastic Oscillator, Commodity Channel Index, Ichimoku Cloud
- **Temporal Features**: Time of day/week/month (cyclical), Day of week patterns, Holiday detection, FX trading session analysis
- **Market Regime Analysis**: Volatility regimes, Trend strength indicators, Market state classification, Combined regime analysis
- **Advanced Analytics**: Seasonal decomposition (STL, MSTL), Fast Fourier Transform cyclical components, Logarithmic returns
- **🆕 Spread Compression Features**: True bid-ask spread analysis, compression detection, spread volatility, and market microstructure features

### Key Architectural Features
- **Model-Aware Generation**: Automatically generates exactly what trained models require
- **Automatic Dependency Resolution**: Handles complex feature dependencies intelligently  
- **🆕 Intelligent Feature Filtering**: Respects `selected_features` configuration - generates only requested features, preventing feature bloat
- **🆕 Spread-Focused Processing**: Optional bid-ask spread data preservation via `spread_focused` configuration
- **Decorator-Based Registration**: Easy feature addition with `@feature_generator` decorator
- **Robust Error Handling**: Graceful failure handling with detailed logging
- **Memory Efficient**: Optimized DataFrame operations to prevent fragmentation
- **Extensible Design**: Simple to add new features without breaking existing functionality

### Usage Examples

#### For Training Data (Historical Processing)
```python
from data.features.enhanced_generation import ModelAwareFeatureGenerator

# Initialize with data
generator = ModelAwareFeatureGenerator(historical_data)

# 🆕 Generate specific features (with intelligent filtering)
selected_features = ['london_session', 'compression_ratio', 'volume_ma_20', 'mid_price']
processed_data = generator.generate_selected(selected_features)
# Result: Only generates the 4 requested features + essential columns
# Filters out unwanted features from generators that produce multiple outputs

# 🆕 Feature group expansion support
spread_features = ['bid_ask_spread_features', 'spread_compression_features'] 
processed_data = generator.generate_selected(spread_features)
# Automatically expands groups to individual features

# Or generate all features (200+ features)
all_features_data = generator.generate_all()
```

#### For Live Trading (Model-Aware)
```python
# Generate features automatically based on model requirements
generator = ModelAwareFeatureGenerator(live_data) 
processed_data = generator.generate_for_model("storage/models/2025-07-03_12:45:35/model_metadata.yml")
```

#### 🆕 Spread-Focused Processing (Bid-Ask Analysis)
```python
# Enable spread-focused processing in hypothesis config
data_pipeline:
  spread_focused: true  # Preserves bid-ask spread data
  selected_features: 
    - "compression_ratio"
    - "bid_ask_spread_features"  # Expands to actual spread features
    - "london_session"

# Results in focused feature set for spread compression analysis:
# - compression_ratio, spread_relative, bid_ask_spread_bps
# - london_session (no other session features)
# - Essential OHLCV + targets preserved
# - Raw spread data filtered unless explicitly requested
```

#### Adding New Features
```python
@feature_generator(
    output_features=['my_custom_indicator'],
    dependencies=['sma_20', 'atr'],  # Automatic dependency handling
    description="My custom technical indicator",
    category="custom"
)
def my_custom_feature(self):
    """Calculate custom indicator"""
    self.df['my_custom_indicator'] = (self.df['sma_20'] / self.df['atr']) * 100
```

### Feature Registry System (`feature_registry.py`)
The backbone of the feature system providing:
- **Automatic Registration**: All decorated features auto-register on import
- **Dependency Graph**: Maintains and resolves complex feature dependencies
- **Feature Discovery**: Dynamic listing of available features and generators
- **Execution Planning**: Topological sorting for optimal generation order
- **Metadata Management**: Rich feature descriptions, categories, and parameters

### 🆕 Intelligent Feature Filtering System
The system now provides precise control over feature generation:

#### How It Works
1. **Generator Execution**: Runs required generators based on dependencies
2. **Post-Generation Filtering**: Keeps only explicitly requested features
3. **Essential Preservation**: Always preserves OHLCV, targets, and structure columns
4. **Spread Data Control**: Filters raw spread data unless explicitly requested

#### Example: Session Feature Filtering
```python
# Request: ['london_session']  
# Generator runs: fx_trading_sessions (produces 7 session features)
# Filter keeps: london_session only (filters out 6 unwanted features)

# Before filtering: 57 features (feature bloat)
# After filtering: 37 features (focused dataset)
```

#### Automatic Feature Group Expansion
```python
# Config: "bid_ask_spread_features" 
# Expands to: ['bid_ask_spread_bps', 'bid_ask_spread_relative']
# Enables logical grouping in hypothesis configurations
```

### Integration Points
The system seamlessly integrates with both:
- **Engineer Pipeline**: For historical data processing and model training
- **Live Data Manager**: For real-time feature generation during strategy execution  
- **Strategy System**: Direct feature generation based on model requirements

Both pipelines use the exact same feature generation system ensuring perfect consistency between training and live trading environments.

## Transformation - transformation.py
### Preparation of features before normalization
- **Handling missing values**: Some steps to handle missing values, as this can mess up normalization and selection later.
  1. Dropping whole features (technically columns) that contain more empty values (```Null```, ```NaN```) than the threshold (default is 50%).
  2. Based on a fixed/static list with feature names, extracts the maximum window size (eg. ```n```) and drops the first ```n``` rows. This can also be fixed (TODO).
    - **<ins>Note</ins>**: This fix _assumes_ all generated features (that requires past data to calculate _and_ the features manually added in the ```high_feature_windows``` list) starts with the abbreviation or one-word of the feature name, followed by "_" and the window size. Eg. For Simple Moving Average with window size 50: ```sma_50```.
- **Handles duplicated features**/columns.

### Normalization
Simple Normalization class that uses **two Normalization Techniques**; [Z-Score](https://www.investopedia.com/terms/z/zscore.asp) for _Unbounded_ (no boundaries of certainty) features and [MinMax Scaker](https://medium.com/@iamkamleshrangi/how-min-max-scaler-works-9fbebb9347da) for _Bounded_ (field that has limited scope) features.<br>
In the `z_score` method; original close price/value is preserved before normalization as default, called `close_og`.<br>
The ```run()``` has some functionality that excludes some features from normalization:
- **Features with prefix (starting with) `target_`**, as this is how target hypothesis columns are calculated and named.
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

## Creating Target Column (for the Hypothesis) - target_hypothesis.py
If the pipeline is being ran with ```hypothesis``` argument (path to `config.yml`), the `Feature Selection` process is skipped and this (file) is ran after `Feature Generation`.<br>
It's basically calculating and setting target column to the hypothesis: In this ```taget_hypothesis.py```-file, based on the `name` of the hypothesis in the ```config.yml```-file.<br>
The approach is to use `shared_utils.list_of_functions_in_class` to dynamically list all method names in the class. Then, we run the **method name that matches with the hypothesis name** - meaning it's **important** they match!<br>
For some more background and format, see the ```README.md``` documentation under the ```hypothesis/``` folder.
> [!IMPORTANT]
> When adding functionality for new hypotheses: Remember to add the prefix ```target_*``` for the _target column to the hypothesis_. This is because the normalization step after generation and target hypothesis will _not normalize_ columns that starts with ```target_```.

## Feature Selection - selection.py
The Selector file performs _very simple_ Feature Selection, by using different techniques/methods, to then select the features that two or more techniques agree about:
1. **Correlation-based Feature Selection**: Selects subsets of features that are highly correlated with the target variable (```close```), but have low correlation with each other.
2. **XGB Regression**: "Ranks" features based on importance, and selects features above ```threshold```-variable.
3. **Recursive Feature Elimination**: Fits  a model (XGB Regressor) and removes the weakest feature (or features) until the specified number of features is reached.
> [!NOTE]
> Before the Selector returns the Features it found most predictive, I have added a manual step for preserving the `close_og` column, meaning the close price value without normalization. This is important to have in the processed file, as the model/strategy will later use this to place and execute trades.



# About providers/ 
Files under this are used solely used for fetching raw historical data. This data is intended to engineer and used for model training.<br>
Default behaviour is to store the files under storage/data and under the respective provider.

## Capital.com
Only uses the broker/capitalcom.py implementation to only get historical data. For more information on the broker itself, see broker/ documentation.<br>
The broker itself has relatively restricted with historical data, especially for 1min intraday timeframe, but useful in early stages. 5min has a little more.
### How to use
Only one way to use this, no matter timeframe, symbol, etc. Example:
```python
from data.providers.capital_com import ProviderCapitalCom

provider = ProviderCapitalCom()
provider.fetch_and_save_historical_data(symbol="GBPUSD", timeframe="MINUTE_5",
                                        from_date="2024-04-15T00:00:00", to_date="2025-05-01T01:00:00",
                                        print_answer=False)
```
### Useful links:
- [API documentation](https://open-api.capital.com/)
- [Requirements to use their API](https://open-api.capital.com/#section/Getting-started)

## Alpha Vantage
Different restrictions on free api token, but seems quite good when you have the paid subscription.<br>
Currently (for this use), intraday can be used for (eg.) "core stocks", but not intraday Forex as it's not a part of free plan.<br>
This is also why methods like "saving the name of raw datafile" is very proprietary and can be seen as "not finished", as I'm planning to use this provider 
(or someone else) when this project has come further down.
### How to use
This one is a little different. Current implementation is only tested, and implemented, with free api key and used on "Core Stock APIs". So if you're using this on forex or others, you will need to change fetch_historical_data() according to the selected "type of API" you need. Documentation is fairly good, so should be okay.<br>
Example of current use:
```python
from data.providers.alpha_vantage import ProviderAlphaVantage

provider = ProviderAlphaVantage()
provider.fetch_and_save_historical_data(symbol="NVDA", timeframe="1min",
                                        month="2025-04", print_answer=False,
                                        store_answer=True)
```
### Useful links:
- [API documentation](https://www.alphavantage.co/documentation/)
- [Get free API token](https://www.alphavantage.co/support/#api-key)
- [Premium API key](https://www.alphavantage.co/premium/) for actually utilizing the providers' possibilities large datasets.



# About plot/ 📊
Simple functionality to plot candlestick data. Can both be run from main.py, but also used in Data Pipeline, after cleaning.<br>
Currently only using have to methods for using either ```plotly.graph_objects``` or ```mplfinance```.
## How to use
Recommended usage is either plotting raw data, or after cleaning the data (eg. in the Data Pipeline), as the final step in the Data Pipeline is normalization.<br>
To simply plot candlestick data:
```bash
python main.py --plot-data 'storage/data/capital_com/GBPUSD/raw/1m/2025-05-15_2025-05-23.csv'
```
## Challenges
- Plotting 1-minute candles the **datasets can get very large**, making the plotting and interaction _slow_, and also incredibly _difficult to see what's going on_. Current solution is a little improved ```plotly.graph_objects``` implementation in the method ```simple_small_go_candle_plot()```. Though this can easily be even more improved. Perhaps splitting up the plotting?
- **Plotting engineered data**: As most of the data has been _normalized_ (even ohlcv), I can't really check to see if the cleaning was done correctly. Solution is to _plot and save dataset after cleaning_ it, as part of the Data Pipeline.



# Live Data Management 📺

## LiveDataManager (`live/live_data_manager.py`)
The core component for real-time data streaming and processing:

### Key Features
- **Real-time Data Streaming**: WebSocket and REST API integration for live market data
- **Rolling Data Buffer**: Maintains configurable buffer of recent OHLCV data (default: 500 data points)
- **Feature Pipeline Integration**: Reuses existing cleaning → validation → feature generation pipeline
- **Thread-safe Operations**: Proper synchronization for concurrent data processing
- **Graceful Error Handling**: Automatic reconnection and error recovery
- **Strategy Integration**: Direct integration with trading strategies
- **Jumpstart Capability**: Preload buffer with historical data for immediate feature generation

### How It Works
```python
# Initialize live data manager
live_manager = LiveDataManager(
    symbol="GBPUSD",
    timeframe="MINUTE", 
    buffer_size=500,
    feature_window=200,
    use_websocket=True,
    jumpstart=True
)

# Start data streaming
live_manager.start()

# Get latest processed data with features
latest_data = live_manager.get_latest_data()
latest_features = live_manager.get_latest_features()

# Stop streaming
live_manager.stop()
```

### Configuration Options
- **symbol**: Trading pair (e.g., "GBPUSD", "BTCUSD")
- **timeframe**: Data resolution ("MINUTE", "MINUTE_5", "HOUR")
- **buffer_size**: Maximum data points to maintain in memory
- **feature_window**: Minimum points needed for feature calculation
- **use_websocket**: Enable WebSocket streaming (faster) vs REST polling
- **jumpstart**: Preload buffer with historical data on startup
- **selected_features**: Specific features to generate (None = all features)

### Integration with Strategy System
The LiveDataManager seamlessly integrates with the strategy system:
```python
# Inside strategy execution
market_data = self.live_data_manager.get_latest_data()
if market_data is not None:
    # Generate predictions
    prediction = self.model.predict(market_data)
    # Process signals and execute trades
    self.process_signals(prediction, market_data)
```

## WebSocket Data Manager (`live/websocket_data_manager.py`)
Specialized WebSocket handler for Capital.com real-time data:
- **Connection Management**: Automatic connection/reconnection handling  
- **Data Parsing**: Real-time parsing of WebSocket market data
- **Error Resilience**: Robust error handling and recovery mechanisms
- **Integration Ready**: Direct integration with LiveDataManager