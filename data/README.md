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
- [Premium API key](https://www.alphavantage.co/premium/) for actually utilizing the providers large datasets.