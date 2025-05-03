# About broker/

## Structure
```
├── base_interface.py       # Interface that the main broker classes implements/uses
├── README.md               # Documentation related to broker/
├── capital_com/            # Capital.com API integration
│   └── capital_com.py/     # Main orchestrator file used to interact with different endpoints of API
│       ├── rest_api/       # REST API functionality. Each class file follows the endpoints from official doc
│       │   └─ ...
│       └── web_socket/     # Web Socket functionality (streaming)
└── ...
```

# Useful info about the different brokers
## Capital.com (API)
Capital.com is currently the only supported broker.<br>
The reason for this choice is the large amount of symbols to pick from, low costs, good UI, and okay API and documentation.
### How to use
#### Initial start and end (session)
```python
from broker.capital_com.capitalcom import CapitalCom

broker = CapitalCom()
broker.start_session()
# Do different stuff between here. When done, close session:
broker.end_session()
```
#### Account related info
```python
broker.switch_active_account(print_answer=False)  # Default switch is constant in config

broker.get_account_capital()  # Must switch active account before running

broker.list_all_accounts(print_answer=True)

# Not account related, but session related
broker.session_details(print_answer=True)
```
#### Data
```python
# Simple fetch, max 1000 values
broker.get_historical_data(epic="GBPUSD", resolution="MINUTE",
                            max=1000,
                            from_date="2025-04-10T12:00:00", to_date="2025-04-10T13:10:00",
                            print_answer=True)

# Fetches and saves data from as long as you back as you want (and capital.com has data)
broker.fetch_and_save_historical_prices(epic="GBPUSD", resolution="MINUTE",
                                        from_date="2025-04-15T00:00:00", to_date="2025-05-01T01:00:00",
                                        print_answer=False)

# Using Websocket, but if you're only using this alone you need to comment out the "simple hack loop", or else it will just stop
broker.sub_live_market_data(symbol="GBPUSD", timeframe="MINUTE")
```
#### Orders
```python
# Has both *_level (price level) and *_price (dollar price) for SL and TP
broker.place_market_order(symbol="GBPUSD", direction="BUY", size="100", stop_level="1.32", profit_level="1.34")

# Modify an active position
broker.modify_position(stop_level="1.25", profit_level="1.29")

# List out all acitve positions
broker.all_positions()

# Close out all positions (currently not added functionality to specify a single one)
broker.close_all_orders(print_answer=True)
```
### Useful links:
- [API documentation](https://open-api.capital.com/)
- [Requirements to use their API](https://open-api.capital.com/#section/Getting-started)