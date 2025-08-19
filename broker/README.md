# About broker/ 📊

## Overview
The broker component provides a **unified interface for trading operations** across different brokers, with comprehensive support for both historical data retrieval and live trading. The system is designed to integrate seamlessly with the strategy execution engine, providing reliable order execution, position management, and real-time market data streaming.

## Key Features 🚀
- **Multi-Broker Support**: Extensible architecture supporting multiple broker integrations
- **Live Trading Integration**: Direct integration with strategy execution system
- **Real-time Data Streaming**: WebSocket and REST API support for live market data
- **Order Management**: Complete order lifecycle management (placement, modification, closure)
- **Centralized Account Management**: Automatic account switching with configurable target accounts
- **Intelligent Position Sizing**: Currency-specific size calculations (e.g., BTCUSD decimal sizing)
- **Live Trading Safety**: Automatic position closure during system shutdown
- **Historical Data**: Comprehensive historical data retrieval for backtesting and analysis

## Structure
```
broker/
├── base_interface.py               # Abstract broker interface for consistency
├── README.md                       # Documentation
│
├── capital_com/                    # Capital.com broker integration
│   ├── capitalcom.py               # Main CapitalCom broker orchestrator
│   │
│   ├── rest_api/                   # REST API implementation
│   │   ├── session.py              # Session management (login/logout with encryption)
│   │   ├── account.py              # Account operations and capital management
│   │   ├── markets_info.py         # Market data, historical prices, symbol info
│   │   └── trading.py              # Order placement, position management, trade execution
│   │
│   └── web_socket/                 # Real-time data streaming
│       └── web_socket.py           # WebSocket market data streaming with connection management
│
└── alpaca/                         # Alpaca broker integration
    ├── alpaca.py                   # Main Alpaca broker orchestrator
    │
    ├── trading_api/                # Trading API endpoint (stock trading for individuals)
    │   └── ...
    ├── market_data_api/            # Real-time Market pricing data and historical data. TODO: Not done yet
    │   └── ...
    ├── options_trading_api/        # Options trading API. TODO: Not done yet
    │   └── ...
    ├── broker_api/                 # For building trading apps and services for end users. TODO: Not done yet
    │   └── ...

```

# Alpaca Integration
Curently a work in progress.<br>
Link to [Alpaca Market Docs](https://docs.alpaca.markets/).<br>
Since Alpaca provides several different types of API's, I think I'll use the main `alpaca.py` class as "main method/orchestrator for running the other ones".

# Capital.com Integration

## Why Capital.com? 📈
Capital.com is currently the primary supported broker, chosen for:
- **Wide Symbol Coverage**: Large selection of forex, commodities, and indices
- **Low Trading Costs**: Competitive spreads and no commission trading
- **Reliable API**: Well-documented REST and WebSocket APIs
- **Real-time Data**: Sub-second market data streaming capabilities
- **Integration Ready**: Full compatibility with strategy execution system

## Strategy Integration 🔄
The broker component is designed to integrate seamlessly with the strategy system:

### Order Manager Integration
```python
# Inside strategy components (OrderManager)
from broker.capital_com.capitalcom import CapitalCom

class OrderManager:
    def __init__(self):
        self.broker = CapitalCom()
        self.broker.start_session()  # Auto-switches to USD_testing account
    
    def execute_order(self, signal):
        # Execute trades with intelligent sizing (e.g., BTCUSD = 0.01 BTC minimum)
        result = self.broker.place_market_order(
            symbol=signal['symbol'],
            direction=signal['direction'],
            size=self._calculate_size(signal),  # Intelligent sizing
            stop_level=signal.get('stop_loss'),
            profit_level=signal.get('take_profit')
        )
        return result
    
    def cleanup(self):
        # Safe cleanup - closes all positions in live mode
        self.broker.close_all_orders()  # Live trading safety
        self.broker.end_session()
```

### Live Data Integration
```python
# Integration with LiveDataManager
from data.live.websocket_data_manager import WebSocketDataManager

# WebSocket streaming for real-time strategy execution  
websocket_manager = WebSocketDataManager()
websocket_manager.connect(symbol="GBPUSD")
```

## Usage Examples

### Basic Broker Operations
```python
from broker.capital_com.capitalcom import CapitalCom

# Initialize and start session (automatically switches to USD_testing account)
broker = CapitalCom()
broker.start_session()  # Auto-switches to configured account

# Account management (optional - already handled by start_session)
capital_info = broker.get_account_capital()

# Place orders with risk management
broker.place_market_order(
    symbol="GBPUSD", 
    direction="BUY", 
    size="100",
    stop_level="1.32",  # Stop loss
    profit_level="1.34"  # Take profit
)

# Monitor and manage positions
positions = broker.all_positions()
broker.modify_position(stop_level="1.30", profit_level="1.35")

# Close session
broker.end_session()
```

### Historical Data Retrieval
```python
# For data engineering (recommended: use data/providers/capital_com.py)
broker.fetch_and_save_historical_prices(
    epic="GBPUSD", 
    resolution="MINUTE",
    from_date="2025-04-15T00:00:00", 
    to_date="2025-05-01T01:00:00"
)

# Quick data fetch (max 1000 points)
data = broker.get_historical_data(
    epic="GBPUSD", 
    resolution="MINUTE",
    max=1000,
    from_date="2025-04-10T12:00:00", 
    to_date="2025-04-10T13:10:00"
)
```

### Real-time Data Streaming
```python
# WebSocket streaming for live strategies
broker.sub_live_market_data(symbol="GBPUSD", timeframe="MINUTE")
# Note: Integrate with LiveDataManager for production use
```

## Configuration

### Account Management
```python
# In config/config.py
VAR_ACCOUNT_NAME_TEST = "USD_testing"  # Target account for all sessions
```

### Strategy Configuration
Broker configuration through hypothesis config files:
```yml
strategy:
  order_management:
    broker: capital_com
    order_type: market
    execution_delay: 0.1
    dry_run: false  # false = live trading, true = simulation mode
  
  risk_management:
    max_position_size: 0.02  # 2% of capital per trade
    stop_loss_pct: 0.1       # 10% stop loss
    max_drawdown: 0.05       # 5% maximum drawdown
```

## Advanced Features 🔧

### Centralized Account Management
All broker sessions automatically switch to your configured account:

```python
# Every start_session() call automatically switches to USD_testing
broker = CapitalCom()
broker.start_session()  # Auto-switches to config.VAR_ACCOUNT_NAME_TEST

# Override account if needed
broker.start_session(account_name="USD_live")
```

### Intelligent Position Sizing
- **BTCUSD**: Minimum 0.01 BTC with decimal precision (e.g., 0.0125 BTC)
- **Forex Pairs**: Integer lot calculations (e.g., 1000 units)
- **Risk-Based**: Sizes calculated based on USD amount and current price

### Live Trading Safety
```python
# Automatic position closure during shutdown (live mode only)
order_manager.cleanup()  # Closes all positions before ending session
```
## Useful links - capital.com:
- [API documentation](https://open-api.capital.com/)
- [Requirements to use their API](https://open-api.capital.com/#section/Getting-started)