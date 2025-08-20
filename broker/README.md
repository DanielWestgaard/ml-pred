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

# Alpaca Integration 🦙

## Overview
Alpaca integration provides commission-free stock trading with a comprehensive REST API. The implementation supports both paper and live trading environments.

## Features ✨
- **Paper & Live Trading**: Switch between environments
- **Complete Order Management**: Market, limit, stop, and bracket orders
- **Position Management**: Position tracking and closure capabilities
- **Account Information**: Real-time account balance and buying power
- **Bulk Operations**: Cancel all orders and close all positions
- **Historical Data**: Complete REST API for historical bars, quotes, and trades
- **Live Data Streaming**: Real-time WebSocket streaming for trades, quotes, and bars
- **Data Storage**: Automatic saving to organized CSV files with proper directory structure
- **Error Handling**: Comprehensive error handling and logging
- **Type Safety**: Full type annotations for better development experience

## Architecture
```
alpaca/
├── alpaca.py                    # Main Alpaca broker orchestrator (BaseBroker interface)
├── tradin_api/
│   └── trading_api.py           # Complete Trading API implementation
└── market_data_api/
    ├── historical_data.py       # Historical data REST API with saving capabilities
    └── live_data.py             # Real-time WebSocket streaming API
```

## Quick Start 🚀
```python
from broker.alpaca.alpaca import Alpaca

# Initialize (uses paper trading by default)
alpaca = Alpaca(print_responses=True)

# Get account info
account = alpaca.get_account_balance()
print(f"Buying power: ${account['buying_power']}")

# Place a market order
order = alpaca.place_market_order(
    symbol="AAPL",
    side="buy", 
    quantity=10
)

# Check positions
positions = alpaca.get_positions()

# Get historical data
df = alpaca.get_historical_data("AAPL", "1d", start_date="2024-01-01T00:00:00")

# Save historical data to CSV
saved_file = alpaca.fetch_and_save_basic_plan_data("AAPL", "1d", days_back=30)
```

## Order Types & Examples 📊

### Market Orders
```python
# Simple market order
order = alpaca.place_market_order("AAPL", "buy", 10)

# Market order with bracket (stop loss + take profit)
order = alpaca.place_market_order(
    symbol="TSLA",
    side="buy",
    quantity=5,
    take_profit=250.0,
    stop_loss=200.0
)
```

### Limit Orders
```python
# Simple limit order
order = alpaca.place_limit_order("NVDA", "buy", 10, price=500.0)

# Limit order with bracket
order = alpaca.place_limit_order(
    symbol="MSFT",
    side="buy", 
    quantity=20,
    price=300.0,
    take_profit=320.0,
    stop_loss=280.0
)
```

### Advanced Orders
```python
# Trailing stop order
order = alpaca.trading_api.create_an_order(
    symbol="SPY",
    side="sell",
    qty=100,
    order_type="trailing_stop",
    trail_percent=2.0  # 2% trailing stop
)

# Good-til-canceled limit order
order = alpaca.trading_api.create_an_order(
    symbol="QQQ",
    side="buy",
    qty=50,
    order_type="limit",
    limit_price=350.0,
    time_in_force="gtc"
)
```

## Historical Data & Data Storage 📊

### Fetching Historical Data
```python
from datetime import datetime, timedelta

# Get historical data as DataFrame
df = alpaca.get_historical_data(
    "AAPL", 
    timeframe="1d", 
    start_date="2024-01-01T00:00:00",
    end_date="2024-12-31T00:00:00"
)

# Get data optimized for basic plan (avoids recent data restrictions)
historical_data = alpaca.historical_data_api.get_historical_bars_basic_plan(
    ["AAPL"], 
    timeframe="1Day", 
    days_back=30
)

# Convert to DataFrame
df = alpaca.historical_data_api.bars_to_dataframe(historical_data, "AAPL")
```

### Saving Data to CSV Files
```python
# Basic plan optimized saving (recommended for free accounts)
saved_file = alpaca.fetch_and_save_basic_plan_data(
    symbol="AAPL",
    timeframe="1d",
    days_back=30,
    save_raw_data=True  # Also saves raw JSON
)
# Result: storage/data/alpaca/AAPL/raw/1d/2025-08-04_2025-08-12.csv

# Save specific date range
saved_file = alpaca.fetch_and_save_historical_data(
    symbol="TSLA",
    timeframe="1h", 
    start_date="2024-08-01T00:00:00",
    end_date="2024-08-10T00:00:00"
)
# Result: storage/data/alpaca/TSLA/raw/1h/2024-08-01_2024-08-10.csv

# Batch save multiple symbols
saved_files = alpaca.historical_data_api.fetch_and_save_multiple_symbols(
    symbols=["AAPL", "MSFT", "TSLA"],
    timeframe="1Day",
    start="2024-01-01T00:00:00Z",
    end="2024-12-31T00:00:00Z"
)
```

### Data Format
Saved CSV files include all OHLCV data plus Alpaca-specific fields:
```csv
Date,Open,High,Low,Close,Volume,Trade_Count,VWAP
2024-08-01 04:00:00,227.69,231.867,214.3328,216.86,83861898,1133352,221.946165
```

### Directory Structure
Data is organized following the same pattern as Capital.com:
```
storage/data/alpaca/
├── AAPL/raw/1d/2025-08-04_2025-08-12.csv
├── MSFT/raw/1h/2025-08-07_2025-08-12.csv  
└── TSLA/raw/1d/2024-08-01_2024-08-09.csv
```

## Live Data Streaming 📡

### WebSocket Streaming
```python
# Simple streaming setup
def on_trade(message):
    symbol = message.get("S")
    price = message.get("p")
    print(f"Trade: {symbol} at ${price}")

def on_quote(message):
    symbol = message.get("S")
    bid = message.get("bp")
    ask = message.get("ap")
    print(f"Quote: {symbol} Bid ${bid} Ask ${ask}")

# Start streaming
alpaca.start_live_stream(
    symbols=["AAPL", "TSLA"], 
    data_types=["trades", "quotes", "bars"],
    trade_handler=on_trade,
    quote_handler=on_quote
)

# Stop streaming
alpaca.stop_live_stream()
```

### Advanced Streaming
```python
# Direct API access for custom handlers
live_api = alpaca.live_data_api

# Connect and authenticate
live_api.connect()

# Add custom handlers
live_api.add_handler("trades", custom_trade_handler)
live_api.add_handler("quotes", custom_quote_handler)

# Subscribe to specific data
live_api.subscribe_trades(["AAPL", "MSFT"])
live_api.subscribe_quotes(["AAPL", "MSFT"]) 
live_api.subscribe_bars(["AAPL", "MSFT"])

# Switch to premium feed (requires subscription)
live_api.set_premium_feed(premium=True)
```

## Portfolio Management 💼

### Position Management
```python
# Get all positions
positions = alpaca.get_positions()

# Get specific position
position = alpaca.trading_api.get_position("AAPL")

# Close specific position (partial)
result = alpaca.trading_api.close_position("AAPL", qty="5")

# Close entire position
result = alpaca.trading_api.close_position("AAPL")

# Close all positions (emergency stop)
result = alpaca.close_all_positions()
print(f"Closed {result['total_closed']} positions")
```

### Order Management
```python
# Get all orders
orders = alpaca.get_orders()

# Get open orders only
open_orders = alpaca.trading_api.get_all_orders(status="open")

# Cancel specific order
success = alpaca.close_order(order_id)

# Cancel all orders (emergency stop)
result = alpaca.cancel_all_orders()
print(f"Cancelled {result['total_cancelled']} orders")
```

## Environment Management 🔄

### Paper vs Live Trading
```python
# Initialize in paper mode (default)
alpaca = Alpaca()

# Switch to live trading (⚠️ USE WITH CAUTION)
alpaca.trading_api.set_live_trading(live=True)

# Switch back to paper trading
alpaca.trading_api.set_live_trading(live=False)
```

### Configuration
Ensure your secrets file contains:
```json
{
    "alpaca_paper_account_key_id": "your_paper_key_id",
    "alpaca_paper_account_secret_key": "your_paper_secret_key"
}
```

## Strategy Integration Example 🎯
```python
from broker.alpaca.alpaca import Alpaca

class AlpacaStrategy:
    def __init__(self):
        self.broker = Alpaca(print_responses=False)
        
    def execute_signal(self, signal):
        """Execute trading signal with risk management"""
        
        # Calculate position size based on account balance
        account = self.broker.get_account_balance()
        buying_power = float(account['buying_power'])
        position_size = min(signal['quantity'], buying_power // signal['price'])
        
        if signal['type'] == 'market':
            order = self.broker.place_market_order(
                symbol=signal['symbol'],
                side=signal['side'],
                quantity=position_size,
                take_profit=signal.get('take_profit'),
                stop_loss=signal.get('stop_loss')
            )
        
        return order
    
    def emergency_stop(self):
        """Emergency portfolio shutdown"""
        # Cancel all pending orders
        cancel_result = self.broker.cancel_all_orders()
        
        # Close all positions
        close_result = self.broker.close_all_positions()
        
        return {
            'orders_cancelled': cancel_result['total_cancelled'],
            'positions_closed': close_result['total_closed']
        }
```

## Error Handling 🛡️
All methods return structured responses with error information:
```python
# Check for errors in responses
order = alpaca.place_market_order("AAPL", "buy", 10)

if "error" in order:
    print(f"Order failed: {order['error']}")
    print(f"Status code: {order.get('status_code', 'unknown')}")
else:
    print(f"Order placed successfully: {order['id']}")
```

## API Methods Reference 📖

### Account Methods
- `get_account_balance()` - Get account information
- `get_positions()` - Get all positions  
- `get_orders()` - Get all orders

### Trading Methods
- `place_market_order()` - Place market order
- `place_limit_order()` - Place limit order
- `close_order()` - Cancel order
- `cancel_all_orders()` - Cancel all orders
- `close_all_positions()` - Close all positions

### Historical Data Methods
- `get_historical_data()` - Get historical data as DataFrame
- `fetch_and_save_historical_data()` - Fetch and save specific date range
- `fetch_and_save_basic_plan_data()` - Fetch and save (basic plan optimized)
- `get_stock_bars()` - Raw historical bars API
- `get_stock_quotes()` - Historical quotes data
- `get_stock_trades()` - Historical trades data
- `get_snapshots()` - Current market snapshots

### Live Data Methods  
- `start_live_stream()` - Start real-time data streaming
- `stop_live_stream()` - Stop streaming and disconnect
- `add_live_data_handler()` - Add custom message handlers
- `sub_live_market_data()` - Subscribe to live data for symbol

### Advanced APIs
- `trading_api.create_an_order()` - Full order creation with all parameters
- `trading_api.get_all_orders()` - Get orders with filtering
- `trading_api.set_live_trading()` - Switch trading environment
- `historical_data_api.*` - Direct access to all historical data methods
- `live_data_api.*` - Direct access to all live streaming methods

## Status & Roadmap 🗺️
- ✅ **Complete Trading API**: All order types and position management
- ✅ **Error Handling**: Comprehensive error handling and logging  
- ✅ **Paper Trading**: Full paper trading support
- ✅ **Live Trading**: Production-ready live trading
- ✅ **Historical Data API**: Complete REST API with CSV saving capabilities
- ✅ **Live Data Streaming**: Real-time WebSocket streaming (trades, quotes, bars)
- ✅ **Data Storage**: Organized CSV files with proper directory structure
- ✅ **Basic Plan Support**: Optimized methods for free tier limitations
- ⏳ **Options Trading**: Options trading support (planned)
- ⏳ **Crypto Trading**: Cryptocurrency trading (planned)
- ⏳ **Advanced Charting**: Integration with plotting libraries (planned)

## Useful Links - Alpaca 🔗
- [Alpaca API Documentation](https://docs.alpaca.markets/)
- [Trading API Reference](https://docs.alpaca.markets/reference/postorder)
- [Paper Trading Setup](https://docs.alpaca.markets/docs/trading/paper-trading)
- [Account Dashboard](https://app.alpaca.markets/)
- [API Authentication](https://docs.alpaca.markets/reference/authentication-2)

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