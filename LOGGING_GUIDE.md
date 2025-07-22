# Centralized Logging Guide 📋

This guide explains how to use the improved centralized logging system implemented across the ml-pred project.

## Quick Start

### Basic Setup (Recommended)
```python
from utils.logging_utils import get_logger

logger = get_logger()  # Auto-detects your module name

# Use the logger
logger.info("Your message here")
```

### Alternative Setups
```python
# Explicit module name
logger = get_logger(__name__)

# Custom logger name
logger = get_logger('trading_system')
```

## Key Improvements Made

### 1. Enhanced Log Format ✨
**Before:** `2025-01-15 10:30:00 - INFO - Trade executed`
**After:** `2025-01-15 10:30:00 - strategy.components.order_manager:45 - INFO - Trade executed`

Now you can see exactly which file and line number each log comes from!

### 2. Centralized Configuration 🎯
All logging configuration is now in:
- **`config/config.py`** - Log formats and levels
- **`utils/logging_utils.py`** - Central logging setup

### 3. Auto-Module Detection 🔍
The `get_logger()` function automatically detects which module is calling it, so you don't need to pass `__name__` anymore.

### 4. Structured Logging for Context 📊
```python
from utils.logging_utils import log_with_context

log_with_context(logger, 'info', 'Trade executed', {
    'symbol': 'BTCUSD',
    'size': 0.1,
    'price': 45000.00,
    'direction': 'long'
})
```

## Log Levels Guidelines

| Level | When to Use | Example |
|-------|-------------|---------|
| `DEBUG` | Detailed diagnostic info, API requests/responses | `logger.debug(f"API response: {response_data}")` |
| `INFO` | General operational messages | `logger.info("Position opened successfully")` |
| `WARNING` | Unexpected but recoverable situations | `logger.warning("Retrying failed API call")` |
| `ERROR` | Error conditions that don't stop execution | `logger.error("Failed to close position, will retry")` |
| `CRITICAL` | System-breaking errors | `logger.critical("Database connection lost")` |

## Migration from Old Logging

### Replace Direct Logging Calls
```python
# OLD - Don't do this
import logging
logging.info("Message here")

# NEW - Do this  
from utils.logging_utils import get_logger
logger = get_logger()
logger.info("Message here")
```

### Replace Print Statements
```python
# OLD - Don't do this
print(f"Processing {count} records")

# NEW - Do this
logger.info(f"Processing {count} records")
```

### Replace Debug Prints
```python
# OLD - Don't do this  
if debug:
    print(json.dumps(api_response, indent=4))

# NEW - Do this
logger.debug(f"API response: {json.dumps(api_response, indent=4)}")
```

## Best Practices

### ✅ DO
- Use the centralized `get_logger()` utility
- Include context with `log_with_context()` for important events
- Use appropriate log levels
- Log before and after critical operations
- Include relevant data (deal IDs, symbols, amounts)

### ❌ DON'T  
- Use `logging.info()` directly
- Use `print()` statements in production code
- Log sensitive data (passwords, API keys)
- Over-log frequent events at INFO level (use DEBUG)
- Forget to import the logger utility

## Examples by Component

### Trading Operations
```python
logger.info("Opening position", extra={'symbol': 'BTCUSD', 'size': 0.1})
logger.error("Position creation failed", extra={'error_code': 'INSUFFICIENT_MARGIN'})
```

### Data Processing  
```python
logger.info("Starting data pipeline")
logger.debug(f"Processing batch {batch_num} of {total_batches}")
logger.info("Data pipeline completed", extra={'records_processed': 10000})
```

### WebSocket Events
```python
logger.info("WebSocket connection established")
logger.debug("Market data update received", extra={'symbol': 'BTCUSD', 'price': 45000})
logger.warning("Connection lost, reconnecting...")
```

### API Interactions
```python
logger.debug("Making API request to /api/v1/positions")
logger.debug("API response received", extra={'status_code': 200, 'response_time_ms': 150})
logger.error("API rate limit exceeded", extra={'retry_after': 60})
```

## Log Output Examples

With the new format, you'll see logs like:
```
2025-01-15 10:30:15 - broker.capital_com.rest_api.trading:101 - INFO - Deal Reference from confirmed position: ABC123
2025-01-15 10:30:16 - strategy.components.order_manager:67 - INFO - Position opened | symbol=BTCUSD size=0.1 price=45000.0
2025-01-15 10:30:17 - data.live.websocket_data_manager:234 - DEBUG - WebSocket message received: {"destination": "ohlc.event"}
```

## Configuration

### Available Log Formats
```python
# config/config.py
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s'
DETAILED_LOG_FORMAT = '%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(funcName)s() - %(message)s'  
SIMPLE_LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
```

### Log Directories
Logs are organized by component:
```
logs/
├── training/
├── data/
├── trades/
├── performance/
├── backtesting/
└── strategy/
```

## Troubleshooting

### Import Issues
If you get import errors:
```python
# Make sure you're importing from the right path
from utils.logging_utils import get_logger
```

### Multiple Logger Instances
The system prevents duplicate configuration, so it's safe to call `get_logger()` multiple times.

### Log Level Not Showing
Check your log level setting in `config/config.py`:
```python
DEFAULT_LOG_LEVEL = logging.DEBUG  # Shows all levels
DEFAULT_LOG_LEVEL = logging.INFO   # Shows INFO and above
```

---

For more examples, see `utils/logging_examples.py` 🚀