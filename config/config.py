import logging
import os


# VARIABLES
PRICE_DECIMALS = 5
# TEST
VAR_ACCOUNT_NAME_TEST = "USD_testing"
# ENVIRONMENTS
DEV_ENV = "development"
TEST_ENV = "test"
PROD_ENV = "production"

# URL's
CAPCOM_LIVE_URL = "api-capital.backend-capital.com"  # Live account
CAPCOM_DEMO_URL = "demo-api-capital.backend-capital.com"  # Demo account
CAPCOM_WEBSOCKET_URL = "wss://api-streaming-capital.backend-capital.com/connect"


# DIRECTORIES
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ml-pred/
BASE_STORAGE_DIR = os.path.join(BASE_DIR, 'storage')
BASE_DATA_STORAGE_DIR = os.path.join(BASE_STORAGE_DIR, 'data')
# Logging dirs
DIFFERENT_LOG_DIRS = ['training', 'data', 'trash', "trades", "performance", "backtesting", "strategy"]
BASE_LOGS_DIR = os.path.join(BASE_DIR, 'logs')
# Capital.com API
CAPCOM_BASE_STORAGE_DIR = os.path.join(BASE_DATA_STORAGE_DIR, 'capital_com')
CAPCOM_RESPONSE_JSON_DIR = os.path.join(CAPCOM_BASE_STORAGE_DIR, 'saved_responses')

# Helper functions for symbol-based storage structure
def get_capcom_raw_data_dir(symbol: str, timeframe: str = None) -> str:
    """Get raw data directory for a specific symbol and optional timeframe."""
    base_path = os.path.join(CAPCOM_BASE_STORAGE_DIR, symbol, 'raw')
    if timeframe:
        return os.path.join(base_path, timeframe)
    return base_path

def get_capcom_processed_data_dir(symbol: str, timeframe: str = None) -> str:
    """Get processed data directory for a specific symbol and optional timeframe."""
    base_path = os.path.join(CAPCOM_BASE_STORAGE_DIR, symbol, 'processed')
    if timeframe:
        return os.path.join(base_path, timeframe)
    return base_path

# Multi-Data Sources storage helper functions
def get_calendar_events_symbol_dir(symbol: str) -> str:
    """Get calendar events directory for a specific symbol."""
    return os.path.join(CALENDAR_EVENTS_BASE_DIR, symbol)

def get_calendar_events_component_dir(symbol: str, component: str) -> str:
    """Get specific calendar component directory for a symbol (earnings, options_expiration, cross_calendar)."""
    return os.path.join(get_calendar_events_symbol_dir(symbol), component)

def get_calendar_events_raw_dir(symbol: str, component: str) -> str:
    """Get raw data directory for a calendar component and symbol."""
    return os.path.join(get_calendar_events_component_dir(symbol, component), 'raw')

def get_calendar_events_analysis_dir(symbol: str, component: str) -> str:
    """Get analysis directory for a calendar component and symbol."""
    return os.path.join(get_calendar_events_component_dir(symbol, component), 'analysis')

def get_calendar_events_processed_dir(symbol: str, component: str) -> str:
    """Get processed directory for a calendar component and symbol."""
    return os.path.join(get_calendar_events_component_dir(symbol, component), 'processed')
# Alpaca API
ALPACA_BASE_STORAGE_DIR = os.path.join(BASE_DATA_STORAGE_DIR, 'alpaca')
ALPACA_RESPONSE_JSON_DIR = os.path.join(ALPACA_BASE_STORAGE_DIR, 'saved_responses')

# Helper functions for Alpaca symbol-based storage structure
def get_alpaca_raw_data_dir(symbol: str, timeframe: str = None) -> str:
    """Get raw data directory for a specific symbol and optional timeframe."""
    base_path = os.path.join(ALPACA_BASE_STORAGE_DIR, symbol, 'raw')
    if timeframe:
        return os.path.join(base_path, timeframe)
    return base_path

def get_alpaca_processed_data_dir(symbol: str, timeframe: str = None) -> str:
    """Get processed data directory for a specific symbol and optional timeframe."""
    base_path = os.path.join(ALPACA_BASE_STORAGE_DIR, symbol, 'processed')
    if timeframe:
        return os.path.join(base_path, timeframe)
    return base_path

# Alpha Vantage API
ALPVAN_BASE_STORAGE_DIR = os.path.join(BASE_DATA_STORAGE_DIR, 'alpha_vantage')
ALPVAN_RAW_DATA_DIR = os.path.join(ALPVAN_BASE_STORAGE_DIR, 'raw')
ALPVAN_RESPONSE_JSON_DIR = os.path.join(ALPVAN_BASE_STORAGE_DIR, 'saved_responses')
# Data Engineering
DATA_ENGINEERING_STATS_DIR = os.path.join(BASE_STORAGE_DIR, 'data_engineering_stats')
CLEANED_DATA_BASE_DIR = os.path.join(DATA_ENGINEERING_STATS_DIR, 'cleaned_data')
FE_SEL_BASE_DIR = os.path.join(DATA_ENGINEERING_STATS_DIR, 'fe_selection')
# ML Model
TRAINED_MODEL_STORAGE = os.path.join(BASE_STORAGE_DIR, 'models')

# Multi-Data Sources
MULTI_DATA_SOURCES_BASE_DIR = os.path.join(BASE_STORAGE_DIR, 'multi_data_sources')
CALENDAR_EVENTS_BASE_DIR = os.path.join(MULTI_DATA_SOURCES_BASE_DIR, 'calendar_events')

# LOGGING
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)-40.40s:%(lineno)-4d - %(levelname)-8s - %(message)s'
DEFAULT_DATE_FORMAT = '%d-%m-%Y %H:%M:%S'
DEFAULT_LOG_LEVEL = logging.INFO
DEBUG_LOG_LEVEL = logging.DEBUG

# Enhanced logging formats for different use cases
DETAILED_LOG_FORMAT = '%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(funcName)s() - %(message)s'
SIMPLE_LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
STRUCTURED_LOG_FORMAT = '%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s - %(extras)s'