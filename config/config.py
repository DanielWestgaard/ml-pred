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
CAPCOM_RAW_DATA_DIR = os.path.join(CAPCOM_BASE_STORAGE_DIR, 'raw')
CAPCOM_PROCESSED_DATA_DIR = os.path.join(CAPCOM_BASE_STORAGE_DIR, 'processed')
CAPCOM_RESPONSE_JSON_DIR = os.path.join(CAPCOM_BASE_STORAGE_DIR, 'saved_responses')
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

# LOGGING
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)-35.35s:%(lineno)-4d - %(levelname)-8s - %(message)s'
DEFAULT_DATE_FORMAT = '%d-%m-%Y %H:%M:%S'
DEFAULT_LOG_LEVEL = logging.INFO
DEBUG_LOG_LEVEL = logging.DEBUG

# Enhanced logging formats for different use cases
DETAILED_LOG_FORMAT = '%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(funcName)s() - %(message)s'
SIMPLE_LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
STRUCTURED_LOG_FORMAT = '%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s - %(extras)s'