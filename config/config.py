import os


# VARIABLES
# ...
# TEST
VAR_ACCOUNT_NAME_TEST = "USD_testing"


# URL's
CAPCOM_LIVE_URL = "api-capital.backend-capital.com"  # Live account
CAPCOM_DEMO_URL = "demo-api-capital.backend-capital.com"  # Demo account
CAPCOM_WEBSOCKET_URL = "wss://api-streaming-capital.backend-capital.com/connect"


# DIRECTORIES
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ml-pred/
BASE_STORAGE_DIR = os.path.join(BASE_DIR, 'storage')
BASE_DATA_STORAGE_DIR = os.path.join(BASE_STORAGE_DIR, 'data')

CAPCOM_BASE_STORAGE_DIR = os.path.join(BASE_DATA_STORAGE_DIR, 'capital_com')
CAPCOM_RAW_DATA_DIR = os.path.join(BASE_DATA_STORAGE_DIR, 'raw')
CAPCOM_PROCESSED_DATA_DIR = os.path.join(BASE_DATA_STORAGE_DIR, 'processed')
CAPCOM_RESPONSE_JSON_DIR = os.path.join(CAPCOM_RAW_DATA_DIR, 'saved_responses')