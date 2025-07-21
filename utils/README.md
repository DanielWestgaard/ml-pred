# About utils/ 🛠️

## Overview
The utils component provides **shared utilities and helper functions** used throughout the entire project. These utilities support configuration management, data processing, model handling, logging, and component integration, ensuring consistency and reducing code duplication across the system.

## Key Features 🚀
- **Configuration Management**: Robust hypothesis and model configuration loading
- **Data Processing Utilities**: Timestamp parsing, data validation, and feature utilities
- **Model Management**: Model metadata loading and model discovery utilities  
- **Integration Support**: Utilities for seamless component integration
- **Error Handling**: Robust error handling and logging utilities
- **Extensible Design**: Easy to extend with new utilities as needed

## Structure
```
utils/
├── alpha_vantage_utils.py          # Alpha Vantage API utilities and helpers
├── broker_utils.py                 # Broker API utilities (Capital.com integration)
├── data_utils.py                   # Data processing, timestamps, features, validation
├── logging_utils.py                # Logging configuration and utilities
├── shared_utils.py                 # Cross-component shared utilities
├── README.md                       # Documentation
│
├── hypothesis/                     # Hypothesis configuration utilities
│   └── config_loader.py            # HypothesisConfigLoader - robust config management
│
└── model/                          # Model utilities and metadata management
    ├── model_metadata_loader.py    # Model metadata loading and management
    └── model_utils.py              # Model training, saving, and handling utilities
```

## Key Utilities 📦

### Configuration Management

#### HypothesisConfigLoader (`hypothesis/config_loader.py`)
Robust configuration management for hypothesis-based workflows:

**Key Features:**
- **YAML Configuration Loading**: Safe loading of hypothesis config files
- **Configuration Validation**: Validates config structure and required fields
- **Dynamic Source Handling**: Supports multiple configuration sources
- **Error Resilience**: Comprehensive error handling and user feedback
- **Integration Ready**: Used by data pipeline, model training, evaluation, and strategy components

**Usage:**
```python
import utils.hypothesis.config_loader as config_loader

# Load and validate hypothesis configuration
config = config_loader.load_hypothesis_config("hypothesis/1_session_based_tod_anomalies/config.yml")
processed_config = config.handle_hypothesis_source()

# Access configuration sections
data_config = processed_config.get('data_pipeline', {})
model_config = processed_config.get('model_training', {})
strategy_config = processed_config.get('strategy', {})
```

### Model Management

#### Model Metadata Loader (`model/model_metadata_loader.py`)
Comprehensive model metadata management system:

**Key Features:**
- **Metadata Loading**: Load model metadata from YAML files
- **Model Discovery**: Find latest models by type or timestamp
- **Feature Management**: Extract model feature requirements
- **Hypothesis Integration**: Link models to their training hypotheses
- **Validation**: Ensure model metadata completeness

**Usage:**
```python
from utils.model.model_metadata_loader import load_model_metadata, get_latest_model

# Load specific model metadata
loader = load_model_metadata("storage/models/2025-07-03_12:45:35")
hypothesis_type = loader.get_hypothesis_type()
features = loader.get_feature_names()
target = loader.get_target_column()

# Find latest model
latest_model = get_latest_model(model_type="xgboost")
```

### Data Processing Utilities

#### Data Utils (`data_utils.py`)
Comprehensive data processing utilities:

**Key Features:**
- **Timestamp Processing**: Robust timestamp parsing with `parse_timestamp_robust()` and `parse_timestamp_series()`
- **Feature Utilities**: Feature specification checking and validation
- **Dataset Management**: Dataset naming conventions and file handling
- **Data Validation**: Data integrity checks and validation utilities
- **Processing Support**: Utilities for data pipeline operations

**Recent Enhancements:**
```python
from utils.data_utils import parse_timestamp_robust, parse_timestamp_series

# Handle mixed timestamp formats
timestamp = parse_timestamp_robust("2025-01-15 10:30:00")  # Handles various formats
timestamp_series = parse_timestamp_series(df['timestamp'])  # Handles pandas Series
```

### Integration Utilities

#### Shared Utils (`shared_utils.py`)
Cross-component utilities for system integration:
- **Component Communication**: Utilities for inter-component communication
- **Configuration Helpers**: Shared configuration processing utilities
- **Error Handling**: Standardized error handling patterns
- **System Integration**: Utilities supporting component integration

#### Broker Utils (`broker_utils.py`)
Capital.com API integration utilities:
- **API Response Processing**: Convert JSON responses to OHLCV format
- **Data Transformation**: Price calculation (mid-price from bid/ask)
- **File Management**: Automated data file saving and organization
- **Error Handling**: Robust API error handling and recovery

#### Logging Utils (`logging_utils.py`)
Centralized logging configuration:
- **Structured Logging**: Consistent logging format across all components
- **Log Level Management**: Configurable log levels per component
- **File Organization**: Automatic log file organization by component and date
- **Error Tracking**: Enhanced error logging and stack trace management

## Integration Examples 🔌

### Strategy Integration
```python
# Strategy using multiple utilities
import utils.hypothesis.config_loader as config_loader
from utils.model.model_metadata_loader import load_model_metadata
from utils.data_utils import parse_timestamp_series

# Load strategy configuration
config = config_loader.load_hypothesis_config(config_path)
strategy_config = config.handle_hypothesis_source()

# Load model metadata
model_loader = load_model_metadata(model_directory)
features = model_loader.get_feature_names()

# Process timestamps
df['timestamp'] = parse_timestamp_series(df['raw_timestamp'])
```

### Data Pipeline Integration
```python
# Data pipeline using configuration and data utilities
from utils.hypothesis.config_loader import load_hypothesis_config
from utils.data_utils import check_feature_specification

config_loader = load_hypothesis_config(hypothesis_path)
config = config_loader.handle_hypothesis_source()

# Check feature specifications
required_features = config.get('data_pipeline', {}).get('required_features', {})
feature_valid = check_feature_specification(required_features)
```