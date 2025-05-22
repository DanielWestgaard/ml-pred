# About tests/
This folder is focusing on testing different components of each system.

# Structure
```
├── utils/                                  # Utility files (eg. creating datasets, cleaning up, etc.)
├── data/                                   # Tests for components related to data pipeline
│   ├── unit_tests/                         # Test individual units (modules, functions, classes) in isolation from the rest of the program/system
│   │   ├── test_cleaning.py
│   │   ├── test_validation.py
│   │   ├── test_generation.py              # Feature generation tests
│   │   ├── test_transformation.py          # Feature and column transformation tests
│   │   └── test_selection.py               # Feature selection tests
│   └── integration_tests/                  # Test whether many separately developed modules work together as expected
│       ├── test_cleaning_validation.py
└── ...
```