# About tests/
This folder is focusing on testing different components of each system.<br>
Currently only "relatively" simple tests for the data pipeline has been implemented. Even though tests should be updated before/after additions/changes, there are still tests to be made. Eg. Testing performance, different market scenarios, edge cases, etc.

# Structure
```
├── utils/                                      # Utilities used for tests
│   └── test_data_utils.py                      # Utils just for data testing (eg. creating datasets, cleaning up, etc.)
├── data/                                       # Tests for components related to data pipeline
│   ├── unit_tests/                             # Test individual units (modules, functions, classes) in isolation from the rest of the program/system
│   │   ├── test_cleaning.py
│   │   ├── test_validation.py
│   │   ├── test_generation.py                  # Feature generation tests
│   │   ├── test_transformation.py              # Feature and column transformation tests
│   │   └── test_selection.py                   # Feature selection tests
│   └── integration_tests/                      # Test whether many separately developed modules work together as expected
│       ├── test_cleaning_validation.py
│       ├── test_generation_transformation.py  
│       ├── test_transformation_selection.py
│       └── test_pipeline.py                    # Testing the whole pipeline
└── ...
```