# About tests/
This folder is focusing on testing different components of each system.<br>
Currently only "relatively" simple tests for the data pipeline has been implemented. Even though tests should be updated before/after additions/changes, there are still tests to be made. Eg. Testing performance, different market scenarios, edge cases, etc.

# Structure
```
├── utils/                                      # Utilities used for tests
│   ├── test_data_utils.py                      # Utils just for data testing (eg. creating datasets, cleaning up, etc.)
│   └── test_hypothesis_utils.py                # TODO: Utils for hypothesis testing (mocking config and methods)
│
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
│
├── hypothesis/                                 # TODO: Testing hypothesis flow and logic in existing functions
│   ├── unit_tests/
│   │   ├── test_config.py                      # Making sure configuration file works and handles mistakes
│   │   ├── test_hypothesis_data.py
│   │   ├── test_hypothesis_model.py
│   │   ├── test_hypothesis_evaluation.py
│   │   └── ...                                 # Potentially new supported components
│   └── integration_tests/
│       ├── test_hypothesis_flows_.py
│
└── ...
```
