# Hypothesis System Tests

This directory contains comprehensive tests for the hypothesis configuration system, covering both unit and integration testing scenarios.

## Test Structure

```
tests/hypothesis/
├── unit_tests/
│   └── test_config.py          # Unit tests for configuration parsing
├── integration_tests/
│   └── test_hypothesis_flow.py # Integration tests for complete flows
├── test_runner.py              # Test runner script
└── README.md                   # This file
```

## Test Coverage

### Unit Tests (`unit_tests/test_config.py`)

Tests the `handle_hypothesis_source()` function in `utils/shared_utils.py`:

- ✅ Valid complete configuration parsing
- ✅ Minimal configuration handling
- ✅ Missing hypothesis name error handling
- ✅ File not found error handling
- ✅ Invalid YAML error handling
- ✅ Partial configuration scenarios
- ✅ Different evaluation types (new/stored)
- ✅ Feature configuration parsing
- ✅ Model parameter extraction
- ✅ Logging behavior validation

### Integration Tests (`integration_tests/test_hypothesis_flow.py`)

Tests the complete hypothesis workflow integration:

- ✅ Configuration parsing integration across components
- ✅ Data pipeline hypothesis integration
- ✅ Model training hypothesis integration
- ✅ Configuration precedence (hypothesis vs direct params)
- ✅ Data path resolution (raw vs processed)
- ✅ Evaluation configuration integration
- ✅ Feature configuration handling
- ✅ Error handling in complete flows
- ✅ Corrupted/empty config file handling
- ✅ Partial configuration robustness

### Utility Tests (`../utils/test_hypothesis_utils.py`)

Tests hypothesis-related utility functions:

- ✅ `handle_hypothesis_source()` comprehensive testing
- ✅ `safely_execute()` function testing
- ✅ `list_of_functions_in_class()` function testing
- ✅ Edge case handling
- ✅ Error scenarios

## Running Tests

### Option 1: Using the test runner
```bash
cd /path/to/ml-pred
python tests/hypothesis/test_runner.py
```

### Option 2: Using unittest
```bash
cd /path/to/ml-pred
python -m unittest discover tests/hypothesis/unit_tests -v
python -m unittest discover tests/hypothesis/integration_tests -v
```

### Option 3: Using pytest (if available)
```bash
cd /path/to/ml-pred
python -m pytest tests/hypothesis/ -v
```

### Option 4: Individual test files
```bash
cd /path/to/ml-pred
python -m unittest tests.hypothesis.unit_tests.test_config -v
python -m unittest tests.hypothesis.integration_tests.test_hypothesis_flow -v
python -m unittest tests.utils.test_hypothesis_utils -v
```

## Test Scenarios Covered

### Configuration File Scenarios
1. **Complete Configuration**: All sections present with valid data
2. **Minimal Configuration**: Only hypothesis name provided
3. **Missing Hypothesis**: Configuration without required hypothesis name
4. **Partial Configurations**: Missing optional sections
5. **Invalid YAML**: Malformed configuration files
6. **Empty Files**: Empty or corrupted configuration files

### Data Flow Scenarios
1. **Data Pipeline Only**: Only data processing configuration
2. **Model Training Only**: Only model training configuration
3. **Complete Flow**: Data pipeline → Model training → Evaluation
4. **Mixed Flows**: Various combinations of components

### Error Handling Scenarios
1. **File System Errors**: Missing files, permission issues
2. **Configuration Errors**: Invalid values, missing required fields
3. **Integration Errors**: Component initialization failures
4. **Runtime Errors**: Execution failures during processing

### Edge Cases
1. **Empty Feature Lists**: Features with empty arrays
2. **None Values**: Configuration with None values
3. **Complex Parameters**: Nested parameter structures
4. **Path Resolution**: Raw vs processed data path handling

## Key Test Benefits

1. **Reliability**: Ensures hypothesis configuration system works consistently
2. **Regression Prevention**: Catches breaking changes early
3. **Documentation**: Tests serve as usage examples
4. **Confidence**: Provides confidence when refactoring
5. **Modularity**: Tests individual components and their integration

## Adding New Tests

When adding new hypothesis features:

1. **Add Unit Tests**: Test the individual function/method
2. **Add Integration Tests**: Test the feature in complete workflows
3. **Update Test Runner**: Ensure new tests are discovered
4. **Document**: Add test scenarios to this README

### Test Template
```python
def test_new_feature(self):
    \"\"\"Test description\"\"\"
    # Arrange
    config = {...}
    config_path = self._create_temp_config(config)
    
    # Act
    result = handle_hypothesis_source(config_path)
    
    # Assert
    self.assertEqual(result['expected_key'], 'expected_value')
```

## Continuous Integration

These tests should be run:
- Before any commits to hypothesis-related code
- In CI/CD pipelines
- Before releases
- When adding new hypothesis types

## Dependencies

Tests require:
- Python 3.7+
- unittest (built-in)
- tempfile (built-in)
- yaml
- pandas (for integration tests)
- unittest.mock (built-in)

Optional:
- pytest (for alternative test runner)