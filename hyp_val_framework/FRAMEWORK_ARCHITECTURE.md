# Hypothesis Validation Framework Architecture 🏗️

*A comprehensive, reusable framework for rigorously testing trading hypotheses*

## Overview

This framework provides multiple levels of hypothesis validation, from quick tests to comprehensive statistical analysis. It's designed to prevent common pitfalls like lookahead bias, data snooping, and overfitting.

## Architecture Layers

```
                    User Interface Layer
                           │
    ┌──────────────────────┼──────────────────────┐
    │                      │                      │
Simple Usage         Standard Usage        Advanced Usage
    │                      │                      │
quick_validator    HypothesisValidator   BaseHypothesisValidator
    │                      │                      │
    └──────────────────────┼──────────────────────┘
                           │
                    Core Framework
                    - Parameter estimation
                    - Signal generation  
                    - Statistical testing
                    - Validation methods
```

## Core Components

### 1. `BaseHypothesisValidator` (Abstract Base Class)

The foundation that all specific hypothesis validators inherit from:

```python
from hyp_val_framework.base_hypothesis_validator import BaseHypothesisValidator

class MyHypothesisValidator(BaseHypothesisValidator):
    def get_hypothesis_name(self) -> str:
        return "My Custom Hypothesis"
    
    def prepare_hypothesis_data(self):
        # Calculate indicators, features, etc.
        pass
    
    def estimate_parameters(self, estimation_data):
        # Estimate thresholds from historical data
        return {'threshold': estimation_data['price'].quantile(0.9)}
    
    def generate_signals(self, data, parameters):
        # Generate trading signals
        return data['price'] > parameters['threshold']
```

**Key Features:**
- Automatic chronological data splits
- Standard statistical testing framework
- Walk-forward and bootstrap validation
- Results summary and saving
- Extensible validation pipeline

### 2. `HypothesisValidator` (Main Interface)

Your enhanced main class that provides multiple validation approaches:

```python
from hyp_val_framework.validate_hypothesis import HypothesisValidator

# Initialize the main Validator
validator = HypothesisValidator("data.csv")

# Quick hypothesis testing
results = validator.quick_test_volume_hypothesis(volume_threshold_pct=95)

# Advanced validation
results = validator.run_comprehensive_validation(MyAdvancedValidator)
```

### 3. `quick_validator.py` (Rapid Testing)

For testing hypotheses without creating full classes:

```python
from hyp_val_framework.quick_validator import quick_validate_hypothesis

def my_signals(data, params):
    return data['volume'] > params['vol_threshold']

def my_params(estimation_data):
    return {'vol_threshold': estimation_data['volume'].quantile(0.9)}

results = quick_validate_hypothesis(
    "data.csv",
    "Volume Spike Test",
    "High volume predicts mean reversion", 
    my_signals,
    my_params
)
```

## Usage Patterns

### Level 1: Quick Testing (Fastest)
```python
# Test pre-built hypothesis templates
from hyp_val_framework.quick_validator import test_volume_hypothesis, test_momentum_hypothesis

results1 = test_volume_hypothesis("data.csv", volume_threshold_pct=90)
results2 = test_momentum_hypothesis("data.csv", lookback_periods=20)
```

### Level 2: Custom Quick Testing
```python
# Test your own hypothesis quickly
validator = HypothesisValidator("data.csv")

results = validator.quick_test_custom_hypothesis(
    "Price Gap Continuation",
    "Large price gaps predict continuation",
    signal_function,
    parameter_function
)
```

### Level 3: Comprehensive Validation
```python
# Full statistical analysis with multiple validation methods
from hyp_val_framework.examples.spread_expansion_validator import SpreadExpansionValidator

validator = SpreadExpansionValidator("data.csv", config)
results = validator.run()
validator.print_validation_summary()
validator.save_results()
```

## Directory Structure

```
hyp_val_framework/
├── README.md                           # Getting started guide
├── FRAMEWORK_ARCHITECTURE.md           # This file
├── HYPOTHESIS_VALIDATION_GUIDE.md      # Methodology guide
│
├── validate_hypothesis.py              # Main interface
├── base_hypothesis_validator.py        # Abstract base class
├── quick_validator.py                  # Rapid testing tools
│
├── examples/
│   ├── spread_expansion_validator.py   # Full implementation example
│   └── volume_spike_validator.py       # Another example
│
└── templates/
    ├── hypothesis_template.py          # Template for new hypotheses
    └── config_template.yml             # Configuration template
```

## Key Design Principles

### 1. **Rigorous by Default**
- Chronological splits prevent lookahead bias
- Parameter estimation on historical data only
- Multiple validation methods (statistical, walk-forward, bootstrap)

### 2. **Easy to Use**
```python
# One-liner hypothesis testing
results = test_volume_hypothesis("data.csv")
```

### 3. **Extensible**
```python
# Add your own validation methods
class MyValidator(BaseHypothesisValidator):
    def run_additional_validation(self, parameters):
        # Your custom validation logic
        return custom_results
```

### 4. **Configuration-Driven**
```python
config = {
    'volume_threshold_pct': 95,
    'test_timeframes': [30, 60],
    'bootstrap_samples': 1000
}
validator = MyValidator("data.csv", config)
```

## Validation Pipeline

Every hypothesis goes through this standardized pipeline:

```
1. Data Preparation
   ├── Load & clean data
   ├── Calculate indicators
   └── Chronological split

2. Parameter Estimation
   ├── Use estimation data only
   ├── Calculate thresholds
   └── Store parameters

3. Signal Generation  
   ├── Apply parameters to validation data
   ├── Generate trading signals
   └── Check signal frequency

4. Hypothesis Testing
   ├── Statistical tests (t-test, effect size)
   ├── Multiple timeframes
   └── Direction analysis

5. Additional Validation
   ├── Walk-forward analysis
   ├── Bootstrap confidence intervals
   └── Custom validation methods

6. Results & Recommendation
   ├── Calculate validation score
   ├── Generate recommendation
   └── Save results
```

## Framework Benefits

### ✅ **Prevents Common Pitfalls**
- **Lookahead bias**: Chronological splits only
- **Data snooping**: Multiple testing corrections
- **Overfitting**: Out-of-sample validation
- **Cherry picking**: Standardized reporting

### ✅ **Speeds Development**
- **Templates**: Pre-built hypothesis patterns
- **Quick testing**: One-line hypothesis validation
- **Reusable**: Standard validation pipeline

### ✅ **Ensures Quality**
- **Statistical rigor**: Effect sizes, confidence intervals
- **Multiple validation**: Walk-forward, bootstrap
- **Reproducible**: Configuration-driven, results saved

## How to Continue Development

### 1. **Short Term (Next Steps)**
1. **Test the framework** with a simple hypothesis
2. **Create more templates** in `quick_validator.py`
3. **Add configuration files** for common hypothesis types
4. **Build more examples** in the `examples/` directory

### 2. **Medium Term (Next Few Weeks)**
1. **Integration testing** with your existing data pipeline
2. **Performance optimization** for large datasets
3. **Plotting integration** for visualization
4. **Documentation** and examples

### 3. **Long Term (Future Features)**
1. **Multiple asset support** (test across instruments)
2. **Portfolio-level validation** (hypothesis combinations)
3. **Live validation** (test hypotheses on live data)
4. **Web interface** for non-technical users

## Example Workflow

Here's how you'd use the framework for a new hypothesis:

```python
# Step 1: Quick test to see if it's worth pursuing
from hyp_val_framework.validate_hypothesis import HypothesisValidator

validator = HypothesisValidator("my_data.csv")
quick_results = validator.quick_test_volume_hypothesis(volume_threshold_pct=90)

if quick_results['validation_score'] > 0.5:
    print("Promising! Let's do comprehensive validation...")
    
    # Step 2: Create comprehensive validator
    class VolumeHypothesisValidator(BaseHypothesisValidator):
        # ... implement required methods
        
    # Step 3: Run full validation
    comprehensive_results = validator.run_comprehensive_validation(VolumeHypothesisValidator)
    
    if comprehensive_results['validation_score'] > 0.7:
        print("✅ Hypothesis validated! Ready for implementation")
    else:
        print("❌ Hypothesis failed comprehensive validation")
```

## Integration with Existing System

The framework integrates seamlessly with your existing components:

```python
# Uses your existing data pipeline
from data.processing.cleaning import DataCleaner  
from data.processing.validation import DataValidator
from utils.data_utils import check_and_return_df

# Plugs into your hypothesis system
from hypothesis.config import load_hypothesis_config
from backtesting.engine import BacktestEngine
```

This allows you to:
1. **Validate hypotheses** before adding them to your system
2. **Filter out weak patterns** early in development  
3. **Build confidence** in implemented strategies
4. **Maintain quality standards** across all hypotheses

---

*This framework transforms hypothesis testing from ad-hoc scripts to a rigorous, reusable system that prevents false discoveries and accelerates development.*