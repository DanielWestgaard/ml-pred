# Hypothesis Validation Guide 📚

*A comprehensive guide for rigorously testing trading hypotheses and avoiding common pitfalls*

## Table of Contents
- [Overview](#overview)
- [Common Pitfalls](#common-pitfalls)
- [Validation Framework](#validation-framework)
- [Step-by-Step Process](#step-by-step-process)
- [Code Architecture](#code-architecture)
- [Spread Expansion Case Study](#spread-expansion-case-study)
- [Best Practices](#best-practices)
- [Tools and Scripts](#tools-and-scripts)

## Overview

Trading hypothesis validation is the process of rigorously testing whether a market pattern or signal has genuine predictive power. The goal is to distinguish between real market inefficiencies and random noise that appears significant due to data mining bias.

### Why Rigorous Validation Matters

- **Prevents false discoveries**: Random patterns can appear significant in-sample
- **Saves time and capital**: Avoids implementing flawed strategies
- **Builds confidence**: Validated patterns are more likely to persist out-of-sample
- **Scientific approach**: Treats trading like a research discipline

## Common Pitfalls 🚨

### 1. Lookahead Bias
**Problem**: Using future information to make past decisions
```python
# BAD: Rolling percentile uses future data points
df['spread_pct'] = df['spread'].rolling(window=240).quantile(0.9)

# GOOD: Calculate percentiles only on historical data
def calculate_historical_percentile(data, window, percentile):
    result = []
    for i in range(len(data)):
        start_idx = max(0, i - window + 1)
        hist_data = data[start_idx:i+1]
        if len(hist_data) >= 10:  # Minimum samples
            result.append(np.percentile(hist_data, percentile * 100))
        else:
            result.append(np.nan)
    return result
```

### 2. Data Snooping Bias
**Problem**: Testing multiple hypotheses on the same dataset
- Testing 20 patterns → 1 will appear significant by chance (p < 0.05)
- Multiple comparisons inflate Type I error rate
- **Solution**: Use Bonferroni correction or hold-out validation

### 3. In-Sample Overfitting
**Problem**: Optimizing parameters on the same data you test on
```python
# BAD: Same data for parameter tuning and testing
best_threshold = optimize_threshold(data)  # Uses full dataset
results = test_strategy(data, best_threshold)  # Same dataset!

# GOOD: Separate data for optimization and testing
train_data = data[:split_point]
test_data = data[split_point:]
best_threshold = optimize_threshold(train_data)
results = test_strategy(test_data, best_threshold)
```

### 4. Insufficient Sample Size
**Problem**: Drawing conclusions from too few observations
- Need minimum 30-50 events for statistical power
- Consider practical significance vs statistical significance
- Use effect size measures (Cohen's d) alongside p-values

## Validation Framework

Our validation framework implements multiple testing methods:

```
                 Raw Data
                    ↓
         ┌─────────────────────┐
         │   Data Preparation  │
         │ - Clean & sort data │
         │ - Calculate features│
         │ - Forward returns   │
         └─────────────────────┘
                    ↓
         ┌─────────────────────┐
         │ Chronological Split │
         │ - Training: 70%     │
         │ - Testing: 30%      │
         │ - NO random mixing  │
         └─────────────────────┘
                    ↓
         ┌─────────────────────┐
         │  Parameter Fitting  │
         │ - Use ONLY train    │
         │ - Percentiles       │
         │ - Thresholds        │
         │ - Volume averages   │
         └─────────────────────┘
                    ↓
         ┌─────────────────────┐
         │   Out-of-Sample     │
         │      Testing        │
         │ - Apply to test     │
         │ - No peeking!       │
         │ - Statistical tests │
         └─────────────────────┘
                    ↓
         ┌─────────────────────┐
         │    Validation       │
         │ - Multiple periods  │
         │ - Walk-forward      │
         │ - Bootstrap CI      │
         └─────────────────────┘
```

## Step-by-Step Process

### Step 1: Hypothesis Formulation
Define your hypothesis clearly and specifically:

```
Example: "When spread expands to 85th percentile with above-average volume,
price will mean-revert (negative returns) over the next 45-60 minutes"

Components:
- Signal: Spread expansion (85th percentile)
- Condition: Above-average volume
- Prediction: Negative returns
- Timeframe: 45-60 minutes
```

### Step 2: Data Preparation
```python
class HypothesisValidator:
    def __init__(self, data_path):
        # Load and clean data
        self.df = pd.read_csv(data_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values('Date').reset_index(drop=True)
        
        # Calculate forward returns
        for minutes in [15, 30, 45, 60]:
            self.df[f'return_{minutes}m'] = (
                self.df['price'].shift(-minutes) - self.df['price']
            ) / self.df['price']
```

### Step 3: Chronological Split
```python
def chronological_split(self, test_size=0.3):
    split_idx = int(len(self.df) * (1 - test_size))
    
    train_df = self.df.iloc[:split_idx].copy()
    test_df = self.df.iloc[split_idx:].copy()
    
    # Ensure no data leakage
    print(f"Training: {train_df['Date'].min()} to {train_df['Date'].max()}")
    print(f"Testing:  {test_df['Date'].min()} to {test_df['Date'].max()}")
    
    return train_df, test_df
```

### Step 4: Parameter Estimation (Training Data Only)
```python
def calculate_parameters(self, train_df):
    # Calculate thresholds ONLY on training data
    parameters = {
        'spread_85pct': train_df['spread'].quantile(0.85),
        'volume_avg': train_df['volume'].mean(),
        'min_events': 20  # Minimum for statistical power
    }
    
    return parameters
```

### Step 5: Signal Generation (Test Data)
```python
def generate_signals(self, test_df, parameters):
    # Apply parameters to test data
    signals = (
        (test_df['spread'] >= parameters['spread_85pct']) &
        (test_df['volume'] > parameters['volume_avg'])
    )
    
    return signals
```

### Step 6: Statistical Testing
```python
def test_hypothesis(self, test_df, signals, return_col):
    signal_returns = test_df[signals][return_col].dropna()
    all_returns = test_df[return_col].dropna()
    
    if len(signal_returns) < parameters['min_events']:
        return None
    
    # Statistical tests
    t_stat, p_value = stats.ttest_ind(signal_returns, all_returns)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(
        ((len(signal_returns) - 1) * signal_returns.std()**2 +
         (len(all_returns) - 1) * all_returns.std()**2) /
        (len(signal_returns) + len(all_returns) - 2)
    )
    cohens_d = (signal_returns.mean() - all_returns.mean()) / pooled_std
    
    return {
        'events': len(signal_returns),
        'mean_signal': signal_returns.mean(),
        'mean_all': all_returns.mean(),
        'p_value': p_value,
        'effect_size': abs(cohens_d),
        'significant': p_value < 0.05
    }
```

### Step 7: Multiple Validation Methods

#### A. Chronological Validation
Test across multiple time periods:
```python
def chronological_validation(self, n_periods=3):
    results = []
    
    for i in range(n_periods):
        # Split data into different periods
        test_start = 0.7 + (i * 0.1)  # 70%, 80%, 90%
        test_end = min(1.0, test_start + 0.2)
        
        # Test hypothesis on this period
        result = self.test_single_period(test_start, test_end)
        results.append(result)
    
    return results
```

#### B. Walk-Forward Analysis
```python
def walk_forward_validation(self, train_months=6, step_months=1):
    results = []
    
    # Rolling window approach
    for start_date in date_range:
        train_end = start_date + timedelta(days=train_months*30)
        test_end = train_end + timedelta(days=step_months*30)
        
        # Train on this window
        parameters = self.fit_parameters(start_date, train_end)
        
        # Test on next window
        result = self.test_parameters(train_end, test_end, parameters)
        results.append(result)
    
    return results
```

#### C. Bootstrap Confidence Intervals
```python
def bootstrap_confidence(self, n_bootstrap=1000):
    bootstrap_effects = []
    
    for i in range(n_bootstrap):
        # Resample with replacement
        boot_signal = np.random.choice(signal_returns, size=len(signal_returns))
        boot_all = np.random.choice(all_returns, size=len(all_returns))
        
        effect = boot_signal.mean() - boot_all.mean()
        bootstrap_effects.append(effect)
    
    # Calculate confidence intervals
    ci_lower = np.percentile(bootstrap_effects, 2.5)
    ci_upper = np.percentile(bootstrap_effects, 97.5)
    
    return ci_lower, ci_upper
```

## Code Architecture

### Main Classes

1. **HypothesisValidator**: Core validation logic
2. **RigorousValidator**: Multiple validation methods
3. **CandlestickAnalyzer**: Visualization and exploration

### Key Scripts

1. **`spread_compression_analyzer.py`**: Initial exploration
2. **`rigorous_validation.py`**: Comprehensive testing
3. **`analyze_spread_data_temp.py`**: Updated plotting script

### Architecture Pattern
```
Exploration → Hypothesis → Validation → Decision
     ↓            ↓           ↓          ↓
  Plot data   Formulate    Test rigor   Implement
  Find patterns  clearly    No shortcuts  or abandon
```

## Spread Expansion Case Study

### Initial Results (Misleading)
- Found 8 "significant" patterns (p < 0.05)
- Effect sizes appeared meaningful
- **Problem**: Used rolling percentiles (lookahead bias)

### Rigorous Testing (Reality)
```python
# Split data chronologically
train_data = data[:70%]
test_data = data[70%:]

# Calculate thresholds on training data ONLY
spread_85pct = train_data['spread'].quantile(0.85)

# Apply to test data
expansion_events = test_data[test_data['spread'] >= spread_85pct]

# Result: No significant patterns found
```

### Why It Failed
1. **Overfitting**: Patterns optimized on full dataset
2. **Multiple testing**: Tried many combinations
3. **Survivorship bias**: Selected "best" results
4. **Insufficient robustness**: Didn't hold across periods

## Best Practices ✅

### Data Handling
- Always sort chronologically
- Use only historical data for parameters
- Implement proper forward returns (shift -N)
- Handle missing data appropriately

### Statistical Testing
```python
# Multiple comparisons correction
def bonferroni_correction(p_values, alpha=0.05):
    corrected_alpha = alpha / len(p_values)
    return [p < corrected_alpha for p in p_values]

# Effect size interpretation
def interpret_effect_size(cohens_d):
    if abs(cohens_d) < 0.2:
        return "negligible"
    elif abs(cohens_d) < 0.5:
        return "small"
    elif abs(cohens_d) < 0.8:
        return "medium"
    else:
        return "large"
```

### Validation Checklist
- [ ] Hypothesis clearly defined before testing
- [ ] Chronological data splits only
- [ ] Parameters fit on training data
- [ ] Minimum sample size requirements met
- [ ] Multiple validation methods used
- [ ] Effect sizes reported alongside p-values
- [ ] Results replicated across time periods

### Red Flags 🚩
- P-hacking: Testing until you find significance
- Cherry-picking: Reporting only favorable results
- Parameter tuning on test data
- Using future information in features
- Ignoring practical significance

## Tools and Scripts

### 1. Quick Exploration
```python
# Use for initial pattern discovery
analyzer = CandlestickAnalyzer('data.csv')
analyzer.print_summary()
analyzer.plot_expansion_analysis()
```

### 2. Hypothesis Testing
```python
# Use for rigorous validation
validator = RigorousValidator('data.csv')
results, score = validator.comprehensive_validation()
```

### 3. Custom Validation
```python
# Template for your own hypotheses
class CustomValidator(RigorousValidator):
    def define_signal(self, data, parameters):
        # Your hypothesis logic here
        return signals
    
    def test_hypothesis(self, data, signals):
        # Your testing logic here
        return results
```

## Example Workflow

### 1. Exploration Phase
```bash
python analyze_spread_data_temp.py
# Choose option 2: New expansion analysis
```

### 2. Initial Testing
```python
from spread_compression_analyzer import SpreadCompressionAnalyzer
analyzer = SpreadCompressionAnalyzer('data.csv')
results = analyzer.test_predictive_power()
```

### 3. Rigorous Validation
```bash
python rigorous_validation.py
# Comprehensive testing framework
```

### 4. Decision Making
- Score > 0.7: ✅ Proceed with confidence
- Score 0.5-0.7: ⚠️ Proceed with caution
- Score < 0.5: ❌ Abandon hypothesis

## Conclusion

Rigorous hypothesis validation is essential for successful quantitative trading. The framework we built demonstrates that many apparent patterns are false positives when tested properly. 

**Key takeaway**: Better to discover a hypothesis is flawed during testing than after deploying capital.

### Next Steps
1. Use this framework for future hypotheses
2. Focus on economic intuition, not just statistical patterns
3. Consider alternative data sources and timeframes
4. Build a library of validated patterns over time

Remember: **Rigorous testing saves time, money, and reputation.**

---

*Generated during spread expansion hypothesis validation - August 2025*