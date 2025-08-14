# Combination Testing Framework

**The systematic discovery engine for market inefficiencies hidden in data source combinations.**

## 🎯 Purpose

This framework addresses the core question: *"What combinations of market data + alternative data sources contain signals that institutions miss due to organizational constraints?"*

### Strategic Philosophy
- **Institutions have organizational silos** - Market desk ≠ Corporate calendar team
- **Compliance slows integration** - We can test combinations faster
- **Scale constraints** - Opportunities too small for institutional capital
- **Cross-departmental gaps** - Data combinations that require multiple teams to coordinate

## 🏗️ Architecture

### Core Components

```
combination_tester/
├── combination_orchestrator.py     # Main coordination and execution
├── test_strategies/                # Different testing approaches
│   ├── statistical_strategy.py     # Phase 1: Correlation + significance
│   ├── ml_strategy.py              # Phase 2: Feature importance + ML validation
│   └── custom_strategy.py          # Phase 3: User-defined testing logic
└── README.md                       # This documentation
```

### Design Principles
1. **Simple but Powerful** - Easy to understand, maintain, and extend
2. **Configuration-Driven** - No hardcoded parameters or assumptions
3. **Modular Strategies** - Easy to add new testing approaches
4. **Comprehensive Results** - Statistical validation + actionable insights
5. **Integration-Ready** - Seamless pipeline integration

## 🚀 Quick Start

### Basic Usage
```bash
# Test CLOV with 4-hour data + calendar events
python main.py --hypothesis "clov_4h_combination_testing.yml"
```

### Configuration File
```yml
hypothesis:
  name: "clov_institutional_blind_spots"
  description: "Find CLOV inefficiencies in calendar event timing"

multi_data:
  purpose: "historical"
  symbol: "CLOV"
  from_date: "2022-01-01"
  to_date: "2025-01-01"
  sources: [calendar_events]

combination_testing:
  enabled: true
  timeframe: "4h"                    # Dynamic: 1m, 4h, 1d, etc.
  strategy: "statistical"            # statistical, ml_based, custom
  
  test_matrix:
    base_features: ["Close", "Volume", "spread_avg"]
    # Calendar features auto-detected from combined data
  
  validation:
    min_correlation_threshold: 0.03  # Lower for 4h data
    significance_level: 0.05         # 5% significance level
    min_data_points: 50             # Minimum sample size
    
  results:
    save_detailed: true
    generate_visualizations: false
    rank_combinations: true
```

## 📊 Test Strategies

### 1. Statistical Strategy (Phase 1) ✅
**Focus**: Correlation analysis + statistical significance

**Validation Criteria**:
- Correlation strength vs. threshold
- Statistical significance (p-values)
- Pattern consistency over time
- Data quality validation (reuses existing components)

**Scoring Framework**:
```python
composite_score = (
    correlation_strength * 0.3 +      # Statistical strength
    significance_score * 0.3 +        # Statistical significance  
    pattern_consistency * 0.2 +       # Stability over time
    blind_spot_potential * 0.2        # Institutional challenge
)
```

**Institutional Blind Spot Scoring**:
- **Complexity Score** - More features = higher institutional coordination challenge
- **Cross-Type Score** - Bonus for combining different data types (market + calendar)
- **Organizational Silo Penalty** - Cross-departmental combinations scored higher

### 2. ML Strategy (Phase 2) 🔄
**Future**: Feature importance testing, predictive power validation

### 3. Custom Strategy (Phase 3) 🔄  
**Future**: User-defined testing logic for specific hypotheses

## 📈 Results & Output

### Execution Summary
```
🏆 Top combination results:
  #1: market_plus_earnings_proximity (score: 0.8745, correlation: 0.6234)
  #2: market_plus_multiple_calendar (score: 0.8234, correlation: 0.5891)
  #3: market_plus_options_expiry (score: 0.7892, correlation: 0.5245)

🎯 Identified 5 potential institutional blind spot combinations
```

### Storage Structure
```
storage/combination_testing/SYMBOL/YYYYMMDD_HHMMSS_STRATEGY/
├── config_used.yml                 # Configuration that generated results
├── results/
│   ├── combination_test_results.json # Comprehensive results
│   └── ranked_combinations.csv       # Top combinations spreadsheet
├── raw_test_data/                   # Individual combination datasets
│   ├── market_only.csv
│   ├── market_plus_earnings_proximity.csv
│   └── ...
└── visualizations/                  # Future: Charts and analysis plots
```

### Results JSON Structure
```json
{
  "summary": {
    "total_combinations": 11,
    "valid_combinations": 8,
    "execution_time_seconds": 16.34,
    "top_score": 0.8745,
    "symbol": "CLOV",
    "strategy_used": "statistical"
  },
  "analysis": {
    "ranked_combinations": [...],
    "top_combinations": [...],
    "institutional_blind_spots": 5
  },
  "test_results": [...]  # Detailed per-combination results
}
```

## 🔧 Configuration Options

### Timeframe Support
```yml
combination_testing:
  timeframe: "1m"    # Ultra high-frequency patterns
  timeframe: "4h"    # Intraday calendar effects
  timeframe: "1d"    # Daily patterns + events
```

### Strategy Selection
```yml
combination_testing:
  strategy: "statistical"    # Phase 1: Correlation + significance
  strategy: "ml_based"       # Phase 2: Feature importance + ML
  strategy: "custom"         # Phase 3: User-defined logic
```

### Validation Thresholds
```yml
validation:
  min_correlation_threshold: 0.05    # Minimum meaningful correlation
  significance_level: 0.05           # Statistical significance threshold
  min_data_points: 100              # Minimum sample size required
```

### Test Matrix Customization
```yml
test_matrix:
  base_features: ["Close", "Volume", "spread_avg"]
  # Calendar features auto-detected from data:
  # - days_to_next_earnings
  # - earnings_proximity_zone  
  # - days_to_options_expiration
  # - is_holiday_period
  # - is_vacation_period
  # - etc.
```

## 🎯 Institutional Blind Spot Detection

### What We're Looking For
1. **Cross-Departmental Combinations**
   - Market desk data + Corporate calendar team data
   - Trading systems data + Compliance/regulatory data
   - Risk management data + Alternative data sources

2. **Scale-Inappropriate Signals**
   - Patterns too small for institutional position sizes
   - High-frequency effects requiring immediate execution
   - Small-cap specific calendar timing effects

3. **Compliance-Slow Integration**
   - Alternative data requiring approval processes
   - Cross-vendor data combinations
   - Real-time decision requirements

### Scoring Methodology
```python
def calculate_blind_spot_score(features):
    # Higher scores = more likely institutional blind spot
    complexity_score = min(len(features) / 5.0, 1.0)
    
    data_types = identify_data_types(features)  # market, calendar, sentiment
    cross_type_score = min(len(data_types) / 3.0, 1.0)
    
    silo_penalty = 0.2 if len(data_types) > 1 else 0
    
    return complexity_score * 0.4 + cross_type_score * 0.4 + silo_penalty * 0.2
```

## 🔄 Integration Points

### 1. Data Pipeline Integration
The combination tester integrates with existing components:
```python
# Uses existing data quality validation
from data.processing.validation import DataValidator
from data.processing.cleaning import DataCleaner

# Leverages multi-data infrastructure  
from multi_data_sources.combiners import TimeSeriesDataCombiner
```

### 2. Main Pipeline Integration
```python
# In main.py - automatic detection and execution
if 'combination_testing' in conf_keys:
    tester = CombinationTestOrchestrator(args.hypothesis)
    results = tester.run()
    
    # Feed results to next pipeline stages
    if 'model_training' in conf_keys:
        use_top_combinations_for_training(results.top_combinations)
```

### 3. Configuration Integration
```python
# Uses existing config_loader.py patterns
config_sections = ['multi_data', 'combination_testing', 'model_training']
```

## 🚀 Extension Points

### Adding New Test Strategies
```python
class CustomTestStrategy(BaseTestStrategy):
    """Your custom testing logic"""
    
    def test_combination(self, data, combination):
        # Your validation logic here
        return {
            'valid': True,
            'custom_metric': 0.85,
            'composite_score': self.calculate_score(...)
        }
```

### Adding New Data Sources
1. Implement data source in `sources/`
2. Add to combiner logic  
3. Update feature detection in orchestrator
4. Test automatically included in combinations

### Adding Custom Validation
```python
validation:
  min_correlation_threshold: 0.03
  significance_level: 0.05
  min_data_points: 50
  custom_threshold: 0.75        # Your custom criteria
```

## 📚 Examples

### Example 1: Basic Market Testing
```yml
# Test GBPUSD with technical features only
multi_data:
  symbol: "GBPUSD"
  sources: []  # No additional sources

combination_testing:
  timeframe: "1m"
  validation:
    min_correlation_threshold: 0.05
```

### Example 2: Advanced Calendar Integration
```yml
# Test CLOV with full calendar events
multi_data:
  symbol: "CLOV" 
  sources: [calendar_events]

combination_testing:
  timeframe: "4h"
  validation:
    min_correlation_threshold: 0.03  # Lower for 4h
    min_data_points: 50             # Fewer points for 4h
```

### Example 3: Multi-Symbol Testing
```python
# Future: Configuration for testing multiple symbols
symbols: ["CLOV", "AAPL", "MSFT"]
combination_testing:
  cross_symbol_analysis: true
  generate_comparative_results: true
```

## 🎯 Success Metrics

### Statistical Validation
- **Valid Combinations Found** - Combinations passing all validation criteria
- **Top Score Achieved** - Highest composite score in testing session
- **Significant Relationships** - Number of statistically significant patterns

### Institutional Blind Spot Discovery
- **Cross-Data-Type Combinations** - Combinations spanning multiple data types
- **Organizational Complexity Score** - Average complexity of top combinations
- **Actionable Signals** - Combinations ready for ML training/backtesting

### System Performance
- **Execution Time** - Speed of combination testing process
- **Data Quality** - Percentage of combinations passing validation
- **Result Reproducibility** - Consistency across different time periods

---

## 🏆 Strategic Value Proposition

**"Find the games institutions aren't playing"**

This framework systematically identifies market inefficiencies by:

1. **Testing combinations faster** than institutional compliance allows
2. **Crossing organizational silos** that institutions struggle to bridge  
3. **Focusing on small-scale opportunities** inappropriate for institutional capital
4. **Leveraging agile data integration** impossible at institutional scale

**Result**: A systematic, scalable approach to discovering and validating market inefficiencies hidden in data source combinations.

---

*Built for maintainability, extensibility, and systematic competitive advantage.*