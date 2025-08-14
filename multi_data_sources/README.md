# Introduction
This folder aims to _create dynamic and modular components_ that are each **responsible for a specific data type**. Wether it being "sentiment analysing SOME (Instagram, Twitter/X, etc.) or news (websites, forums, etc.)", weather reports, new laws/regulations, rumors, etc. At one point, the more and weirder data types I have at my disposal, the better. The reason is that I intend to systematically test combinations of data sources with price data, to find real inefficiencies (in a separate component/system/folder).<br>
**The goal isn't to beat institutions at their own game, but to find the games they're not playing**.


## Background
**Pure price analysis alone is mostly a dead end**, as institutions have been running the most sophisticated pattern recognition algorithms on this for decades. This project aims not to compete directly with them, as that would be a losing battle, but rather use the knowledge to our advantage and exploit small inefficiencies.<br>
Furthermore, **the markets are not made up of pure prices, but all kinds of data** (some/news sentiment, emotions/fear, weather, earnings, politics, law, etc.) which gives investors and system their reasons to buy or sell. This being, my feeling (also based on research) is that the markets cannot be fully efficient, since we live in a day and age where there are almost infinetly ways to get data and sources that can affect prices/markets.<br>
The real alpha is in the **intersections and combinations** that institutions struggle with due to organizational silos, data integartion challenges, and their focus on large-scale opportunities.


## Focus and purpose on "multi-data"
> Because of this Background, I really think that my competetive moat is an agile approach to multi-data.

1. **Data Integration Agility** - I can quickly combine (eg.):
    - Earnings surprises × social media sentiment × calendar timing
    - News sentiment × analyst revisions × option flow
    - Economic calendar × sector rotation × retail trading flows

2. **Weird Combination Testing** - Institutions often have teams that specialize in one data type. I can test bizarre combinations, like:
    - Weather data × retail stock performance × social media mentions
    - Congressional trading disclosures × small-cap momentum × earnings calendar
    - Supply chain disruption news × inventory levels × stock reactions

3. **Speed to New Data Sources** - When new alternative data becomes available, I can integrate and test it much faster than institutions with their compliance and approval processes.

**The key insight**: _Inefficiencies live in the GAPS between data types, not within pure price patterns_.

---

# Structure
```
├── README.md                       # This documentation
├── data_types_todo.md              # Light plan for sources to implement and why
│
├── ds_orchestrator.py              # An "orchestrator" to easily use singular or multiple data source components
│
├── combiners/                      # ✨ Data format combination system
│   ├── README.md                   # Detailed combiner documentation
│   ├── base_combiner.py            # Abstract base class for combiners
│   ├── time_series_combiner.py     # Main implementation: CSV + JSON → unified datasets
│   └── specialized_combiners/      # Future specialized combiners
│
├── combination_tester/             # 🎯 Systematic inefficiency discovery
│   ├── README.md                   # Detailed combination testing documentation
│   ├── combination_orchestrator.py # Main orchestrator for testing combinations
│   └── test_strategies/            # Different testing approaches
│       ├── statistical_strategy.py # Correlation + significance testing
│       └── future strategies...    # ML, custom strategies
│
└── sources/                        # All different data source implementations
    ├── ds_base.py                  # Base class all individual data source components implements
    │
    ├── calendar_events/            # Calendar/Event Data
    │   ├── README.md               # Detailed info regarding this data source implementation
    │   ├── cross_calendar.py       # main class?
    │   └── complexities/           # Individual "things" that will make up the calendar (eg. earnings, holidays, etc.) that is too "complex" for a single method
    │       ├── earnings.py         # ...
    │       └── options_expiration.py # ...
    │
    ├── .../
    │
    └── ...
```

---

# Plan


## For this component
My initial thought is that each component should have ways to get
- **Historical data**: for processing, training, analysis, testing combinations, etc.
- **Live data**: Almost to monitor features/data and get it when something changes. Similar to connecting to a Broker`s websocket and getting market prices every new candle. Some components might be able to get "new data" every minute (in pace with live market data), but others might not get new data as quickly (like waiting for new tweets or news reports) - as long as this is "reported" once new/fresh data is available.


## Future implementations and work
> Once some of the data-type/components are working, I need to start thinking/working on usage and incorporating this into my existing system.

# Usage Examples

## Quick Start
Run the multi-data sources orchestrator with your configuration:

```bash
# Standalone execution
python main.py --multi-data "multi_data_sources/example_config.yml"

# As part of hypothesis flow  
python main.py --hypothesis "multi_data_sources/example_config.yml"
```

## Configuration Structure
```yml
hypothesis:
  name: "multi_data_test"
  description: "Testing multi-data source integration"

multi_data:
    # Historical or "live" data
    purpose: "historical"       # or "live"

    # General details
    symbol: "AAPL"
    stock: true                 # bool
    from_date: "2022-01-01"     # YYYY-MM-DD
    to_date: "2025-01-01"       # YYYY-MM-DD

    # Data sources wanted
    sources: [  
        test_data               # Available: test_data
        # calendar_events       # TODO: Implement first data source
    ]

# Optional: Include other sections for full pipeline
# data_pipeline:
#   raw_source: "path/to/market/data.csv"
```

## Available Data Sources
- `test_data`: Demo data source generating dummy time series data
- `calendar_events`: ✅ **Complete** - Earnings, options expiry, holidays, fiscal quarters

## 🆕 Data Combination System
**Problem Solved**: How to combine CSV market data with JSON calendar events?

**Solution**: Use `combiners/TimeSeriesDataCombiner` to transform discrete events into continuous time-series features:

```python
from multi_data_sources.combiners import TimeSeriesDataCombiner

combiner = TimeSeriesDataCombiner()
combined_df = combiner.combine_data(
    symbol='GBPUSD',
    timeframe='1m', 
    date_range={'start': '2024-01-01', 'end': '2024-12-31'},
    data_sources=['calendar_events']
)
# Output: Time-indexed CSV with market data + calendar event features
```

**Key Benefits**:
- ✅ **Preserves original storage formats** - No need to change existing CSV/JSON structure
- ✅ **Time-aligned features** - Discrete events become continuous time-series features
- ✅ **Systematic testing** - Easy to test different data source combinations
- ✅ **Live trading ready** - Same logic works for real-time data
- ✅ **Reusable architecture** - Extensible for future data sources

## 🎯 Combination Testing System
**Problem**: How to systematically find market inefficiencies in data combinations?

**Solution**: Use `combination_tester/` to automatically test feature combinations and identify institutional blind spots:

```python
# Configuration-driven testing
python main.py --hypothesis "clov_combination_testing.yml"

# Results: Ranked combinations with statistical significance
# Example output:
# 🏆 Top combination results:
#   #1: market_plus_earnings_proximity (score: 0.8745, correlation: 0.6234)
#   #2: market_plus_multiple_calendar (score: 0.8234, correlation: 0.5891)
#   🎯 Identified 5 potential institutional blind spot combinations
```

**Configuration Example**:
```yml
combination_testing:
  enabled: true
  timeframe: "4h"                    # Dynamic timeframe support
  strategy: "statistical"            # or "ml_based", "custom"
  validation:
    min_correlation_threshold: 0.03
    significance_level: 0.05
```

**Features**:
- ✅ **Dynamic timeframes** - Works with 1m, 4h, 1d, etc.
- ✅ **Real calendar events** - Earnings, options expiry, holidays
- ✅ **Statistical validation** - Correlation + significance testing
- ✅ **Institutional blind spot scoring** - Cross-data-type combination ranking
- ✅ **Comprehensive results** - Storage, analysis, and ranked recommendations

## Integration with Main Pipeline

The orchestrator integrates seamlessly with your existing hypothesis-driven workflow:

1. **Standalone Mode**: `python main.py --multi-data config.yml`
2. **Hypothesis Flow**: Include `multi_data:` section in hypothesis config
3. **Pipeline Integration**: Results can feed into data processing and ML training

## Output Format

The orchestrator returns structured results:
```python
{
    'config': {...},           # Configuration used
    'sources': {               # Results from each data source
        'test_data': {
            'data': DataFrame,     # Historical data or subscription info
            'status': 'success',   # success/failed/subscribed  
            'timestamp': '...'     # Execution timestamp
        }
    },
    'summary': {
        'total_sources': 1,
        'successful_sources': 1,
        'failed_sources': 0,
        'successful_source_names': ['test_data'],
        'failed_source_names': []
    }
}
```

### Storage
For storing the "historical data" of different data types, this should be done in `storage/` (continuing with a centralized storage). The storage structure follows a **symbol-first** approach:

```
storage/multi_data_sources/SOURCE_TYPE/SYMBOL/COMPONENT/
```

**Example**: `storage/multi_data_sources/calendar_events/CLOV/earnings/raw/`

All storage paths are centrally managed via helper functions in `config/config.py` to avoid hardcoding paths across components.<br>

## 🎯 Next Steps: Systematic Inefficiency Discovery

### **Phase 1: Data Collection** ✅ Complete
- `sources/calendar_events/` - Institutional blind spots identified and implemented
- Storage structure optimized with symbol-first approach

### **Phase 2: Data Combination** ✅ Complete  
- `combiners/TimeSeriesDataCombiner` - CSV + JSON format bridging solved
- Time-indexed feature engineering operational
- Multiple combination testing framework ready

### **Phase 3: Systematic Testing** ✅ Complete
```bash
# Test different combinations to find real inefficiencies
python main.py --hypothesis "clov_4h_combination_testing.yml"

# Output: storage/combination_testing/SYMBOL/TIMESTAMP_STRATEGY/ with:
# - Ranked combinations with statistical significance
# - Top institutional blind spot opportunities
# - Raw combination datasets for further analysis
# - Comprehensive validation results
```

### **Phase 4: Pipeline Integration** 🔄 Next
1. **Enhanced Data Pipeline**: Integrate combiner with `data/pipelines/engineer_pipeline.py`
2. **Feature Selection**: Identify most predictive combinations from testing results  
3. **ML Training**: Use combined datasets for hypothesis-driven model training
4. **Live Trading**: Deploy same feature engineering logic for real-time execution

### **Your Competitive Moat is Now Operational** 🏆
- ✅ **Agile Multi-Data Integration** - Beat institutions at organizational flexibility
- ✅ **Weird Combination Testing** - Systematic approach to find gaps they miss  
- ✅ **Speed to New Data Sources** - Architecture ready for rapid expansion
- ✅ **Games They're Not Playing** - Small-cap calendar effects + institutional blind spots targeted

**The foundation for systematic inefficiency discovery is complete and ready for deployment!**
