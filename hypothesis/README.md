# Hypothesis Testing System 🧪

## Overview 📊
This folder documents testing of different **market hypotheses** - systematic approaches to finding patterns, anomalies, **inefficiencies**, and repeatable opportunities in financial markets.

The fundamental premise: while beating the market consistently is extremely challenging, there may be higher chances of success by identifying and exploiting smaller, specific market inefficiencies through **hypothesis-driven trading strategies**.


## Structure 📁
```
hypothesis/
├── 1_session_based_tod_anomalies/          # Session-based time-of-day anomalies
│   ├── config.yml                          # Hypothesis configuration
│   └── README.md                           # Detailed hypothesis description
├── spread_compression_patterns/            # Spread compression trading (PRODUCTION-READY)
│   ├── config_complete.yml                 # Complete pipeline configuration
│   ├── config_backtest.yml                 # Backtesting configuration
│   └── README.md                           # Strategy documentation
└── README.md                               # This documentation
```

## Hypothesis Development Process 🔬

### Creating a New Hypothesis
Each hypothesis follows a **standardized structure** for consistency and integration:
1. Create a new folder named after your hypothesis.
2. Create a `config.yml` file for configuring the hypothesis and other potential parameters, following the given format.
3. Optionally, though recommended, a `README.md` describing the hypothesis in more detail and with better formulations.
4. Add a method in the (`data/features/`)`target_hypothesis.py` that calculates the target column, representing the hypothesis (that a model will predict).
5. If you want to be able to **evaluate the model** trained on your hypothesis, you must also create a **dedicated hypothesis evaluation file**, `evaluation/hypothesis_specific/`, and (same as with the calculation of the target) must name (the file) the same as hypothesis_name in the config-file (matching).
6. **For strategy execution support**: If you want to use the strategy component with your hypothesis, you must:
   - Create a **hypothesis-specific strategy implementation** in `strategy/hypothesis_specific/` that inherits from `BaseStrategy`
   - Add your hypothesis type and strategy class to the `STRATEGIES` dictionary in `strategy/strategy_factory.py`
   - Ensure your _hypothesis name matches exactly_ between the config file, target calculation method, evaluation file, and strategy factory

> [!IMPORTANT] 
> **Hypothesis Name Consistency**: The `name` field in your hypothesis config file must exactly match:
> - Function name in `data/features/target_hypothesis.py`
> - File name (without .py) in `evaluation/hypothesis_specific/` 
> - Key in `EVALUATORS` dictionary in `evaluation/evaluation_factory.py`
> - Key in `STRATEGIES` dictionary in `strategy/strategy_factory.py` 
> 
> **Example**: If your hypothesis is named `momentum_breakout`, ensure:
> ```python
> # In config.yml
> name: momentum_breakout
> 
> # In target_hypothesis.py  
> def momentum_breakout(self, df): # Method name matches
> 
> # In evaluation/hypothesis_specific/momentum_breakout.py # File name matches
> 
> # In evaluation_factory.py
> EVALUATORS = {
>     'momentum_breakout': MomentumBreakoutEvaluation,  # Key matches
> }
> 
> # In strategy_factory.py  
> STRATEGIES = {
>     'momentum_breakout': MomentumBreakoutStrategy,  # Key matches
> }
> ```

## config.yml structure - WIP
These are the different componentes/sections you can specify in the configuration-file. None of these are required (I think), but could be thought of as flags when running `main.py`, to tell the main orchestrator what components to run, and potentially specify how (features, parameters, etc.).
- **Data Pipeline**: 
- **Model Training**: 

This should serve as an example template to how features are selected to new hypotheses.
```yml
hypothesis:
    name: <name-of-the-hypothesis>

# Data Pipeline on raw data
data_pipeline:
    raw_source: <source-to-raw-data>                             # Must have
    spread_focused: false                                        # 🆕 True: Enable bid-ask spread data preservation (default: false)
  
    selected_features: null                                     # null = all features, or specify list:
                                                                # ['sma_20', 'rsi_14', 'atr'] = only these features
                                                                # 🆕 System now intelligently filters to only requested features!

# Train a ML Model
model_training:
    type: <name-of-ml-model>                          # Must have. Eg. 'xgboost', 'random_forest'

    processed_data: <path-to-processed-data>          # Optional: If this is not specified, I will assume you´ve ran Data Pipeline first (meaning using raw_source to get the processed)
    
    target_column: <target-column-name>               # Optional: Specific target column to use (e.g., 'target_spread_compression_binary')
                                                      # If not specified, automatic detection:
                                                      # - Classification models prefer binary targets  
                                                      # - Regression models prefer continuous targets

    parameters:                                       # Completely or somewhat empty means using 'default' hyperparameter values
        - n_estimators: 100
        - max_depth: 12

# Evaluate Trained Model
evaluate_model: 
    type: <stored-or-new>                             # Optional: Keeping it simple: 'stored' means already trained model, 'new' means you just ran model training 'in this run'
    model_folder: <path-to-processed-data>            # Only provide if you are evaluating a 'stored' model. Path to model folder

# Strategy Execution - Complete Trading System Configuration
strategy:
    hypothesis_name: session_based_tod_anomalies
    model_folder: storage/models/2025-07-03_12:45:35  # Optional if using strat_model_folder below
    strat_model_folder: storage/models/2025-07-03_12:45:35  # Strategy model location

    # Live data configuration
    data:
        symbol: GBPUSD                    # Trading symbol
        timeframe: MINUTE                 # Data resolution
        buffer_size: 500                  # Data points to maintain in buffer
        feature_window: 35                # Minimum points for feature calculation (>= 30)
        dry_run: true                     # Simulation mode (false for live trading)
        jumpstart: true                   # Preload buffer with historical data
        use_websocket: true               # WebSocket streaming (faster than REST)
        selected_features: null           # Specific features (null = all features)

    # Strategy execution parameters
    execution:
        max_loops: 100                    # Maximum strategy iterations
        loop_interval: 5                  # Seconds between strategy loops
        
    # Signal generation and filtering
    signal_processing:
        confidence_threshold: 0.7         # Minimum prediction confidence
        signal_filters:                   # Applied signal filters
            - session_filter              # Only trade during specific sessions
            - volatility_filter           # Filter based on market volatility

    # Risk management configuration
    risk_management:
        max_position_size: 0.02           # 2% of capital per position
        stop_loss_pct: 0.01              # 1% stop loss
        max_drawdown: 0.05               # 5% maximum portfolio drawdown
        position_sizing_method: fixed    # fixed, volatility_adjusted, kelly
        
    # Portfolio and position management
    position_management:
        max_concurrent_positions: 3       # Maximum simultaneous positions
        position_tracking: detailed       # detailed, basic
        correlation_check: true           # Check position correlations
        
    # Order execution configuration
    order_management:
        broker: capital_com               # Broker to use for execution
        order_type: market                # market, limit
        execution_delay: 0.1              # Delay between order components
        dry_run: true                     # Paper trading mode - false will place actual trades with broker
        slippage_buffer: 0.0001           # Price slippage buffer
        
    # Real-time monitoring
    monitoring:
        update_interval: 1.0              # Position update frequency (seconds)
        performance_tracking: true        # Track strategy performance
        alert_thresholds:                 # Alert configuration
            max_loss: 0.02                # Alert if position loss > 2%
            max_drawdown: 0.03            # Alert if drawdown > 3%
```
> [!IMPORTANT]
> The ```name``` of the hypothesis _must match_ the _function name_ in the ```data/hypothesis/target_hypothesis.py```. I plan to make it so you can specify the function-name dynamically in the Data Pipeline, but not yet.<br>
> Feature names in ```selected_features``` must match the actual feature column names generated by the ModelAwareFeatureGenerator. 

> [!NOTE]
> To use specific features, provide a list of exact feature names (e.g., ```['sma_20', 'rsi_14', 'atr']```). To generate all available features, use ```null```. The ModelAwareFeatureGenerator will automatically handle dependencies and generate features in the optimal order.

> [!NOTE]
> **🆕 Intelligent Feature Filtering**: The system now respects `selected_features` configuration precisely. When you request specific features, the system:
> - Runs required generators (which may produce multiple features)  
> - **Filters** the result to keep only your requested features + essential columns
> - Prevents feature bloat (e.g., requesting `london_session` no longer gives you all 7 session features)
> - Reduces datasets from (eg.) 57+ features to your focused selection (eg. ~25-37 features)

> [!NOTE]  
> **🆕 Spread-Focused Processing**: Set `spread_focused: true` in `data_pipeline` to:
> - Preserve bid-ask spread data during data ingestion
> - Enable spread compression analysis features
> - Support market microstructure hypotheses
> - Use with spread-related features like `'compression_ratio'`, `'bid_ask_spread_features'`

## Usage Examples 🚀

The hypothesis system supports a **complete end-to-end workflow** from data processing to live trading strategy execution. Configure and run the entire pipeline through the hypothesis configuration file.

### Complete Workflow Options

#### 1. Full Hypothesis Workflow (Recommended)
Configure all components in `config.yml` and run the complete pipeline:
```bash
# Run complete hypothesis workflow
python main.py -hyp 'hypothesis/1_session_based_tod_anomalies/config.yml'
```

This will execute configured components in order:
- **Data Pipeline** (if `data_pipeline` configured)
- **Model Training** (if `model_training` configured)  
- **Model Evaluation** (if `evaluate_model` configured)
- **Strategy Execution** (if `strategy` configured)

#### 2. Strategy Execution Only
Run just the trading strategy with a trained model:
```bash
# Run strategy execution only
python main.py -strat 'hypothesis/1_session_based_tod_anomalies/config.yml'
```

#### 3. Component-Specific Execution
Run individual components with hypothesis configuration (can drop first flag before `--hypothesis`):
```bash
# Data Pipeline only
python main.py -eg --hypothesis 'hypothesis/1_session_based_tod_anomalies/config.yml'

# Model Training only  
python main.py -train --hypothesis 'hypothesis/1_session_based_tod_anomalies/config.yml'

# Strategy execution only
python main.py -strat --hypothesis 'hypothesis/1_session_based_tod_anomalies/config.yml'
```

#### 🆕 4. Spread-Focused Hypothesis Example
```bash
# Run spread compression hypothesis with intelligent filtering
python main.py -hyp 'hypothesis/spread_compression_patterns/config_process_data.yml'
```

**Example Configuration:**
```yml
hypothesis:
  name: spread_compression_patterns

data_pipeline:
  raw_source: storage/data/capital_com/GBPUSD/raw/5m/2024-12-01_2025-01-23.csv
  spread_focused: true  # 🆕 Preserves bid-ask spread data

  selected_features:
    # Core spread features (auto-expanded from groups)
    - "bid_ask_spread_features"      # → bid_ask_spread_bps, bid_ask_spread_relative
    - "spread_compression_features"  # → compression_ratio, compression_consistency, etc.
    
    # Specific individual features
    - "london_session"              # Only london_session (not all 7 session features)
    - "volume_ma_20"
    - "mid_price"
```

**Result:** Focused 37-feature dataset instead of 57+ feature bloat

### Strategy Integration Examples

#### Live Trading Strategy
```python
from strategy.strategy import StrategyOrchestrator

# Initialize with hypothesis configuration
orchestrator = StrategyOrchestrator(
    hypothesis_config="hypothesis/1_session_based_tod_anomalies/config.yml"
)

# Execute trading strategy
results = orchestrator.run()

# Stop strategy gracefully
orchestrator.stop()
```

#### Simulation Mode
Set `dry_run: true` in the strategy configuration for paper trading:
```yml
strategy:
    data:
        dry_run: true          # Enable simulation mode
    order_management:
        dry_run: true          # Paper trading only
```

#### Custom Strategy Development
```python
import utils.hypothesis.config_loader as config_loader
from strategy.strategy_factory import StrategyFactory

# Load hypothesis configuration
config = config_loader.load_hypothesis_config(config_path)
processed_config = config.handle_hypothesis_source()

# Create strategy based on hypothesis
strategy = StrategyFactory.create_strategy(
    hypothesis_type=processed_config['name'],
    hypothesis_config=config_path
)

# Execute custom strategy logic
results = strategy.run()
```

## Best Practices 💡

### Development Guidelines
1. **🔍 Start with Exploration**: Analyze market data to identify potential patterns
2. **📊 Validate Hypotheses**: Use statistical tests to confirm patterns exist
3. **🤖 Model Development**: Train ML models on validated patterns
4. **📈 Backtesting**: Thorough historical testing before live deployment
5. **🎯 Live Testing**: Start with simulation mode, then small live positions

### Configuration Best Practices
1. **🔐 Start with Simulation**: Always set `dry_run: true` when testing strategies
2. **🧪 Incremental Testing**: Test components individually before full workflow
3. **🛡️ Risk Management**: Configure appropriate position sizing and stop losses
4. **✅ Model Validation**: Ensure model evaluation passes before strategy execution
5. **📊 Monitoring**: Enable performance tracking and alerts

### Integration Points 🔗
- **Data Pipeline**: Processes raw market data into ML-ready features
- **Model Training**: Trains ML models on hypothesis-specific targets
- **Backtesting**: Historical simulation using `backtesting/` component
- **Live Trading**: Real-time execution using `strategy/` component
- **Evaluation**: Performance analysis using `evaluation/` component

> [!NOTE]
> The centralized configuration approach eliminates the need to remember multiple command flags and ensures consistent component integration across the entire hypothesis workflow.