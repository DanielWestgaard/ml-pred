# Backtesting System 📊

## Overview 🎯
The backtesting system provides **rapid, accurate simulation** of trading strategies using historical data. It leverages a lot of existing strategy components to ensure similar behavior between backtesting and live trading (`strategy/`) while optimizing for speed and comprehensive analysis.

## Key Features 🚀
- **✨ Multi-Model Support**: Automatic detection and handling of both regression and classification models
- **🔄 Component Reuse**: Leverages existing `strategy/components` (`RiskManager`, `SignalProcessor`, `PositionManager`)
- **⚡ High Performance**: Complete backtests much faster than "real time waiting" with live testing
- **📈 Realistic Simulation**: Accurate transaction costs, slippage, and market impact modeling
- **🎯 Intelligent Signal Processing**: Binary classification optimized for spread compression strategies
- **🧠 Model-Driven Exits**: Positions close when model predicts compression period is over - TODO: MORE UNIFIED APPROACH FOR OTHER MODELS AND STRATEGIES
- **🎯 Perfect Feature Alignment**: Uses exact `model.feature_names_in_` for prediction consistency
- **📊 Comprehensive Analytics**: Performance metrics and risk analysis
- **🎛️ Easy Configuration**: Simple YAML configuration for rapid hypothesis testing
- **🔍 Detailed Logging**: Full trade-by-trade analysis and position tracking

## How It Works 🔄

The backtesting system simulates live trading by processing historical data step-by-step, using the same components as the live strategy:

### Data Flow Process
1. **Historical Data** → `BacktestDataManager` loads and iterates through time-series data
2. **Uses processed data** → For easier and faster use with focus on testing the actual model, the system will use processed (back-) testing data
3. **Model Prediction** → Trained ML model generates predictions on features
4. **Signal Processing** → `SignalProcessor` converts predictions to BUY/SELL signals
5. **Risk Management** → `RiskManager` validates position sizing and risk limits
6. **Trade Execution** → `SimulationBroker` executes trades with realistic costs
7. **Position Management** → `PositionManager` tracks open/closed positions
8. **Performance Analysis** → Comprehensive metrics and reporting

## Architecture 🏗️

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           BACKTESTING SYSTEM                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────────────┐  │
│ │ BacktestOrches- │  │ BacktestDataMgr  │  │   Trained ML Model      │  │
│ │    trator       │◄─┤  (Historical     │  │ • Random Forest         │  │
│ │ (Main Controller│  │   Data Stream)   │  │ • Model Metadata        │  │
│ │  & Orchestration│  └──────────────────┘  │ • Feature Names         │  │
│ │     Loop)       │                        └─────────────────────────┘  │
│ └─────────────────┘                                                     │
│          │                                                              │
│          ▼                                                              │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │                     REUSED STRATEGY COMPONENTS                      │ │
│ │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────────┐│ │
│ │  │ SignalProcessor │ │   RiskManager   │ │   PositionManager       ││ │
│ │  │ • Confidence    │ │ • Position Size │ │ • Open Positions        ││ │
│ │  │   Thresholds    │ │ • Stop Loss %   │ │ • Closed Positions      ││ │
│ │  │ • Signal        │ │ • Take Profit % │ │ • Portfolio Tracking    ││ │
│ │  │   Filtering     │ │ • Max Drawdown  │ │ • Position History      ││ │
│ │  └─────────────────┘ └─────────────────┘ └─────────────────────────┘│ │
│ └─────────────────────────────────────────────────────────────────────┘ │
│          │                                                              │
│          ▼                                                              │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │                    BACKTESTING-SPECIFIC LAYER                       │ │
│ │  ┌─────────────────┐ ┌──────────────────┐ ┌────────────────────────┐│ │
│ │  │ SimulationBroker│ │ Feature Generator│ │ Performance Analyzer   ││ │
│ │  │ • Instant Fills │ │ • Model-Aware    │ │ • Sharpe Ratio         ││ │
│ │  │ • Realistic     │ │ • Exact Feature  │ │ • Max Drawdown         ││ │
│ │  │   Costs (1.3pip)│ │   Alignment      │ │ • Win Rate             ││ │
│ │  │ • Slippage      │ │ • Buffer Mgmt    │ │ • Trade Analysis       ││ │
│ │  │   Modeling      │ │ • Time-Series    │ │ • Equity Curves        ││ │
│ │  └─────────────────┘ └──────────────────┘ └────────────────────────┘│ │
│ └─────────────────────────────────────────────────────────────────────┘ │
│                                    │                                    │
│                                    ▼                                    │
│                       ┌─────────────────────────┐                       │
│                       │     JSON Results        │                       │
│                       │ • Detailed Metrics      │                       │
│                       │ • Trade-by-Trade Log    │                       │
│                       │ • Performance Report    │                       │
│                       │ • Risk Analysis         │                       │
│                       └─────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Relationships

- **BacktestOrchestrator**: Central coordinator that manages the simulation loop
- **BacktestDataManager**: Streams historical data, maintains buffer for feature generation
- **Strategy Components**: **90% code reuse** from `strategy/components/`
  - `SignalProcessor`: Converts model predictions → actionable signals
  - `RiskManager`: Validates position sizing, stop-loss, take-profit
  - `PositionManager`: Tracks portfolio state and position lifecycle
- **SimulationBroker**: Replaces live broker API with instant execution + realistic costs
- **Performance Analyzer**: Generates comprehensive trading performance metrics

## Structure 📁

```
backtesting/
├── backtester.py                        # BacktestOrchestrator - Main controller
├── simulation_broker.py                 # SimulationBroker - Instant fills with realistic costs
├── backtest_data_manager.py             # BacktestDataManager - Historical data iteration  
├── performance_analyzer.py              # BacktestPerformanceAnalyzer - Comprehensive analytics
└── README.md                            # This documentation
```

## Configuration ⚙️

Add a `backtesting` section to your hypothesis configuration:

```yaml
# hypothesis/spread_compression_patterns/config_backtest.yml
hypothesis:
  name: spread_compression_patterns

backtesting:
    enabled: true                                     # Enable backtesting mode
    model_folder: storage/models/2025-07-27_12:53:02  # Trained model path
    
    # Data configuration
    data_source: storage/data/capital_com/GBPUSD/processed/1m/2024-06-01_2025-05-23.csv
    start_date: '2025-02-03'                          # Signal-rich period for testing
    end_date: '2025-02-04'                            # 1-2 days for quick testing
    buffer_size: 200                                  # Historical buffer for feature generation
    
    # Capital and costs
    initial_capital: 10000                            # Starting capital for simulation
    transaction_costs: 0.0013                         # 1.3 pips for GBPUSD (realistic spread)
    slippage: 0.0002                                  # 0.2 pips average slippage
    
    # Execution settings
    execution:
        max_loops: 2000                               # Limit for quick testing
        loop_interval: 0                              # No delays in backtesting - TODO: Is this even worth having in the code?
    
    # Signal processing (optimized for spread compression)
    signal_processing:
        # Basic vs Advanced filtering
        use_advanced_filtering: true           # Enable AdvancedSignalProcessor with intelligent filtering
        processor_type: advanced               # Options: 'basic', 'advanced', 'smart', 'ultra'
        
        # Core signal thresholds
        confidence_threshold: 0.50             # Minimum model confidence (0.0-1.0) to generate signals
        min_confidence_delta: 0.01             # Minimum confidence change to override cooldowns
        min_signal_strength: 0.08              # Minimum signal strength after processing
        
        # Time-based filtering (prevents over-trading)
        cooldown_minutes: 0.25                 # Wait time between signals (15 seconds)
        max_signals_per_hour: 240              # Maximum signals per hour limit
        confidence_window_minutes: 10          # Time window for confidence analysis
        
        # Advanced filters (list of filter names to apply)
        signal_filters: ['session_filter']     # Options: ['session_filter', 'volatility_filter', 'volume_filter']
        
        # Additional confirmations
        volume_confirmation_required: false    # Require volume confirmation for signals
    
    # Risk management (optimized for microstructure trading)
    risk_management:
        max_position_size: 0.02                       # 2% of capital per position
        stop_loss_pct: 0.02                           # 2% stop loss (microstructure appropriate)
        take_profit_pct: 0.04                         # 4% take profit (2:1 risk/reward)
        max_drawdown: 0.05                            # 5% maximum drawdown limit
        max_concurrent_positions: 3                   # Maximum 3 simultaneous positions
        
        # Position sizing methods
        position_size_scaling: confidence             # Options: 'fixed', 'confidence', 'volatility', 'kelly'

    
    # Position management
    position_management:
        max_concurrent_positions: 3                   # Consistent with risk management
        position_tracking: detailed                   # Detailed position tracking for analysis
```

### Configuration Details 📋

#### Signal Processing Options

**Processor Types:**
- `basic`: Simple confidence-based filtering with basic thresholds
- `advanced`: Intelligent filtering with cooldowns and rate limiting
- `smart`: Hybrid approach with directional change detection
- `ultra`: Maximum permissiveness for high-frequency strategies

**Signal Filters Available:**
- `session_filter`: Only trade during preferred market sessions (London/NY overlap)
- `volatility_filter`: Filter signals based on market volatility conditions
- `volume_filter`: Require volume confirmation for signal validation

**Example Configurations:**

```yaml
# Conservative trading (few high-quality signals)
signal_processing:
    processor_type: advanced
    confidence_threshold: 0.70
    cooldown_minutes: 5
    max_signals_per_hour: 12
    signal_filters: ['session_filter', 'volatility_filter']

# Aggressive trading (more frequent signals)
signal_processing:
    processor_type: ultra
    confidence_threshold: 0.30
    cooldown_minutes: 0.1
    max_signals_per_hour: 300
    signal_filters: []

# Balanced approach (recommended for most strategies)
signal_processing:
    processor_type: smart
    confidence_threshold: 0.50
    cooldown_minutes: 1.0
    max_signals_per_hour: 60
    signal_filters: ['session_filter']
```

#### Position Sizing Methods

**Available Options:**
- `fixed`: Fixed percentage of capital per trade (uses `max_position_size`)
- `confidence`: Scale position size by signal confidence (higher confidence = larger position)
- `volatility`: Adjust position size based on market volatility (more volatile = smaller position)
- `kelly`: Kelly Criterion-based position sizing (requires win rate history)

**Examples:**

```yaml
# Fixed sizing (2% per trade regardless of signal strength)
risk_management:
    position_size_scaling: fixed
    max_position_size: 0.02

# Confidence-based scaling (0.5-4% based on signal confidence)
risk_management:
    position_size_scaling: confidence
    max_position_size: 0.04  # Maximum for highest confidence signals
    min_position_size: 0.005 # Minimum for lowest confidence signals

# Volatility-adjusted sizing (smaller positions in volatile markets)
risk_management:
    position_size_scaling: volatility
    max_position_size: 0.03
    volatility_lookback: 20  # Rolling window for volatility calculation
```

## Usage Examples 🚀

### Running a Backtest

```python
from backtesting.backtester import BacktestOrchestrator

# Initialize with hypothesis configuration
backtester = BacktestOrchestrator(
    hypothesis_config_path="hypothesis/spread_compression_patterns/config_complete.yml"
)

# Run the backtest
results = backtester.run_backtest()

# Print summary
print(f"Total Return: {results['backtest_summary']['total_return_pct']:.2f}%")
print(f"Sharpe Ratio: {results['risk_metrics']['sharpe_ratio']:.3f}")
print(f"Max Drawdown: {results['trading_performance']['max_drawdown_pct']:.2f}%")
```

### Using with Specific Model

```python
# Test a specific trained model
backtester = BacktestOrchestrator(
    model_folder="storage/models/2025-07-27_12:53:02"
)

results = backtester.run_backtest()
```

### Advanced Performance Analysis

```python
from backtesting.performance_analyzer import BacktestPerformanceAnalyzer

# Load and analyze results
analyzer = BacktestPerformanceAnalyzer(
    results_file="backtesting/results/backtest_results_20250727_143022.json"
)

# Generate comprehensive report
report = analyzer.generate_performance_report()
print(report)

# Create visualizations
analyzer.create_equity_curve("equity_curve.png")

# Export detailed analysis
files = analyzer.export_detailed_analysis("analysis_output/")
```

## Component Details 🧩

### BacktestOrchestrator (`backtester.py`)
**Main backtesting controller that coordinates all components**
- Loads hypothesis configuration and trained models
- Manages historical data iteration and feature generation  
- Orchestrates signal processing, risk management, and trade execution
- Calculates comprehensive performance metrics and saves results

**Key Methods:**
- `run_backtest()`: Execute complete backtesting simulation
- `get_progress()`: Monitor backtesting progress in real-time
- `stop()`: Gracefully stop backtesting

### SimulationBroker (`simulation_broker.py`)
**Realistic broker simulation with instant fills**
- **Instant Order Execution**: No network delays or API limitations
- **Realistic Transaction Costs**: Configurable spreads and commissions
- **Slippage Modeling**: Random slippage within realistic ranges
- **Position Tracking**: Full trade lifecycle management
- **Performance Metrics**: Real-time equity and risk tracking

**Key Features:**
- Automatic position sizing with risk management
- Stop loss and take profit execution
- Comprehensive trade logging
- Realistic cost modeling for GBPUSD (1.3 pips spread)

### BacktestDataManager (`backtest_data_manager.py`)
**Historical data iteration and management**
- **Data Buffer Management**: Maintains historical context for feature generation
- **Temporal Data Flow**: Simulates real-time data streaming with historical data
- **Date Range Filtering**: Configurable start/end dates for testing periods
- **Progress Tracking**: Real-time progress monitoring and statistics

**Key Features:**
- Efficient data iteration with configurable buffer sizes
- Support for temporal train/test splits
- Data quality validation and missing value handling

### BacktestPerformanceAnalyzer (`performance_analyzer.py`)
**Comprehensive performance analysis and reporting**
- **Advanced Metrics**: Sharpe ratio, Sortino ratio, profit factor, drawdown analysis
- **Visual Analytics**: Equity curves, monthly performance, trade distribution
- **Risk Analysis**: Detailed risk-adjusted performance metrics
- **Comparative Analysis**: Benchmark comparison and performance rating

**Key Features:**
- Automated performance reports with key insights
- Monthly and yearly performance breakdowns
- Trade pattern analysis and exit reason categorization
- Export capabilities for detailed analysis

## Performance Metrics 📊

The backtesting system provides comprehensive performance analysis:

### Financial Metrics
- **Total Return**: Absolute and percentage returns
- **Annualized Return**: Risk-adjusted annual performance
- **Monthly Return Rate**: Consistent performance tracking
- **Profit Factor**: Ratio of gross profits to gross losses

### Risk Metrics  
- **Sharpe Ratio**: Risk-adjusted returns vs volatility
- **Sortino Ratio**: Downside risk-adjusted returns
- **Maximum Drawdown**: Peak-to-trough equity decline
- **Value at Risk (VaR)**: Potential loss estimation

### Trading Metrics
- **Win Rate**: Percentage of profitable trades
- **Average Holding Period**: Trade duration analysis
- **Trade Frequency**: Daily/weekly trading activity
- **Exit Reason Analysis**: Stop loss vs take profit patterns

### Advanced Analytics
- **Consecutive Win/Loss Streaks**: Risk of ruin analysis
- **Monthly Performance Breakdown**: Seasonal pattern detection
- **Benchmark Comparison**: Relative performance analysis
- **Performance Rating**: Automated strategy scoring (1-10 scale)

## Integration with Existing System 🔌

The backtesting system seamlessly integrates with your existing project:

### Strategy Components (90% Reuse)
- ✅ **RiskManager**: Identical position sizing and risk controls
- ✅ **SignalProcessor**: Same signal generation and filtering logic  
- ✅ **PositionManager**: Consistent position tracking and management
- ✅ **Feature Generation**: UnifiedFeatureGenerator ensures feature consistency

### Configuration System
- ✅ **YAML Configuration**: Extends existing hypothesis config format
- ✅ **Model Integration**: Automatic model loading from training results
- ✅ **Data Pipeline**: Uses same processed data as model training

### Results Storage
- ✅ **Model Directory**: Results saved to `storage/models/<model>/backtesting/`
- ✅ **JSON Format**: Structured results for programmatic analysis
- ✅ **Trade Logging**: Detailed trade-by-trade records

## Best Practices 💡

### Configuration Guidelines
1. **Start Date**: Use end of training period to avoid look-ahead bias, and make sure this is not data the model is trained on
2. **Transaction Costs**: Use realistic spreads for your broker/symbol
3. **Position Sizing**: Conservative sizing (1-2%) for initial testing
4. **Buffer Size**: Minimum 200 periods for stable feature generation

### Performance Analysis
1. **Multiple Timeframes**: Test different market periods (trending vs ranging)
2. **Walk-Forward Testing**: Use rolling windows for robust validation
3. **Risk-Adjusted Metrics**: Focus on Sharpe ratio and drawdown, not just returns
4. **Signal Quality**: Analyze exit reasons to optimize stop loss/take profit levels

### Model Validation
1. **Out-of-Sample Testing**: Always test on unseen data
2. **Feature Stability**: Monitor feature importance consistency
3. **Signal Degradation**: Watch for performance decay over time
4. **Market Regime Changes**: Test across different market conditions
