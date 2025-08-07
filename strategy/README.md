# Live Trading Strategy System 🎯

## Overview 📈
The strategy component is a **complete live trading system** that transforms ML model predictions into actionable trading decisions through a **hypothesis-driven approach**. This system provides both **simulation and live trading capabilities**, seamlessly integrating all project components including real-time data management, model predictions, risk management, and broker execution.

## Key Features 🚀
- **🎯 Full Trading System**: Complete end-to-end from data ingestion to trade execution
- **🤖 Multi-Model Support**: Automatic detection and handling of both regression and classification models
- **📡 Live & Simulation Modes**: Support for both paper trading (simulation) and live market execution
- **⚡ Real-time Data Integration**: Live market data streaming via WebSocket and REST APIs (currently only through Capital.com's websocket API)
- **🧠 Hypothesis-Driven Architecture**: Each market hypothesis gets its own specialized strategy implementation
- **📊 Enhanced Signal Processing**: Probability-based confidence for classification, magnitude-based for regression
- **🛡️ Comprehensive Risk Management**: Multi-layer risk controls with position sizing, stop-losses, and drawdown protection
- **🔧 Automated Account Management**: Centralized account switching with configurable target accounts
- **🚨 Live Trading Safety**: Automatic position closure during system shutdown for live trading protection
- **🔗 Component Integration**: Seamlessly ties together `data/`, `broker/`, `model/`, and `evaluation/` components

## How It Works 🔄

The strategy system processes live market data in real-time, using trained ML models to generate trading signals:

### Live Trading Flow
1. **Real-time Data** → Live market data streams via WebSocket/REST APIs
2. **Feature Generation** → Processes live data into features for ML model
3. **Model Prediction** → Trained ML model generates predictions on live features
4. **Signal Processing** → Converts predictions into BUY/SELL/HOLD signals
5. **Risk Management** → Validates signals against risk parameters
6. **Trade Execution** → Executes trades through broker API (Capital.com)
7. **Position Management** → Tracks and monitors live positions
8. **Performance Tracking** → Records all trading activity and performance

## Architecture Overview 🏗️
The strategy system follows a **3-tier architecture**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       LIVE TRADING STRATEGY SYSTEM                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────────────┐  │
│ │ StrategyOrches- │  │ StrategyFactory  │  │   Hypothesis Config     │  │
│ │    trator       │◄─┤ (Determines      │  │ • Session Anomalies     │  │
│ │ (Main Entry     │  │  Strategy Type)  │  │ • Price Prediction      │  │
│ │  Point & Life-  │  └──────────────────┘  │ • Spread Compression    │  │
│ │  cycle Mgmt)    │                        └─────────────────────────┘  │
│ └─────────────────┘                                                     │
│          │                                                              │
│          ▼                                                              │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │                    HYPOTHESIS-SPECIFIC STRATEGIES                   │ │
│ │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────────┐│ │
│ │  │  BaseStrategy   │ │SessionAnomaly   │ │SpreadCompression        ││ │
│ │  │ (Abstract Base) │ │   Strategy      │ │   Strategy              ││ │
│ │  │ • Model Loading │ │ • TOD Anomalies │ │ • PRODUCTION-READY      ││ │
│ │  │ • Component     │ │ • Session Logic │ │ • 100% OOB Accuracy     ││ │
│ │  │   Initialization│ │ • Volume Filters│ │ • Perfect RF Model      ││ │
│ │  └─────────────────┘ └─────────────────┘ └─────────────────────────┘│ │
│ └─────────────────────────────────────────────────────────────────────┘ │
│          │                                                              │
│          ▼                                                              │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │                        STRATEGY COMPONENTS                          │ │
│ │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────────┐│ │
│ │  │ SignalProcessor │ │   RiskManager   │ │   PositionManager       ││ │
│ │  │ • Model Type    │ │ • Position Size │ │ • Portfolio State       ││ │
│ │  │   Auto-Detection│ │ • Stop Loss %   │ │ • Position Tracking     ││ │
│ │  │ • Confidence    │ │ • Take Profit % │ │ • Performance Metrics   ││ │
│ │  │   Thresholds    │ │ • Drawdown Ctrl │ │ • Real-time Updates     ││ │
│ │  └─────────────────┘ └─────────────────┘ └─────────────────────────┘│ │
│ └─────────────────────────────────────────────────────────────────────┘ │
│          │                                                              │
│          ▼                                                              │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │                     LIVE TRADING LAYER                              │ │
│ │  ┌─────────────────┐ ┌──────────────────┐ ┌────────────────────────┐│ │
│ │  │  OrderManager   │ │  LiveDataManager │ │  PositionMonitor       ││ │
│ │  │ • Broker API    │ │ • WebSocket      │ │ • Real-time P&L        ││ │
│ │  │ • Capital.com   │ │ • Live Streaming │ │ • Risk Monitoring      ││ │
│ │  │ • Auto-Sizing   │ │ • Buffer Mgmt    │ │ • Performance Tracking ││ │
│ │  │ • Safety Stops  │ │ • Error Handling │ │ • Alert System         ││ │
│ │  └─────────────────┘ └──────────────────┘ └────────────────────────┘│ │
│ └─────────────────────────────────────────────────────────────────────┘ │
│                                    │                                    │
│                                    ▼                                    │
│                       ┌─────────────────────────┐                       │
│                       │     Live Results        │                       │
│                       │ • Real-time Trading     │                       │
│                       │ • Position Tracking     │                       │
│                       │ • Performance Metrics   │                       │
│                       │ • Safety & Monitoring   │                       │
│                       └─────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────────────┘
```

### Architecture Layers

1. **Orchestration Layer**: `StrategyOrchestrator` (strategy.py:11) - Main entry point that determines and runs the appropriate strategy
2. **Strategy Layer**: Hypothesis-specific implementations (`SessionAnomalyStrategy`, `PricePredictionStrategy`, `SpreadCompressionStrategy`) 
3. **Component Layer**: Modular components for signal processing, risk management, position tracking, and order execution
4. **Live Trading Layer**: Real-time data streams, broker integration, and live position monitoring

## Structure
```
strategy/
├── strategy.py                          # StrategyOrchestrator - Main orchestrator & entry point
├── strategy_factory.py                  # Factory pattern for hypothesis-specific strategy selection
├── README.md                            # Documentation
│
├── hypothesis_specific/                 # Hypothesis-specific strategy implementations
│   ├── base_strategy.py                 # BaseStrategy - Abstract base class with shared functionality
│   ├── session_anomaly_strategy.py      # SessionAnomalyStrategy - Session-based anomaly detection
│   ├── price_prediction_strategy.py     # PricePredictionStrategy - Price prediction strategy
│   └── spread_compression_strategy.py   # SpreadCompressionStrategy - Spread compression
│
├── components/                          # Modular strategy components
│   ├── signal_processor.py              # SignalProcessor - Convert predictions → actionable signals
│   ├── risk_manager.py                  # RiskManager - Risk rules & position sizing
│   ├── position_manager.py              # PositionManager - Portfolio state & position tracking
│   └── order_manager.py                 # OrderManager - Order execution & management
│
└── monitoring/                          # Strategy monitoring and performance tracking
    └── position_monitor.py              # PositionMonitor - Real-time position monitoring
```


## How the Strategy System Works 🔄

### 1. Strategy Orchestration Flow
1. **StrategyOrchestrator** determines hypothesis type from config or model metadata
2. **StrategyFactory** creates the appropriate hypothesis-specific strategy
3. **BaseStrategy** provides common functionality (model loading, component initialization)
4. **Hypothesis-specific strategy** implements custom trading logic
5. **Components** handle signal processing, risk management, and execution

### 2. Real-time Trading Loop
```python
# Simplified trading loop flow
while trading_active:
    # 1. Get live market data
    market_data = live_data_manager.get_latest_data()
    
    # 2. Generate features and predictions
    features = feature_pipeline.process(market_data)
    prediction = model.predict(features)
    
    # 3. Generate trading signals
    signals = signal_processor.process(prediction, market_data)
    
    # 4. Apply risk management
    validated_signals = risk_manager.validate(signals)
    
    # 5. Execute trades
    orders = order_manager.execute(validated_signals)
    
    # 6. Update position tracking
    position_manager.update(orders, market_data)
```

## Configuration ⚙️

Strategy configuration is embedded in hypothesis config files:

```yaml
strategy:
    hypothesis_name: session_based_tod_anomalies
    model_folder: storage/models/2025-07-03_12:45:35

    # Data stream configuration
    data:
        symbol: GBPUSD                    # Trading pair
        timeframe: MINUTE                 # Data timeframe (MINUTE, HOUR, DAY)
        buffer_size: 500                  # Historical data buffer for feature generation
        dry_run: true                     # true = simulation, false = live trading
        use_websocket: true               # Enable live WebSocket data streaming
        jumpstart: true                   # Preload historical data buffer (essential)

    # Execution settings
    execution:
        max_loops: 100                    # Maximum trading iterations
        loop_interval: 5                  # Seconds between trading cycles

    # Signal processing (converts ML predictions to trading signals)
    signal_processing:
        confidence_threshold: 0.3         # Minimum model confidence for signal generation
        signal_filters: ['session_filter', 'volatility_filter']  # Available filters
        model_type: 'auto'                # 'auto', 'regression', 'binary_classification'

    # Risk management (position sizing and risk controls)
    risk_management:
        max_position_size: 0.02           # 2% of capital per position
        stop_loss_pct: 0.01               # 1% stop loss
        take_profit_pct: 0.02             # 2% take profit (optional)
        max_drawdown: 0.05                # 5% maximum portfolio drawdown
        max_concurrent_positions: 3        # Maximum simultaneous positions

    # Position tracking and management
    position_management:
        max_concurrent_positions: 3        # Must match risk_management setting
        position_tracking: detailed        # 'basic' or 'detailed'

    # Order execution (broker integration)
    order_management:
        broker: capital_com               # Broker API to use
        order_type: market                # 'market' or 'limit'
        dry_run: false                    # false = live trading, true = simulation
        execution_delay: 0.1              # Delay between order placement (seconds)

    # Real-time monitoring
    monitoring:
        update_interval: 1.0              # Position update frequency (seconds)
```

### Configuration Details 📋

#### Trading Modes
- **Live Trading**: `dry_run: false` + `use_websocket: true` - Real trades with live data
- **Paper Trading**: `dry_run: true` + `use_websocket: true` - Simulated trades with live data
- **Backtest Mode**: Use `backtesting/` component for historical simulation

#### Signal Processing Options
- **Model Types**: `auto` (recommended), `regression`, `binary_classification`, `multiclass_classification`
- **Available Filters**: `session_filter`, `volatility_filter`, `volume_filter`
- **Confidence Thresholds**: 0.3-0.7 for classification, 0.1-0.5 for regression

#### Risk Management Modes
- **Conservative**: Small position sizes (1-2%), tight stops, multiple filters
- **Moderate**: Medium position sizes (2-3%), balanced risk/reward
- **Aggressive**: Larger positions (3-5%), wider stops, fewer filters

**Example Configurations:**

```yaml
# Conservative live trading
strategy:
    data:
        dry_run: false
        use_websocket: true
    signal_processing:
        confidence_threshold: 0.7
        signal_filters: ['session_filter', 'volatility_filter']
    risk_management:
        max_position_size: 0.01
        stop_loss_pct: 0.008

# Aggressive simulation
strategy:
    data:
        dry_run: true
        use_websocket: true
    signal_processing:
        confidence_threshold: 0.3
        signal_filters: []
    risk_management:
        max_position_size: 0.05
        stop_loss_pct: 0.02
```

## Component Details 🧩

### StrategyOrchestrator (`strategy.py`)
**Main entry point for strategy execution**
- Determines hypothesis type from config or model metadata  
- Creates appropriate strategy via StrategyFactory
- Provides graceful start/stop functionality
- Handles errors and cleanup

### BaseStrategy (`hypothesis_specific/base_strategy.py`)
**Abstract base class with shared functionality**
- Model and metadata loading
- Strategy configuration management
- Component initialization (signal processor, risk manager, etc.)
- Common results saving logic
- Template methods for hypothesis-specific implementations

### Hypothesis-Specific Strategies
- **SessionAnomalyStrategy**: Implements session-based volume anomaly detection with live data streaming
- **PricePredictionStrategy**: Implements price prediction based trading  
- **SpreadCompressionStrategy**: ⭐ **PRODUCTION-READY** - Implements spread compression trading with perfect Random Forest model (100% OOB accuracy)
- ...
- Each strategy inherits from BaseStrategy and implements `generate_signals()` and `execute_strategy()`

### Strategy Components
- **✨ Enhanced SignalProcessor**: Converts model predictions into actionable trading signals with automatic model type detection
  - **Classification Support**: Uses `predict_proba()` for confidence scores
  - **Regression Support**: Uses prediction magnitude for confidence
  - **Auto-Detection**: Automatically determines model type from model object or metadata
- **RiskManager**: Implements position sizing, stop-losses, and portfolio-level risk controls
- **PositionManager**: Tracks portfolio state, open positions, and performance metrics
- **OrderManager**: Executes trades through broker APIs with automatic position sizing, centralized account management, and live trading safety features

## Usage Examples 🚀

### Running a Strategy

#### Spread Compression Strategy (Production-Ready)
```bash
# Run complete spread compression pipeline
python main.py -hyp "hypothesis/spread_compression_patterns/config_complete.yml"

# Run backtesting with trained model
python main.py -hyp "hypothesis/spread_compression_patterns/config_backtest.yml"

# Run strategy with spread compression config
python main.py -strat "hypothesis/spread_compression_patterns/config_complete.yml"
```

#### Other Strategies
```bash
# Session anomaly strategy
python main.py -strat "hypothesis/1_session_based_tod_anomalies/config.yml"

# Run with any hypothesis configuration
python main.py -hyp 'path/to/hypothesis/config.yml'
```

### Programmatic Usage
```python
from strategy.strategy import StrategyOrchestrator

# Initialize with hypothesis config
orchestrator = StrategyOrchestrator(
    hypothesis_config="hypothesis/1_session_based_tod_anomalies/config.yml"
)

# Run the strategy
results = orchestrator.run()

# Stop gracefully
orchestrator.stop()
```

### Custom Strategy Development
```python
from strategy.hypothesis_specific.base_strategy import BaseStrategy

class MyCustomStrategy(BaseStrategy):
    def generate_signals(self, market_data):
        # Implement custom signal generation logic
        pass
    
    def execute_strategy(self):
        # Implement custom strategy execution logic
        pass
```

## Integration Points 🔌
- **Data Component**: Live data streaming via `LiveDataManager`
- **Model Component**: Model loading and predictions via `model_metadata_loader`
- **Broker Component**: Trade execution via broker APIs
- **Evaluation Component**: Performance tracking and analysis
- **Storage**: Results and performance metrics saved to `storage/models/<model>/strategy/`

## Live Trading Safety Features 🛡️

### Automatic Position Closure
When running in live trading mode (`dry_run: false`), the system _automatically closes all open positions during shutdown_:

```python
# Triggered on Ctrl+C or system shutdown
# Only in live trading mode - dry run is unaffected
logging.info("🔄 Live trading cleanup: Closing all open positions...")
broker.close_all_orders()
logging.info("✅ All positions successfully closed during cleanup")
```

### Account Management
- **Centralized Account Switching**: All broker sessions automatically switch to your configured test account
- **Consistent Account Usage**: Ensures all components (data, trading, monitoring) use the same account
- **Configurable Target Account**: Set via `config.VAR_ACCOUNT_NAME_TEST` (default: "USD_testing")

### Position Sizing
- **Automatic BTCUSD Sizing**: Intelligent size calculation for Bitcoin pairs (minimum 0.01 BTC)
- **Currency-Specific Logic**: Proper unit conversion for different asset classes
- **Risk-Based Sizing**: Position sizes calculated based on risk management parameters

## Results & Performance Tracking 📊
Strategy results are automatically saved:
```
storage/models/<model_timestamp>/strategy/
├── strategy_results.json      # Trading performance metrics
└── position_logs/             # Detailed position tracking
```
