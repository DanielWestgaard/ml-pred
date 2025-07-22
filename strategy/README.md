# About strategy/ 🎯

## Overview 📈
The strategy component is a **complete trading system orchestrator** that transforms ML model predictions into actionable trading decisions through a **hypothesis-driven approach**. This system provides both **simulation and live trading capabilities**, seamlessly integrating all project components including data management, model predictions, risk management, and broker execution.

## Key Features 🚀
- **Full Trading System**: Complete end-to-end trading system from data ingestion to trade execution
- **Live & Simulation Modes**: Support for both paper trading (simulation) and live market execution
- **Real-time Data Integration**: Live market data streaming via WebSocket and REST APIs
- **Hypothesis-Driven Architecture**: Each market hypothesis gets its own specialized strategy implementation. This must be manually implemented for the hypothesis.
- **Comprehensive Risk Management**: Multi-layer risk controls with position sizing, stop-losses, and drawdown protection
- **Automated Account Management**: Centralized account switching with configurable target accounts
- **Live Trading Safety**: Automatic position closure during system shutdown for live trading protection
- **Component Integration**: Seamlessly ties together `data/`, `broker/`, `model/`, and `evaluation/` components

## Architecture Overview 🏗️
The strategy system follows a **3-tier architecture**:

1. **Orchestration Layer**: `StrategyOrchestrator` - Main entry point that determines and runs the appropriate strategy
2. **Strategy Layer**: Hypothesis-specific implementations (`SessionAnomalyStrategy`, `PricePredictionStrategy`) 
3. **Component Layer**: Modular components for signal processing, risk management, position tracking, and order execution

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
│   └── price_prediction_strategy.py     # PricePredictionStrategy - Price prediction strategy
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

### 3. Configuration-First Approach
Strategy configuration is embedded in hypothesis config files:
```yml
strategy:
    hypothesis_name: session_based_tod_anomalies
    model_folder: storage/models/2025-07-03_12:45:35

    data:
        symbol: GBPUSD
        timeframe: MINUTE
        buffer_size: 500
        dry_run: true  # Simulation mode
        use_websocket: true  # Subscribe to live market data from capital.com
        jumpstart: true  # Preload historical data - a "must-have"

    execution:
        max_loops: 100
        loop_interval: 5

    signal_processing:
        confidence_threshold: 0.7
        signal_filters: ['session_filter', 'volatility_filter']

    risk_management:
        max_position_size: 0.02  # 2% of capital
        stop_loss_pct: 0.01      # 1% stop loss
        max_drawdown: 0.05       # 5% max drawdown

    position_management:
        max_concurrent_positions: 3
        position_tracking: detailed

    order_management:
        broker: capital_com
        order_type: market
        dry_run: false  # false = live trading, true = simulation mode
        execution_delay: 0.1

    monitoring:
        update_interval: 1.0
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
- ...
- Each strategy inherits from BaseStrategy and implements `generate_signals()` and `execute_strategy()`

### Strategy Components
- **SignalProcessor**: Converts model predictions into actionable trading signals
- **RiskManager**: Implements position sizing, stop-losses, and portfolio-level risk controls
- **PositionManager**: Tracks portfolio state, open positions, and performance metrics
- **OrderManager**: Executes trades through broker APIs with automatic position sizing, centralized account management, and live trading safety features

## Usage Examples 🚀

### Running a Strategy
```bash
# Run strategy with hypothesis configuration
python main.py -strat "hypothesis/1_session_based_tod_anomalies/config.yml"
# .. or ..
python main.py -hyp 'hypothesis/1_session_based_tod_anomalies/config_fixed.yml'

# Run strategy with specific model
python main.py -strat --model-dir "storage/models/2025-07-03_12:45:35"
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
