# About strategy/
...


## Structure
```
strategy/
├── strategy.py                          # Main orchestrator
├── strategy_factory.py                  # Hypothesis-specific strategy selector
├── README.md                            # Documentation
│
├── hypothesis_specific/                 # Hypothesis-specific strategy implementations
│   ├── base_strategy.py                 # Abstract strategy interface
│   ├── session_anomaly_strategy.py      # Session-based anomaly strategy
│   └── price_prediction_strategy.py     # Price prediction strategy
│
├── components/                          # Core strategy components
│   ├── signal_processor.py             # Convert predictions → signals
│   ├── risk_manager.py                  # Risk rules & position sizing
│   ├── position_manager.py             # Portfolio state tracking
│   └── order_manager.py                 # Order execution & management
│
└── configs/                             # Strategy configuration templates
    ├── session_anomaly_config.yml       # Session anomaly strategy config
    └── price_prediction_config.yml      # Price prediction strategy config
```


## Key Design Principles 🧪

### 1. Hypothesis-Driven Strategy
- Each hypothesis gets its own strategy implementation
- Factory pattern for dynamic strategy selection
- Consistent with the evaluation architecture

### 2. Modular Components

- **Signal Processor**: Converts model predictions to actionable signals
- **Risk Manager**: Position sizing, stop-loss, drawdown limits
- **Position Manager**: Tracks portfolio state and open positions
- **Order Manager**: Executes and manages orders via broker

### 3. Configuration-First Approach

Following the hypothesis config pattern:
```yml
strategy:
    hypothesis_name: session_based_tod_anomalies
    model_folder: storage/models/2025-07-03_12:45:35

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
    execution_delay: 0.1
```

### 4. Integration Points

- Model Predictions → Signal Processing
- Broker API → Order Management
- Storage → Strategy results & performance
- main.py → Orchestration with -strategy flag

### 5. Storage Integration

Following the storage pattern:
```
storage/
├── strategy_results/
│   ├── <datetime>/
│   │   ├── strategy_metadata.yml
│   │   ├── performance_metrics.json
│   │   ├── trade_log.csv
│   │   └── positions_history.csv
```
