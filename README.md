# ML-Driven Trading System 🚀

A **complete hypothesis-driven trading platform** that discovers and exploits market inefficiencies through machine learning. The system focuses on finding micro-opportunities that large institutions ignore due to scalability constraints.

## Key Philosophy 💡
Rather than competing directly with institutional players, this system leverages individual trader advantages:
- **Agility**: Quick adaptation to market changes
- **Small-scale focus**: Opportunities too small for institutional capital
- **Hypothesis-driven**: Systematic testing of market inefficiencies
- **End-to-end automation**: From data processing to live trading

![Current approach](images/current_approach_2025-05-27.png)

## System Capabilities ⚡

**🔄 Complete Trading Pipeline**: Data → Features → ML Models → Backtesting → Live Trading  
**🧠 Hypothesis Testing**: Systematic exploration of market inefficiencies  
**🚀 GPU-Accelerated ML**: XGBoost, Random Forest, PyTorch, TensorFlow support  
**📊 Advanced Backtesting**: Historical simulation with realistic costs and slippage  
**📡 Live Trading**: Real-time execution via Capital.com API with safety controls  
**🎯 Risk Management**: Multi-layer position sizing, stop-losses, drawdown protection

## System Components 📁

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| **[`hypothesis/`](hypothesis/README.md)** | Market hypothesis testing | Configurable trading strategies, pattern definitions |
| **[`data/`](data/README.md)** | Data pipeline | Live streaming, feature generation (200+ features) |
| **[`model/`](model/README.md)** | ML training system | GPU acceleration, multiple frameworks, walk-forward validation |
| **[`backtesting/`](backtesting/README.md)** | Historical simulation | Realistic costs, component reuse, comprehensive metrics |
| **[`strategy/`](strategy/README.md)** | Live trading system | Real-time execution, risk management, WebSocket integration |
| **[`evaluation/`](evaluation/README.md)** | Performance analysis | Model evaluation, trading metrics, visualization |
| **[`broker/`](broker/README.md)** | Market connectivity | Capital.com API, order management, account switching |

> [!TIP]
> Each component has detailed documentation in its respective README.md file. Start with `hypothesis/README.md` to understand the system's approach.

## Setup repo 🛠️
If this is the first time using the repo, follow these instructions:
1. Clone the environment (eg.): ``` git clone git@github.com:DanielWestgaard/ml-pred.git ```
2. Create a virtual environment (in root of project): ``` python3.11 -m venv venv ```
    - Recommended Python version: 3.11
    - Make sure to activate it: ```source venv/bin/activate```
3. Install dependencies: ``` pip install -r requirements.txt ```
4. Create secrets path: ``` mkdir secrets; cd secrets; touch secrets.txt; cd .. ```. Then, fill in the following (based on what you will use in the repo):
    - **For using Capital.com's [API](https://open-api.capital.com/)**: 
        1. Make sure you're following these [getting started steps](https://open-api.capital.com/#section/Getting-started).
        2. Paste in these variables in the secrets-file:
            ```
            API_KEY_CAP=
            PASSWORD_CAP=
            EMAIL=
            ```
        3. Fill in the values with from your own properties.
    - **For using [Alpha Vantage API](https://www.alphavantage.co/documentation/)**:
        1. Create your own [API token](https://www.alphavantage.co/support/#api-key).
        2. Paste in this variable in the secrets-file: ```alpha_vantage_free_api_key=```.
        3. Fill in your own API token for its value.
### Recommended, but not needed
- If you're using VSCode, I highly recommend installing these extensions:
    - [Excalidraw](https://marketplace.visualstudio.com/items?itemName=pomdtr.excalidraw-editor): an open-source virtual whiteboard tool designed for collaborative, freehand drawing and diagramming. This extension allows you to create and draw in VSCode, and view much more easily. Extension ID: pomdtr.excalidraw-editor
    - [Rainbow CSV](https://marketplace.visualstudio.com/items/?itemName=mechatroner.rainbow-csv): Prettier view of .csv-files.
    - [Material Icon Theme](https://marketplace.visualstudio.com/items/?itemName=PKief.material-icon-theme): Helps massively when you have many folders (icon some names).
- `gitstats` cmd-line tool (already in `requirements.txt`): Simple and easy tool to get statistics about your repository. Eg: `gitstats /Users/dmartinsen/pers-prog/ml-pred /Users/dmartinsen/pers-prog/ml-pred`

## Quick Start 🚀

### 1. Complete Hypothesis Workflow
Run the entire pipeline from data processing to live trading:
```bash
python main.py -hyp 'hypothesis/spread_compression_patterns/config.yml'
```

### 2. Individual Components
Execute specific components with targeted flags:
```bash
# Data processing only
python main.py -eg --hypothesis 'hypothesis/config.yml'

# Model training only  
python main.py -train --hypothesis 'hypothesis/config.yml'

# Backtesting only
python main.py -backtest --hypothesis 'hypothesis/config.yml'

# Live trading only
python main.py -strat 'hypothesis/config.yml'
```

### 3. Production-Ready Example
The spread compression strategy is tested and working-ish:
```bash
# Backtest with trained model
python main.py -hyp 'hypothesis/spread_compression_patterns/config_backtest.yml'

# Live trading (simulation mode)
python main.py -strat 'hypothesis/spread_compression_patterns/config_complete.yml'
```

> [!NOTE]  
> See component-specific README files for detailed configuration options and advanced usage.

## System Highlights 🌟

### Latest Improvements (2025)
- **🏗️ Unified Architecture**: Complete integration between backtesting and live trading systems
- **🚀 GPU Acceleration**: Automatic hardware optimization for ML training (CUDA/Metal/CPU)
- **📊 Advanced Backtesting**: 90% component reuse ensures identical behavior with live trading
- **🧠 Production Models**: Random Forest classifier with perfect OOB accuracy for spread compression
- **📡 Real-time Integration**: WebSocket streaming with Capital.com API for live market data

### Key Achievements
- **🎯 Hypothesis-Driven**: Systematic exploration of market inefficiencies
- **⚡ End-to-End Pipeline**: Seamless data → model → backtest → live trading workflow  
- **🛡️ Risk Management**: Multi-layer safety controls for live trading
- **📈 Proven Strategy**: Spread compression pattern detection with validated results

---

**For detailed component documentation, see individual README files in each directory.**