# Spread Compression Trading Strategy

**✅ VALIDATED HYPOTHESIS**: Use spread compression to predict price movements 15-75 minutes before they occur with **intelligent exit strategies**.

## 🎯 **Core Discovery**

When bid-ask spreads suddenly compress, it predicts significant price movements with **1.83x movement ratio** on 1-minute GBPUSD data. This represents a genuine market microstructure inefficiency - spread compression signals market maker expectations of increased activity.

**Key Validation (July 2025)**:
- **Compression signals**: 1.29x movement prediction (85,548 events)
- **Volume signals**: 1.52x movement prediction (strongest individual predictor)  
- **Combined multi-feature signals**: 1.83x movement prediction (4,673 premium events)
- **Critical insight**: 1-minute data essential (5-minute smooths out compression patterns)

## 📁 **Configuration Files**

### **🎯 Configurations (Current)**
The configuration files often vary, change, and gets removed, but are often marked with what they're targeting/doing. Eg.<br>
1. `hypothesis/spread_compression_patterns/config_process_data_v3.yml` for pure data processing (specifying target and features).
2. `hypothesis/spread_compression_patterns/config_model_training_smote.yml` for training a model on processed data.
3. `hypothesis/spread_compression_patterns/config_backtest_v3.yml` for backtesting a trained model.

## ⚙️ **Intelligent Exit Strategy Parameters**
> [!WARNING]
> These results are from very early in the project, and most likely don't hold up to the current progress this hypothesis has come to. `TODO`: find optimal parameters when model works.
### **🎯 Confidence Exit Threshold** (`confidence_exit_threshold`)
Exit when model confidence for opposite direction exceeds threshold.

| Setting | Value | Performance | Description |
|---------|-------|-------------|-------------|
| **Optimized** | `0.35` | **Best (-0.34%)** | Conservative exits, fewer false exits ⭐ |
| Baseline | `0.30` | -0.35% | Moderate sensitivity |
| Aggressive | `0.25` | -0.34% | Quick exits, more responsive |
| Conservative | `0.40` | Not tested | Very selective exits |

### **📉 Signal Strength Decay** (`signal_strength_decay`)
Exit when signal deteriorates by this percentage from peak.

| Setting | Value | Sensitivity | Description |
|---------|-------|-------------|-------------|
| **Optimized** | `0.17` | Enhanced | **Recommended** - Better decay detection |
| Baseline | `0.25` | Standard | Original setting |
| Aggressive | `0.15` | Very High | Quick response to deterioration |
| Conservative | `0.30` | Low | Allow more signal deterioration |

### **⏱️ Minimum Signal Life** (`minimum_signal_life`)
Minimum position holding time (minutes) to avoid noise.

| Setting | Value | Protection | Impact |
|---------|-------|------------|---------|
| **Optimized** | `5` | High | **Best** - Reduces premature exits |
| Baseline | `3` | Moderate | Standard protection |
| Aggressive | `2` | Low | More responsive to changes |

### **📊 Volatility Exit Multiplier** (`volatility_exit_multiplier`)
Exit when volatility exceeds this multiple of entry volatility.

| Setting | Value | Sensitivity | Use Case |
|---------|-------|-------------|----------|
| **Optimized** | `1.3` | High | **Best** - Early breakout detection |
| Baseline | `1.5` | Moderate | Standard monitoring |
| Conservative | `2.0` | Low | Allow higher volatility |

## 🚀 **Usage**

### **Quick Start**
Eg.
```bash
# Run optimized configuration (recommended)
python main.py --hyp 'hypothesis/spread_compression_patterns/config_backtest_improved_simple.yml'

# Run baseline for comparison
python main.py --hyp 'hypothesis/spread_compression_patterns/config_backtest.yml'
```

## 🔧 **Optimized Feature Set (5 Enhanced Hybrid Features)**
Came to this conclusion by testing: using feature importance, training model and backtesting it:
```python
'microstructure_momentum_fusion',
'depth_stability', 
'volume_ratio',
'institutional_footprint',
'top_predictor_interaction'
```

## 🎛️ **Signal Tracking for Intelligent Exits**
> [!WARNING]
> Old info.
The optimized configuration tracks these features for decay detection:
- `compression_ratio_micro` - Core compression measurement
- `volume_ratio` - Volume confirmation
- `signal_strength_v2` - Enhanced signal strength
- `compression_consistency` - Pattern consistency
- `volatility_short` - Volatility monitoring
- `spread_baseline` - Spread baseline comparison

## 📅 **Optimal Testing Periods**

High-signal periods identified by analysis:
1. **2025-04-07 to 2025-04-14** - Score: 40.4, 975 signals ⭐
2. **2025-04-14 to 2025-04-21** - Score: 35.2, 842 signals
3. **2025-03-31 to 2025-04-07** - Score: 32.8, 756 signals

Important **datasets used for training/backtesting** to ensure model is testing on unseen data:
- Training set: `storage/data/capital_com/GBPUSD/processed/1m/2024-01-01_2025-03-01.csv`
- Backtesting set: `storage/data/capital_com/GBPUSD/processed/1m/2025-03-02_2025-07-01.csv`

> [!NOTE]
> If the features you're focusing on change, **both** of these datasets must be processed again! That path is easy: their `raw` counterparts are at the equivalent place `storage/data/capital_com/GBPUSD/raw/`

## 🎯 **Target Distribution**

**Validated Target Distribution** (Training Focus):
- **Tier 0**: 402,311 events (no signal)
- **Tier 1**: 24,709 events (basic compression + movement) 
- **Tier 2**: 626 events (+ consistency) - **Highest quality (0.000842 avg movement)**
- **Tier 3+4**: 95 events (+ volume + London session)

## 📚 **Documentation**

- **Implementation Details:** `docs/SPREAD_COMPRESSION_IMPLEMENTATION_SUMMARY.md`
